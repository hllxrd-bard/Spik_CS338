import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.head import YOLOXHead


class ViTBackboneSingleScale(nn.Module):
    """
    Minimal ViT backbone for event detection.

    Input:
        x: [B, T, 2, H, W]

    Internally:
        [B, T, 2, H, W] -> [B, 2T, H, W]

    Output:
        feat: [B, embed_dim, H/patch_size, W/patch_size]
    """

    def __init__(
        self,
        T=16,
        img_size=(256, 256),
        patch_size=16,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.T = T
        self.in_chans = 2 * T
        self.img_h, self.img_w = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        if self.img_h % patch_size != 0 or self.img_w % patch_size != 0:
            raise ValueError(f"img_size must be divisible by patch_size, got {img_size}, patch={patch_size}")

        self.grid_h = self.img_h // patch_size
        self.grid_w = self.img_w // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(
            self.in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, T, 2, H, W]
        B, T, C, H, W = x.shape

        if T != self.T:
            raise ValueError(f"Expected T={self.T}, got T={T}")
        if C != 2:
            raise ValueError(f"Expected polarity channels=2, got C={C}")
        if H != self.img_h or W != self.img_w:
            raise ValueError(f"Expected input size {(self.img_h, self.img_w)}, got {(H, W)}")

        # [B, T, 2, H, W] -> [B, 2T, H, W]
        x = x.reshape(B, T * C, H, W)

        # [B, embed_dim, Gh, Gw]
        x = self.patch_embed(x)

        Gh, Gw = x.shape[-2], x.shape[-1]

        # [B, C, Gh, Gw] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)

        # [B, N, C] -> [B, C, Gh, Gw]
        x = x.transpose(1, 2).reshape(B, self.embed_dim, Gh, Gw)

        return x


class LightweightViTPyramid(nn.Module):
    """
    Build P3/P4/P5 from one ViT feature map.

    Input:
        feat: [B, C, H/16, W/16]

    Output:
        P3: [B, C, H/8,  W/8]
        P4: [B, C, H/16, W/16]
        P5: [B, C, H/32, W/32]
    """

    def __init__(self, channels=256):
        super().__init__()

        self.p3_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.p4_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.p5_down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feat):
        p3 = F.interpolate(feat, scale_factor=2, mode="nearest")
        p3 = self.p3_refine(p3)

        p4 = self.p4_refine(feat)
        p5 = self.p5_down(feat)

        return [p3, p4, p5]


class ViTYOLOXDetector(nn.Module):
    """
    Controlled ANN ViT baseline:
        event tensor -> ViT backbone -> lightweight pyramid -> same YOLOXHead
    """

    def __init__(
        self,
        num_classes=2,
        T=16,
        img_size=(256, 256),
        patch_size=16,
        embed_dim=256,
        depth=4,
        num_heads=8,
    ):
        super().__init__()

        self.backbone = ViTBackboneSingleScale(
            T=T,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        self.neck = LightweightViTPyramid(channels=embed_dim)

        self.head = YOLOXHead(
            num_classes=num_classes,
            strides=(8, 16, 32),
            in_channels=(embed_dim, embed_dim, embed_dim),
            depthwise=False,
            act="silu",
        )

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        feats = self.neck(feat)
        outputs, losses = self.head(feats, labels=labels)
        return outputs, losses