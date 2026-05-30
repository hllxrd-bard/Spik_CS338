import torch
import torch.nn as nn
import torch.nn.functional as F

from model_det import SpikformerBackboneSingleScale
from yolox.head import YOLOXHead


class LightweightSpikformerPyramid(nn.Module):
    """
    Build a lightweight P3/P4/P5 pyramid from the final Spikformer feature map.

    Input:
        feat: [B, C, H/16, W/16]

    Output:
        P3: [B, C, H/8,  W/8]
        P4: [B, C, H/16, W/16]
        P5: [B, C, H/32, W/32]
    """

    def __init__(self, channels: int = 256):
        super().__init__()

        self.p3_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.p4_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.p5_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor):
        p3 = F.interpolate(feat, scale_factor=2, mode="nearest")
        p3 = self.p3_refine(p3)

        p4 = self.p4_refine(feat)

        p5 = self.p5_down(feat)

        return [p3, p4, p5]


class SpikformerYOLOXDetector(nn.Module):
    """
    Spikformer-Det with YOLOX-style detection head.

    This keeps the Spikformer event input convention:
        x: [B, T, 2, H, W]

    Then:
        Spikformer backbone -> spatial feature map -> lightweight pyramid -> YOLOXHead
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 2,
        embed_dims: int = 256,
        num_heads: int = 16,
        depths: int = 2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.backbone = SpikformerBackboneSingleScale(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=4,
            depths=depths,
            drop_path_rate=drop_path_rate,
            sr_ratios=1,
        )

        self.neck = LightweightSpikformerPyramid(channels=embed_dims)

        self.head = YOLOXHead(
            num_classes=num_classes,
            strides=(8, 16, 32),
            in_channels=(embed_dims, embed_dims, embed_dims),
            depthwise=False,
            act="silu",
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        feat = self.backbone(x)
        feats = self.neck(feat)
        outputs, losses = self.head(feats, labels=labels)
        return outputs, losses