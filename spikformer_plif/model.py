import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from timm.models.layers import DropPath

from yolox.head import YOLOXHead


class MLPPLIF(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        t, b, c, n = x.shape
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(t, b, self.c_hidden, n).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(t, b, c, n).contiguous()
        x = self.fc2_lif(x)
        return x


class SSAPLIF(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        self.attn_lif = MultiStepParametricLIFNode(
            init_tau=2.0,
            v_threshold=0.5,
            detach_reset=True,
            backend="cupy",
        )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, x):
        t, b, c, n = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(t, b, c, n).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = (
            q_conv_out.transpose(-1, -2)
            .reshape(t, b, n, self.num_heads, c // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(t, b, c, n).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = (
            k_conv_out.transpose(-1, -2)
            .reshape(t, b, n, self.num_heads, c // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(t, b, c, n).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = (
            v_conv_out.transpose(-1, -2)
            .reshape(t, b, n, self.num_heads, c // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        attn = q @ k.transpose(-2, -1)
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(t, b, c, n).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(t, b, c, n))

        return x


class BlockPLIF(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSAPLIF(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPPLIF(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class DynamicSPSPLIF(nn.Module):
    """
    Dynamic version of SPS.

    Input:
        x: [T, B, C, H, W]

    Output:
        tokens: [T, B, embed_dims, Hf * Wf]
        Hf, Wf
    """

    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, 3, 1, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, 3, 1, 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, 3, 1, 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, 3, 1, 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

    def _conv_bn_lif_pool(self, x, conv, bn, lif, pool, t, b):
        x = conv(x)
        _, c, h, w = x.shape
        x = bn(x).reshape(t, b, c, h, w).contiguous()
        x = lif(x).flatten(0, 1).contiguous()
        x = pool(x)
        return x

    def forward(self, x):
        t, b, c, h, w = x.shape

        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"Input H,W must be divisible by 16, got {(h, w)}")

        x = x.flatten(0, 1)

        x = self._conv_bn_lif_pool(x, self.proj_conv, self.proj_bn, self.proj_lif, self.maxpool, t, b)
        x = self._conv_bn_lif_pool(x, self.proj_conv1, self.proj_bn1, self.proj_lif1, self.maxpool1, t, b)
        x = self._conv_bn_lif_pool(x, self.proj_conv2, self.proj_bn2, self.proj_lif2, self.maxpool2, t, b)
        x = self._conv_bn_lif_pool(x, self.proj_conv3, self.proj_bn3, self.proj_lif3, self.maxpool3, t, b)

        _, c_out, hf, wf = x.shape

        x_rpe = self.rpe_conv(x)
        x_rpe = self.rpe_bn(x_rpe).reshape(t, b, c_out, hf, wf).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0, 1)

        x = x + x_rpe
        x = x.reshape(t, b, c_out, hf * wf).contiguous()

        return x, hf, wf


class SpikformerBackboneSingleScalePLIF(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        num_heads=16,
        mlp_ratios=4,
        depths=2,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratios=1,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.patch_embed = DynamicSPSPLIF(in_channels=in_channels, embed_dims=embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.blocks = nn.ModuleList(
            [
                BlockPLIF(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for i in range(depths)
            ]
        )

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        tokens, hf, wf = self.patch_embed(x)

        for blk in self.blocks:
            tokens = blk(tokens)

        t, b, c, n = tokens.shape
        feat = tokens.reshape(t, b, c, hf, wf).contiguous()

        feat = feat.mean(dim=0)

        return feat


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


class SpikformerYOLOXDetectorPLIF(nn.Module):
    """
    Spikformer-Det with YOLOX-style detection head using PLIF neurons.

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

        self.backbone = SpikformerBackboneSingleScalePLIF(
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
