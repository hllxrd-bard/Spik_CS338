from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import _cfg

from model import Block


class DynamicSPS(nn.Module):
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
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, 3, 1, 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, 3, 1, 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, 3, 1, 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def _conv_bn_lif_pool(self, x, conv, bn, lif, pool, T, B):
        x = conv(x)
        _, C, H, W = x.shape
        x = bn(x).reshape(T, B, C, H, W).contiguous()
        x = lif(x).flatten(0, 1).contiguous()
        x = pool(x)
        return x

    def forward(self, x):
        T, B, C, H, W = x.shape

        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(f"Input H,W must be divisible by 16, got {(H, W)}")

        x = x.flatten(0, 1)

        x = self._conv_bn_lif_pool(x, self.proj_conv, self.proj_bn, self.proj_lif, self.maxpool, T, B)
        x = self._conv_bn_lif_pool(x, self.proj_conv1, self.proj_bn1, self.proj_lif1, self.maxpool1, T, B)
        x = self._conv_bn_lif_pool(x, self.proj_conv2, self.proj_bn2, self.proj_lif2, self.maxpool2, T, B)
        x = self._conv_bn_lif_pool(x, self.proj_conv3, self.proj_bn3, self.proj_lif3, self.maxpool3, T, B)

        _, C_out, Hf, Wf = x.shape

        x_rpe = self.rpe_conv(x)
        x_rpe = self.rpe_bn(x_rpe).reshape(T, B, C_out, Hf, Wf).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0, 1)

        x = x + x_rpe
        x = x.reshape(T, B, C_out, Hf * Wf).contiguous()

        return x, Hf, Wf


class SpikformerBackboneSingleScale(nn.Module):
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
        self.patch_embed = DynamicSPS(in_channels=in_channels, embed_dims=embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.blocks = nn.ModuleList([
            Block(
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
        ])

    def forward(self, x):
        # x: [B, T, 2, H, W]
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # [T, B, C, H, W]

        tokens, Hf, Wf = self.patch_embed(x)

        for blk in self.blocks:
            tokens = blk(tokens)

        T, B, C, N = tokens.shape
        feat = tokens.reshape(T, B, C, Hf, Wf).contiguous()

        # Option A baseline: temporal average.
        feat = feat.mean(dim=0)  # [B, C, Hf, Wf]

        return feat


class YOLOLikeSingleScaleHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()

        hidden = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.pred = nn.Conv2d(hidden, 5 + num_classes, kernel_size=1)

    def forward(self, feat):
        x = self.stem(feat)
        return self.pred(x)


class SpikformerDetectorOptionA(nn.Module):
    def __init__(self, num_classes=2, in_channels=2, embed_dims=256):
        super().__init__()

        self.backbone = SpikformerBackboneSingleScale(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=16,
            mlp_ratios=4,
            depths=2,
            drop_path_rate=0.1,
            sr_ratios=1,
        )

        self.head = YOLOLikeSingleScaleHead(
            in_channels=embed_dims,
            num_classes=num_classes,
        )

    def forward(self, x):
        feat = self.backbone(x)
        pred = self.head(feat)
        return pred