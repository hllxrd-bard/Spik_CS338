from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.head import YOLOXHead
from yolox.network_blocks import BaseConv, CSPLayer, DWConv


def _to_3tuple_depths(depths: Union[int, Tuple[int, int, int]]):
    if isinstance(depths, int):
        return (depths, depths, depths)

    if isinstance(depths, str):
        parts = [int(x.strip()) for x in depths.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"depths string must have 3 ints, got {depths}")
        return tuple(parts)

    if len(depths) != 3:
        raise ValueError(f"depths must be int or 3-tuple, got {depths}")

    return tuple(int(d) for d in depths)


def _make_stage_heads(channels: Tuple[int, int, int], requested_heads: int) -> Tuple[int, int, int]:
    heads = []

    for c in channels:
        h = min(requested_heads, c)
        while h > 1 and c % h != 0:
            h -= 1
        heads.append(h)

    return tuple(heads)


class SpatialTransformerStage(nn.Module):
    """
    Transformer encoder over one spatial feature map.

    Input:
        x: [B, C, H, W]

    Output:
        x: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        grid_size: Tuple[int, int],
        depth: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.channels = channels
        self.grid_h, self.grid_w = grid_size
        self.num_patches = self.grid_h * self.grid_w

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, channels))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(channels)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _get_pos_embed(self, h: int, w: int):
        if h == self.grid_h and w == self.grid_w:
            return self.pos_embed

        # Interpolate positional embedding if input size changes.
        pos = self.pos_embed.reshape(1, self.grid_h, self.grid_w, self.channels)
        pos = pos.permute(0, 3, 1, 2).contiguous()
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.flatten(2).transpose(1, 2).contiguous()
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        if c != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {c}")

        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        x = x + self._get_pos_embed(h, w)

        x = self.blocks(x)
        x = self.norm(x)

        x = x.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return x


class ConvDownsampleStage(nn.Module):
    """
    Simple conv downsample stage.

    stride=2 halves spatial resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, act: str = "silu"):
        super().__init__()

        self.block = BaseConv(
            in_channels,
            out_channels,
            ksize=3,
            stride=2,
            act=act,
        )

    def forward(self, x):
        return self.block(x)


class MultiscaleViTBackbone(nn.Module):
    """
    Multiscale ViT-style backbone for detection.

    Input:
        x: [B, T, 2, H, W]

    Internally:
        [B, T, 2, H, W] -> [B, 2T, H, W]

    Output:
        P3, P4, P5 with strides 8, 16, 32.
    """

    def __init__(
        self,
        T: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        in_channels_per_timestep: int = 2,
        embed_dims: int = 256,
        num_heads: int = 8,
        depths: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act: str = "silu",
    ):
        super().__init__()

        self.T = int(T)
        self.in_channels_per_timestep = int(in_channels_per_timestep)
        self.in_chans = self.T * self.in_channels_per_timestep

        self.img_h, self.img_w = img_size

        if self.img_h % 32 != 0 or self.img_w % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")

        c1 = max(embed_dims // 8, 1)
        c2 = max(embed_dims // 4, 1)
        c3 = embed_dims // 2
        c4 = embed_dims
        c5 = embed_dims * 2

        self.out_channels = (c3, c4, c5)

        stage_depths = _to_3tuple_depths(depths)
        stage_heads = _make_stage_heads(self.out_channels, num_heads)

        # H/2, H/4, H/8
        self.stem1 = ConvDownsampleStage(self.in_chans, c1, act=act)
        self.stem2 = ConvDownsampleStage(c1, c2, act=act)
        self.stem3 = ConvDownsampleStage(c2, c3, act=act)

        # H/16, H/32
        self.stem4 = ConvDownsampleStage(c3, c4, act=act)
        self.stem5 = ConvDownsampleStage(c4, c5, act=act)

        grid3 = (self.img_h // 8, self.img_w // 8)
        grid4 = (self.img_h // 16, self.img_w // 16)
        grid5 = (self.img_h // 32, self.img_w // 32)

        self.stage3 = SpatialTransformerStage(
            channels=c3,
            grid_size=grid3,
            depth=stage_depths[0],
            num_heads=stage_heads[0],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.stage4 = SpatialTransformerStage(
            channels=c4,
            grid_size=grid4,
            depth=stage_depths[1],
            num_heads=stage_heads[1],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.stage5 = SpatialTransformerStage(
            channels=c5,
            grid_size=grid5,
            depth=stage_depths[2],
            num_heads=stage_heads[2],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected x as [B,T,C,H,W], got shape={tuple(x.shape)}")

        b, t, c, h, w = x.shape

        if t != self.T:
            raise ValueError(f"Expected T={self.T}, got T={t}")
        if c != self.in_channels_per_timestep:
            raise ValueError(f"Expected C={self.in_channels_per_timestep}, got C={c}")
        if h != self.img_h or w != self.img_w:
            raise ValueError(f"Expected input size {(self.img_h, self.img_w)}, got {(h, w)}")

        # [B, T, 2, H, W] -> [B, 2T, H, W]
        x = x.reshape(b, t * c, h, w).contiguous()

        x = self.stem1(x)   # H/2
        x = self.stem2(x)   # H/4

        x = self.stem3(x)   # H/8
        p3 = self.stage3(x)

        x = self.stem4(p3)  # H/16
        p4 = self.stage4(x)

        x = self.stem5(p4)  # H/32
        p5 = self.stage5(x)

        return p3, p4, p5


class YOLOPAFPNFromFeatures(nn.Module):
    """
    YOLOX PAFPN adapted to receive (P3, P4, P5) tensors directly.
    Same interface as the Spikformer version.
    """

    def __init__(
        self,
        depth: float = 1.0,
        in_channels: Tuple[int, int, int] = (128, 256, 512),
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError(f"YOLOPAFPNFromFeatures expects 3 input scales, got {in_channels}")

        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        n = max(round(3 * depth), 1)

        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1, act=act)
        self.C3_p4 = CSPLayer(2 * in_channels[1], in_channels[1], n, False, depthwise=depthwise, act=act)

        self.reduce_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1, act=act)
        self.C3_p3 = CSPLayer(2 * in_channels[0], in_channels[0], n, False, depthwise=depthwise, act=act)

        self.bu_conv2 = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        self.C3_n3 = CSPLayer(2 * in_channels[0], in_channels[1], n, False, depthwise=depthwise, act=act)

        self.bu_conv1 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.C3_n4 = CSPLayer(2 * in_channels[1], in_channels[2], n, False, depthwise=depthwise, act=act)

    @staticmethod
    def _upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, features: Sequence[torch.Tensor]):
        if len(features) != 3:
            raise ValueError(f"Expected three feature maps (P3,P4,P5), got {len(features)}")

        x2, x1, x0 = features

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self._upsample(fpn_out0, x1.shape[-2:])
        f_out0 = torch.cat([f_out0, x1], dim=1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self._upsample(fpn_out1, x2.shape[-2:])
        f_out1 = torch.cat([f_out1, x2], dim=1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)
        pan_out0 = self.C3_n4(p_out0)

        return pan_out2, pan_out1, pan_out0


class ViTYOLOXDetector(nn.Module):
    """
    Multiscale ViT-YOLOX detector.

    Input convention matches Spikformer:
        x: [B, T, 2, H, W]

    labels:
        [B, max_boxes, 5] in YOLOX format [class, cx, cy, w, h]
    """

    def __init__(
        self,
        num_classes: int = 3,
        T: int = 16,
        img_size: Tuple[int, int] = (256, 256),
        embed_dims: int = 256,
        depth: float = 1.0,
        depths: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()

        self.backbone = MultiscaleViTBackbone(
            T=T,
            img_size=img_size,
            in_channels_per_timestep=2,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
        )

        self.neck = YOLOPAFPNFromFeatures(
            depth=depth,
            in_channels=self.backbone.out_channels,
            depthwise=depthwise,
            act=act,
        )

        self.head = YOLOXHead(
            num_classes=num_classes,
            strides=(8, 16, 32),
            in_channels=self.backbone.out_channels,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        feats = self.backbone(x)
        feats = self.neck(feats)
        outputs, losses = self.head(feats, labels=labels)
        return outputs, losses