from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from yolox.head import YOLOXHead
from yolox.network_blocks import BaseConv, CSPLayer, DWConv


class SpikingMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c, n = x.shape

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(t, b, self.hidden_features, n).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(t, b, self.out_features, n).contiguous()
        x = self.fc2_lif(x)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, scale: float = 0.25):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(
            tau=2.0,
            v_threshold=0.5,
            detach_reset=True,
            backend="cupy",
        )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def _project(self, x: torch.Tensor, conv: nn.Conv1d, bn: nn.BatchNorm1d, lif: MultiStepLIFNode) -> torch.Tensor:
        t, b, c, n = x.shape
        out = conv(x.flatten(0, 1))
        out = bn(out).reshape(t, b, c, n).contiguous()
        out = lif(out)
        out = out.transpose(-1, -2).reshape(t, b, n, self.num_heads, self.head_dim)
        return out.permute(0, 1, 3, 2, 4).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c, n = x.shape

        q = self._project(x, self.q_conv, self.q_bn, self.q_lif)
        k = self._project(x, self.k_conv, self.k_bn, self.k_lif)
        v = self._project(x, self.v_conv, self.v_bn, self.v_lif)

        if n > self.head_dim:
            kv = k.transpose(-2, -1) @ v
            x = q @ kv
        else:
            attn = q @ k.transpose(-2, -1)
            x = attn @ v

        x = x.transpose(3, 4).reshape(t, b, c, n).contiguous()
        x = self.attn_lif(x * self.scale)

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(t, b, c, n).contiguous()
        x = self.proj_lif(x)
        return x


class BatchDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class SpikformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attn = SpikingSelfAttention(dim=dim, num_heads=num_heads)
        self.drop_path = BatchDropPath(drop_path)
        self.mlp = SpikingMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class SpikingConvPoolStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b = x.shape[:2]
        x = self.conv(x.flatten(0, 1))
        _, c, h, w = x.shape
        x = self.bn(x).reshape(t, b, c, h, w).contiguous()
        x = self.lif(x).flatten(0, 1).contiguous()
        x = self.pool(x)
        _, c, h, w = x.shape
        return x.reshape(t, b, c, h, w).contiguous()


class SpikeRelativePositionEmbedding(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        rpe = self.conv(x.flatten(0, 1))
        rpe = self.bn(rpe).reshape(t, b, c, h, w).contiguous()
        rpe = self.lif(rpe)
        return x + rpe


class SpikformerFeatureStage(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        drop_path_rates: Sequence[float],
    ):
        super().__init__()
        self.rpe = SpikeRelativePositionEmbedding(channels)
        self.blocks = nn.ModuleList(
            [
                SpikformerBlock(
                    dim=channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        x = self.rpe(x)
        x = x.reshape(t, b, c, h * w).contiguous()
        for block in self.blocks:
            x = block(x)
        x = x.reshape(t, b, c, h, w).contiguous()
        return x.mean(dim=0)


class MultiscaleSpikformerBackbone(nn.Module):
    """
    Spikformer-style event backbone for detection.

    Input:  [B, T, C, H, W]
    Output: (P3, P4, P5) with strides (8, 16, 32).
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dims: int = 256,
        num_heads: int = 16,
        depths: Union[int, Tuple[int, int, int]] = 2,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        c1 = max(embed_dims // 8, 1)
        c2 = max(embed_dims // 4, 1)
        c3 = embed_dims // 2
        c4 = embed_dims
        c5 = embed_dims * 2

        self.out_channels = (c3, c4, c5)

        if isinstance(depths, int):
            stage_depths = (depths, depths, depths)
        else:
            if len(depths) != 3:
                raise ValueError(f"depths must be int or 3-tuple, got {depths}")
            stage_depths = tuple(int(d) for d in depths)

        stage_heads = self._make_stage_heads(self.out_channels, num_heads)
        total_depth = sum(stage_depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist() if total_depth > 0 else []
        dpr3 = dpr[:stage_depths[0]]
        dpr4 = dpr[stage_depths[0]:stage_depths[0] + stage_depths[1]]
        dpr5 = dpr[stage_depths[0] + stage_depths[1]:]

        self.sps1 = SpikingConvPoolStage(in_channels, c1)
        self.sps2 = SpikingConvPoolStage(c1, c2)
        self.sps3 = SpikingConvPoolStage(c2, c3)
        self.sps4 = SpikingConvPoolStage(c3, c4)
        self.sps5 = SpikingConvPoolStage(c4, c5)

        self.stage3 = SpikformerFeatureStage(c3, stage_heads[0], stage_depths[0], mlp_ratio, dpr3)
        self.stage4 = SpikformerFeatureStage(c4, stage_heads[1], stage_depths[1], mlp_ratio, dpr4)
        self.stage5 = SpikformerFeatureStage(c5, stage_heads[2], stage_depths[2], mlp_ratio, dpr5)

    @staticmethod
    def _make_stage_heads(channels: Tuple[int, int, int], requested_heads: int) -> Tuple[int, int, int]:
        heads = []
        for c in channels:
            h = min(requested_heads, c)
            while h > 1 and c % h != 0:
                h -= 1
            heads.append(h)
        return tuple(heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected x as [B,T,C,H,W], got shape={tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(f"Input H,W must be divisible by 32 for P3/P4/P5, got {(h, w)}")

        x = x.permute(1, 0, 2, 3, 4).contiguous()
        x = self.sps1(x)
        x = self.sps2(x)

        x = self.sps3(x)
        p3 = self.stage3(x)

        x = self.sps4(x)
        p4 = self.stage4(x)

        x = self.sps5(x)
        p5 = self.stage5(x)

        return p3, p4, p5


class YOLOPAFPNFromFeatures(nn.Module):
    """YOLOX PAFPN adapted to receive (P3, P4, P5) tensors directly."""

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

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class SpikformerYOLOXDetector(nn.Module):
    """
    Multiscale Spikformer detector with YOLOX-style PAFPN/head.

    Input convention is unchanged:
        x: [B, T, 2, H, W]
        labels: [B, max_boxes, 5] in YOLOX format [class, cx, cy, w, h]
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 2,
        embed_dims: int = 256,
        num_heads: int = 16,
        depths: Union[int, Tuple[int, int, int]] = 2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.backbone = MultiscaleSpikformerBackbone(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
        )

        self.neck = YOLOPAFPNFromFeatures(
            depth=1.0,
            in_channels=self.backbone.out_channels,
            depthwise=False,
            act="silu",
        )

        self.head = YOLOXHead(
            num_classes=num_classes,
            strides=(8, 16, 32),
            in_channels=self.backbone.out_channels,
            depthwise=False,
            act="silu",
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        feats = self.backbone(x)
        feats = self.neck(feats)
        outputs, losses = self.head(feats, labels=labels)
        return outputs, losses
