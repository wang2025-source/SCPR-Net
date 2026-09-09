"""EMMA-SAFM V5: source-attentive frequency/state-space fusion.

The pretrained EMMA is kept intact.  Shallow source-image cross attention and
hierarchical spatial/global/local-frequency features are injected into the
decoder, while Joint ASSM remains responsible for deep global interaction.
Every decoder adapter is zero initialized, so V5 starts from EMMA exactly.

The source-query attention is a compact linear-complexity adaptation of SIBA
(ICCV 2025).  The spatial/global/local Fourier stratification is adapted from
HFIN (CVPR 2024); it is not a verbatim reimplementation of either full model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.assm_mambairv2 import ASSMResidual
from nets.joint_assm_v3 import CommonPrivateAdapter, JointCrossModalASSM
from nets.ship_high_order import SHIPHighOrderInteraction
from nets.Ufuser import Ufuser
from nets.Ufuser_assm_ship_task_v2 import LocalCNNBlockV2, ResizeConv
from nets.Ufuser_joint_assm_ship_v3 import DeepGuidedSkip, LightPrivateFusion


class SourceBoostSuppress(nn.Module):
    """CBSM-inspired raw-source conditioning without copying SIBA's backbone."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels,
                      padding_mode="reflect"),
        )
        hidden = max(4, channels // 4)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, source: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if source.shape[-2:] != size:
            source = F.interpolate(source, size=size, mode="bilinear", align_corners=False)
        x = self.embed(source)
        spatial = self.spatial(torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1))
        return x + x * self.channel(x) * spatial


class LinearSourceCrossAttention(nn.Module):
    """Raw source is Q; the opposite-modality feature supplies K and V."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      padding_mode="reflect"),
        )

    def forward(self, source_query: torch.Tensor, opposite: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = opposite.shape
        q = self.q(source_query).flatten(2).transpose(1, 2).softmax(dim=-1)
        k = self.k(opposite).flatten(2).transpose(1, 2).softmax(dim=1)
        v = self.v(opposite).flatten(2).transpose(1, 2)
        context = torch.bmm(k.transpose(1, 2), v)
        out = torch.bmm(q, context).transpose(1, 2).reshape(batch, channels, height, width)
        return self.project(out)


class BidirectionalSourceAttention(nn.Module):
    """I-SCA/V-SCA-inspired bidirectional source-to-feature retrieval."""

    def __init__(self, channels: int, scale_init: float = 5e-2) -> None:
        super().__init__()
        self.ir_source = SourceBoostSuppress(channels)
        self.vi_source = SourceBoostSuppress(channels)
        self.ir_queries_visible = LinearSourceCrossAttention(channels)
        self.vi_queries_infrared = LinearSourceCrossAttention(channels)
        self.ir_scale = nn.Parameter(torch.full((1, channels, 1, 1), scale_init))
        self.vi_scale = nn.Parameter(torch.full((1, channels, 1, 1), scale_init))

    def forward(
        self, raw_ir: torch.Tensor, raw_vi: torch.Tensor,
        ir: torch.Tensor, vi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = ir.shape[-2:]
        q_ir = self.ir_source(raw_ir, size)
        q_vi = self.vi_source(raw_vi, size)
        # IR source retrieves visible texture; VI source retrieves IR saliency.
        visible = vi + self.vi_scale * self.ir_queries_visible(q_ir, vi)
        infrared = ir + self.ir_scale * self.vi_queries_infrared(q_vi, ir)
        return infrared, visible


class GlobalFourierFusion(nn.Module):
    """Separately mix magnitude and unit-phase in the global Fourier domain."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.magnitude_gate = nn.Conv2d(2 * channels, channels, 1)
        self.phase_gate = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        dtype = infrared.dtype
        fi = torch.fft.rfft2(infrared.float(), norm="ortho")
        fv = torch.fft.rfft2(visible.float(), norm="ortho")
        mi, mv = fi.abs(), fv.abs()
        gate_m = torch.sigmoid(self.magnitude_gate(torch.cat((torch.log1p(mi), torch.log1p(mv)), 1)))
        gate_p = torch.sigmoid(self.phase_gate(torch.cat((torch.log1p(mi), torch.log1p(mv)), 1)))
        magnitude = gate_m * mi + (1.0 - gate_m) * mv
        unit_i = fi / mi.clamp_min(1e-6)
        unit_v = fv / mv.clamp_min(1e-6)
        unit = gate_p * unit_i + (1.0 - gate_p) * unit_v
        unit = unit / unit.abs().clamp_min(1e-6)
        return torch.fft.irfft2(magnitude * unit, s=infrared.shape[-2:], norm="ortho").to(dtype)


class LocalFourierFusion(nn.Module):
    """Overlap-window Fourier fusion with fold normalization."""

    def __init__(self, channels: int, window: int = 8) -> None:
        super().__init__()
        self.window = window
        self.stride = window // 2
        self.gate = nn.Conv1d(2 * channels, channels, 1)

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = infrared.shape
        window = min(self.window, height, width)
        stride = max(1, window // 2)
        if height < 2 or width < 2:
            return 0.5 * (infrared + visible)
        win1 = torch.hann_window(window, periodic=False, device=infrared.device,
                                dtype=torch.float32).clamp_min(0.1)
        win2 = torch.outer(win1, win1)

        def patches(x: torch.Tensor) -> torch.Tensor:
            u = F.unfold(x.float(), kernel_size=window, stride=stride)
            return u.reshape(batch, channels, window, window, -1).permute(0, 1, 4, 2, 3)

        pi, pv = patches(infrared) * win2, patches(visible) * win2
        fi = torch.fft.rfft2(pi, norm="ortho")
        fv = torch.fft.rfft2(pv, norm="ortho")
        energy_i = torch.log1p(fi.abs()).mean(dim=(-1, -2))
        energy_v = torch.log1p(fv.abs()).mean(dim=(-1, -2))
        gate = torch.sigmoid(self.gate(torch.cat((energy_i, energy_v), 1)))
        fused = torch.fft.irfft2(
            gate[..., None, None] * fi + (1.0 - gate[..., None, None]) * fv,
            s=(window, window), norm="ortho",
        ) * win2
        fused = fused.permute(0, 1, 3, 4, 2).reshape(batch, channels * window * window, -1)
        output = F.fold(fused, (height, width), kernel_size=window, stride=stride)
        count = fused.shape[-1]
        norm = (win2.square().reshape(1, window * window, 1)
                .expand(batch, window * window, count))
        norm = F.fold(norm, (height, width), kernel_size=window, stride=stride)
        return (output / norm.clamp_min(1e-4)).to(infrared.dtype)


class HierarchicalFrequencyFusion(nn.Module):
    """Spatial + global Fourier + local Fourier stratification and integration."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = LocalCNNBlockV2(4 * channels, channels)
        self.global_fourier = GlobalFourierFusion(channels)
        self.local_fourier = LocalFourierFusion(channels)
        self.integrate = nn.Sequential(
            nn.Conv2d(3 * channels, 2 * channels, 1), nn.GELU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(torch.cat((infrared, visible, infrared * visible,
                                          torch.abs(infrared - visible)), 1))
        global_frequency = self.global_fourier(infrared, visible)
        local_frequency = self.local_fourier(infrared, visible)
        return self.integrate(torch.cat((spatial, global_frequency, local_frequency), 1))


class ShallowSourceFrequencyDetail(nn.Module):
    def __init__(self, ship_order: int = 4, scale_init: float = 5e-2) -> None:
        super().__init__()
        self.source1 = BidirectionalSourceAttention(8, scale_init)
        self.source2 = BidirectionalSourceAttention(16, scale_init)
        self.frequency1 = HierarchicalFrequencyFusion(8)
        self.frequency2 = HierarchicalFrequencyFusion(16)
        self.ship2 = SHIPHighOrderInteraction(16, order=ship_order)
        nn.init.constant_(self.ship2.spatial.gamma, scale_init)
        nn.init.constant_(self.ship2.channel.gamma, scale_init)
        self.detail1 = LocalCNNBlockV2(3 * 8, 8)
        self.detail2 = LocalCNNBlockV2(5 * 16, 16)

    def forward(self, raw_ir, raw_vi, i1, v1, i2, v2):
        i1a, v1a = self.source1(raw_ir, raw_vi, i1, v1)
        i2a, v2a = self.source2(raw_ir, raw_vi, i2, v2)
        frequency1 = self.frequency1(i1a, v1a)
        frequency2 = self.frequency2(i2a, v2a)
        ship_vi, ship_ir = self.ship2(v2a, i2a)
        detail1 = self.detail1(torch.cat((i1a, v1a, frequency1), 1))
        detail2 = self.detail2(torch.cat((i2a, v2a, ship_ir, ship_vi, frequency2), 1))
        return {"detail1": detail1, "detail2": detail2,
                "frequency1": frequency1, "frequency2": frequency2}


class DeepJointStateSpace(nn.Module):
    def __init__(self, d_state=16, num_tokens=8, layer_scale_init=5e-2,
                 num_classes=9) -> None:
        super().__init__()
        channels = 32
        self.joint3 = JointCrossModalASSM(channels, d_state, num_tokens,
                                          layer_scale_init=layer_scale_init)
        self.fuse3 = LocalCNNBlockV2(3 * channels, channels)
        self.joint4 = JointCrossModalASSM(channels, d_state, num_tokens,
                                          layer_scale_init=layer_scale_init)
        self.decompose4 = CommonPrivateAdapter(channels)
        self.private4 = LightPrivateFusion(channels)
        self.fuse4 = LocalCNNBlockV2(2 * channels, channels)
        self.bottleneck = ASSMResidual(channels, d_state, num_tokens)
        nn.init.constant_(self.bottleneck.scale1, layer_scale_init)
        nn.init.constant_(self.bottleneck.scale2, layer_scale_init)
        self.up4 = ResizeConv(channels, channels)
        self.guided3 = DeepGuidedSkip(channels, LocalCNNBlockV2(2 * channels, channels))
        self.semantic_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, i3, v3, i4, v4, output_size):
        i3j, v3j, c3 = self.joint3(i3, v3)
        global3_skip = self.fuse3(torch.cat((i3j, v3j, c3), 1))
        i4j, v4j, c4 = self.joint4(i4, v4)
        split4 = self.decompose4(i4j, v4j, c4)
        split4["source_ir"], split4["source_vi"] = i4j, v4j
        private = self.private4(split4["private_ir"], split4["private_vi"])
        global4 = self.bottleneck(self.fuse4(torch.cat((split4["common"], private), 1)))
        global3 = self.guided3(self.up4(global4), global3_skip)
        semantic = F.interpolate(self.semantic_head(split4["common"]), size=output_size,
                                 mode="bilinear", align_corners=False)
        return {"global4": global4, "global3": global3,
                "scale4": split4, "semantic_logits": semantic}


class ZeroDecoderAdapter(nn.Module):
    """Feature adapter whose initial output is exactly zero."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(in_channels, out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UfuserSAFMV5(nn.Module):
    """Full EMMA plus scale-specific source/frequency/state-space injection."""

    def __init__(self, d_state=16, num_tokens=8, ship_order=4,
                 layer_scale_init=5e-2, num_classes=9) -> None:
        super().__init__()
        self.base = Ufuser()
        self.shallow = ShallowSourceFrequencyDetail(ship_order, layer_scale_init)
        self.deep = DeepJointStateSpace(d_state, num_tokens, layer_scale_init, num_classes)
        self.inject4 = ZeroDecoderAdapter(32, 32)
        self.inject3 = ZeroDecoderAdapter(32, 32)
        self.inject2 = ZeroDecoderAdapter(16, 16)
        self.inject1 = ZeroDecoderAdapter(8, 8)
        self.logit_detail = ZeroDecoderAdapter(8, 1)

    def load_emma(self, state: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    def _encode(self, infrared: torch.Tensor, visible: torch.Tensor) -> dict[str, torch.Tensor]:
        b = self.base
        i1 = b.I_en_1(infrared); i2 = b.I_en_2(b.I_down1(i1))
        i3 = b.I_en_3(b.I_down2(i2)); i4 = b.I_en_4(b.I_down3(i3))
        v1 = b.V_en_1(visible); v2 = b.V_en_2(b.V_down1(v1))
        v3 = b.V_en_3(b.V_down2(v2)); v4 = b.V_en_4(b.V_down3(v3))
        f1 = b.f_1(torch.cat((i1, v1), 1)); f2 = b.f_2(torch.cat((i2, v2), 1))
        f3 = b.f_3(torch.cat((i3, v3), 1)); f4 = b.f_4(torch.cat((i4, v4), 1))
        return {"i1": i1, "i2": i2, "i3": i3, "i4": i4,
                "v1": v1, "v2": v2, "v3": v3, "v4": v4,
                "f1": f1, "f2": f2, "f3": f3, "f4": f4}

    def _decode_base(self, f: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.base
        d4 = b.de_4(f["f4"])
        d3 = b.de_3(torch.cat((b.up4(d4), f["f3"]), 1))
        d2 = b.de_2(torch.cat((b.up3(d3), f["f2"]), 1))
        d1 = b.de_1(torch.cat((b.up2(d2), f["f1"]), 1))
        logits = b.last[0](d1)
        return d1, logits

    def forward_features(self, infrared: torch.Tensor, visible: torch.Tensor):
        f = self._encode(infrared, visible)
        shallow = self.shallow(infrared, visible, f["i1"], f["v1"], f["i2"], f["v2"])
        deep = self.deep(f["i3"], f["v3"], f["i4"], f["v4"], infrared.shape[-2:])
        b = self.base
        d4 = b.de_4(f["f4"]) + self.inject4(deep["global4"])
        d3 = b.de_3(torch.cat((b.up4(d4), f["f3"]), 1)) + self.inject3(deep["global3"])
        d2 = b.de_2(torch.cat((b.up3(d3), f["f2"]), 1)) + self.inject2(shallow["detail2"])
        d1 = b.de_1(torch.cat((b.up2(d2), f["f1"]), 1)) + self.inject1(shallow["detail1"])
        logits = b.last[0](d1) + self.logit_detail(shallow["detail1"])
        fused = torch.sigmoid(logits)
        with torch.no_grad():
            _, base_logits = self._decode_base({key: value.detach() for key, value in f.items()
                                                if key.startswith("f")})
            base_output = torch.sigmoid(base_logits)
        return {"fused": fused, "base": base_output,
                "logit_delta": logits - base_logits.detach(),
                "scale4": deep["scale4"], "semantic_logits": deep["semantic_logits"],
                **shallow, "global3": deep["global3"], "global4": deep["global4"]}

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor,
                return_aux: bool = False):
        outputs = self.forward_features(infrared, visible)
        return outputs if return_aux else outputs["fused"]
