"""Pareto-calibrated region detail successor built strictly on trained V10.

V12 keeps the complete V10 fusion image as a stable base and learns only a
bounded, source-derived residual.  Its zero-parameter starting rule was chosen
on held-out MSRS validation images: add 0.10 of the locally stronger source
detail.  A region router may then adapt this safe rule without ever producing
the unbounded feature drift that destabilized V11.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Ufuser_source_wave_assm_v10 import UfuserSourceWaveASSMV10


def _detail(image: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    return image - F.avg_pool2d(image, kernel, 1, kernel // 2)


class RegionParetoDetailHead(nn.Module):
    """Bounded region-adaptive correction with a validated analytic start."""

    def __init__(self, region_size=16, source_gain=0.05,
                 agreement_gain=0.15, fused_gain=0.10,
                 max_source_delta=0.05, max_fused_gain=0.10,
                 max_intensity_gain=0.01) -> None:
        super().__init__()
        self.region_size = max(1, int(region_size))
        self.source_gain = float(source_gain)
        self.agreement_gain = float(agreement_gain)
        self.fused_gain = float(fused_gain)
        self.max_source_delta = float(max_source_delta)
        self.max_fused_gain = float(max_fused_gain)
        self.max_intensity_gain = float(max_intensity_gain)
        self.router = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32,
                      padding_mode="reflect"),
            nn.GELU(), nn.Conv2d(32, 16, 1), nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )
        # Exact analytic safe rule at initialization.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, base, infrared, visible):
        detail_ir, detail_vi = _detail(infrared), _detail(visible)
        detail_base = _detail(base)
        magnitude_ir, magnitude_vi = detail_ir.abs(), detail_vi.abs()
        choose_ir = magnitude_ir >= magnitude_vi
        selected_detail = torch.where(choose_ir, detail_ir, detail_vi)
        selected_source = torch.where(choose_ir, infrared, visible)
        agreement = (1.0 - (detail_ir - detail_vi).abs()
                     / (magnitude_ir + magnitude_vi + 1.0 / 255.0)).clamp(0.0, 1.0)
        confidence = (magnitude_ir - magnitude_vi).abs()
        confidence = confidence / (confidence.mean((-2, -1), keepdim=True) + 1e-4)
        router_input = torch.cat((
            infrared, visible, base, detail_ir, detail_vi, detail_base,
            (infrared - visible).abs(), confidence,
            magnitude_ir, magnitude_vi,
        ), 1)
        height, width = base.shape[-2:]
        region = min(self.region_size, height, width)
        regions = F.avg_pool2d(router_input, region, region, ceil_mode=True)
        controls = F.interpolate(self.router(regions), (height, width),
                                 mode="bilinear", align_corners=False)
        source_delta, fused_control, intensity_control = controls.chunk(3, 1)
        source_gain = (self.source_gain + self.agreement_gain * agreement
                       + self.max_source_delta * torch.tanh(source_delta))
        fused_gain = self.fused_gain + self.max_fused_gain * torch.tanh(fused_control)
        intensity_gain = self.max_intensity_gain * torch.tanh(intensity_control)
        residual = (source_gain * selected_detail
                    + fused_gain * detail_base
                    + intensity_gain * (selected_source - base))
        output = (base + residual).clamp(0.0, 1.0)
        return output, {
            "source_gain": source_gain, "fused_gain": fused_gain,
            "intensity_gain": intensity_gain,
            "selected_detail": selected_detail, "safe_output":
            (base + (self.source_gain + self.agreement_gain * agreement)
             * selected_detail + self.fused_gain * detail_base).clamp(0.0, 1.0),
            "residual": residual,
        }


class UfuserV10RegionDetailV12(nn.Module):
    """Trained V10 plus a mandatory, bounded region-calibrated image head."""

    def __init__(self, d_state=16, num_tokens=9, num_classes=9,
                 initial_new=0.01, region_size=16, source_gain=0.05,
                 agreement_gain=0.15, fused_gain=0.10,
                 max_source_delta=0.05, max_fused_gain=0.10,
                 max_intensity_gain=0.01) -> None:
        super().__init__()
        self.v10 = UfuserSourceWaveASSMV10(
            d_state, num_tokens, num_classes, initial_new)
        self.head = RegionParetoDetailHead(
            region_size, source_gain, agreement_gain, fused_gain,
            max_source_delta, max_fused_gain,
            max_intensity_gain)
        self.register_buffer("head_strength", torch.tensor(1.0))

    def load_v10(self, state) -> None:
        self.v10.load_state_dict(state, strict=True)

    def set_head_strength(self, strength: float) -> None:
        self.head_strength.fill_(min(1.0, max(0.0, float(strength))))

    def forward(self, infrared, visible, return_aux=False):
        base = self.v10(infrared, visible)
        enhanced, auxiliary = self.head(base, infrared, visible)
        fused = base + self.head_strength * (enhanced - base)
        outputs = {"fused": fused, "base": base, **auxiliary}
        return outputs if return_aux else fused


__all__ = ["UfuserV10RegionDetailV12", "RegionParetoDetailHead"]
