"""EMMA Pareto-Wavelet V6.

Evidence-driven successor to V5: source attention is restricted to the deepest
low-frequency features; Haar wavelets manipulate only high-frequency bands;
SHIP and decoder feature injections are removed.  Both branches meet once at a
single bounded pre-sigmoid logit correction, initialized exactly to zero.

Wavelet high-frequency guidance follows the separation/invertibility principle
used by HDW-SR (CVPR 2026).  Deep raw-source querying follows SIBA (ICCV 2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Ufuser import Ufuser
from nets.Ufuser_assm_ship_task_v2 import ResizeConv
from nets.Ufuser_safm_v5 import BidirectionalSourceAttention, DeepJointStateSpace


def haar_dwt(x: torch.Tensor):
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a + b + c + d) * 0.5,
            (-a - b + c + d) * 0.5,
            (-a + b - c + d) * 0.5,
            (a - b - c + d) * 0.5)


def haar_iwt(ll, lh, hl, hh):
    a = (ll - lh - hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5
    output = torch.empty(ll.shape[0], ll.shape[1], ll.shape[2] * 2,
                         ll.shape[3] * 2, device=ll.device, dtype=ll.dtype)
    output[..., 0::2, 0::2] = a; output[..., 0::2, 1::2] = b
    output[..., 1::2, 0::2] = c; output[..., 1::2, 1::2] = d
    return output


class HighBandSelector(nn.Module):
    """Select source coefficients independently for LH/HL/HH bands."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(9, hidden, 3, padding=1, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(hidden, 6, 1),
        )

    def forward(self, ir_bands, vi_bands):
        ir = torch.cat(ir_bands, 1); vi = torch.cat(vi_bands, 1)
        logits = self.gate(torch.cat((ir, vi, torch.abs(ir - vi)), 1))
        weights = logits.reshape(logits.shape[0], 2, 3, *logits.shape[-2:]).softmax(1)
        weights = weights.unsqueeze(3)
        ir_stack = torch.stack(ir_bands, 1); vi_stack = torch.stack(vi_bands, 1)
        fused = weights[:, 0] * ir_stack + weights[:, 1] * vi_stack
        return tuple(fused[:, index] for index in range(3)), weights


class InvertibleHighFrequencyBranch(nn.Module):
    """Two-level perfect-reconstruction Haar path with no learned low-frequency rewrite."""

    def __init__(self) -> None:
        super().__init__()
        self.level1 = HighBandSelector()
        self.level2 = HighBandSelector()
        self.refine = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(8, 8, 3, padding=1, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor):
        lli, *hi1 = haar_dwt(infrared); llv, *hv1 = haar_dwt(visible)
        lli2, *hi2 = haar_dwt(lli); llv2, *hv2 = haar_dwt(llv)
        fused2, weights2 = self.level2(tuple(hi2), tuple(hv2))
        mid_detail = haar_iwt(torch.zeros_like(lli2), *fused2)
        fused1, weights1 = self.level1(tuple(hi1), tuple(hv1))
        detail = haar_iwt(mid_detail, *fused1)
        detail = self.refine(detail)
        # Enforce a local zero-mean correction: this branch cannot rewrite illumination.
        detail = detail - F.avg_pool2d(detail, 17, stride=1, padding=8)
        return detail, {"level1": weights1, "level2": weights2}


class LowFrequencyJointBranch(nn.Module):
    """Deep source attention + Joint ASSM; output is explicitly low-pass."""

    def __init__(self, d_state=16, num_tokens=8, layer_scale_init=5e-2,
                 num_classes=9) -> None:
        super().__init__()
        self.source4 = BidirectionalSourceAttention(32, layer_scale_init)
        self.deep = DeepJointStateSpace(d_state, num_tokens, layer_scale_init, num_classes)
        self.up3 = ResizeConv(32, 16)
        self.up2 = ResizeConv(16, 8)
        self.head = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, raw_ir, raw_vi, i3, v3, i4, v4, output_size):
        i4a, v4a = self.source4(raw_ir, raw_vi, i4, v4)
        outputs = self.deep(i3, v3, i4a, v4a, output_size)
        candidate = self.head(self.up2(self.up3(outputs["global3"])))
        candidate = F.avg_pool2d(candidate, 9, stride=1, padding=4)
        outputs["candidate"] = candidate
        return outputs


class UfuserParetoV6(nn.Module):
    """Frozen EMMA base with one bounded logit correction point."""

    def __init__(self, d_state=16, num_tokens=8, layer_scale_init=5e-2,
                 num_classes=9, max_logit_scale=0.25) -> None:
        super().__init__()
        self.base = Ufuser()
        self.low_branch = LowFrequencyJointBranch(
            d_state, num_tokens, layer_scale_init, num_classes
        )
        self.high_branch = InvertibleHighFrequencyBranch()
        self.global_scale_logit = nn.Parameter(torch.zeros(1))
        self.high_scale_logit = nn.Parameter(torch.zeros(1))
        self.max_logit_scale = float(max_logit_scale)

    def load_emma(self, state):
        self.base.load_state_dict(state, strict=True)

    def _base_features(self, infrared, visible):
        b = self.base
        i1 = b.I_en_1(infrared); i2 = b.I_en_2(b.I_down1(i1))
        i3 = b.I_en_3(b.I_down2(i2)); i4 = b.I_en_4(b.I_down3(i3))
        v1 = b.V_en_1(visible); v2 = b.V_en_2(b.V_down1(v1))
        v3 = b.V_en_3(b.V_down2(v2)); v4 = b.V_en_4(b.V_down3(v3))
        f1 = b.f_1(torch.cat((i1, v1), 1)); f2 = b.f_2(torch.cat((i2, v2), 1))
        f3 = b.f_3(torch.cat((i3, v3), 1)); f4 = b.f_4(torch.cat((i4, v4), 1))
        d4 = b.de_4(f4); d3 = b.de_3(torch.cat((b.up4(d4), f3), 1))
        d2 = b.de_2(torch.cat((b.up3(d3), f2), 1))
        d1 = b.de_1(torch.cat((b.up2(d2), f1), 1))
        base_logits = b.last[0](d1)
        return {"i3": i3, "v3": v3, "i4": i4, "v4": v4,
                "base_logits": base_logits}

    def forward_features(self, infrared, visible):
        features = self._base_features(infrared, visible)
        low = self.low_branch(infrared, visible, features["i3"], features["v3"],
                              features["i4"], features["v4"], infrared.shape[-2:])
        high, high_weights = self.high_branch(infrared, visible)
        global_scale = self.max_logit_scale * torch.tanh(self.global_scale_logit)
        high_scale = self.max_logit_scale * torch.tanh(self.high_scale_logit)
        global_delta = global_scale * torch.tanh(low["candidate"])
        high_delta = high_scale * torch.tanh(high)
        delta = global_delta + high_delta
        base = torch.sigmoid(features["base_logits"])
        fused = torch.sigmoid(features["base_logits"] + delta)
        return {"fused": fused, "base": base, "logit_delta": delta,
                "global_delta": global_delta, "high_delta": high_delta,
                "global_scale": global_scale, "high_scale": high_scale,
                "scale4": low["scale4"], "semantic_logits": low["semantic_logits"],
                "high_weights": high_weights}

    def forward(self, infrared, visible, return_aux=False):
        outputs = self.forward_features(infrared, visible)
        return outputs if return_aux else outputs["fused"]
