"""Source-conditioned wavelet ASSM fusion (V10).

V10 is deliberately a *main-path* successor to V9.  The new modules do not
produce a small output-side correction: they replace the four EMMA fusion
features and condition every decoder skip.  The design combines three
published ideas while keeping their roles separate:

* raw-source queries for shallow cross-modal interaction (SIBA, ICCV 2025);
* frequency-specific low/high processing (WaveMamba, ICCV 2025);
* attentive paired state-space fusion for deep low-frequency structure
  (MambaIRv2, CVPR 2025).

The implementation reuses the already audited local implementations of the
SIBA-style query block and paired attentive ASSM.  Existing V1--V9 files are
not modified.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Ufuser import Restormer_CNN_block, Ufuser
from nets.Ufuser_end2end_v7 import PairedSemanticASSMFusion, SIBAShallowFusion
from nets.Ufuser_pareto_assm_v8 import DeepSemanticHead, ExactCommonPrivateFusion
from nets.Ufuser_pareto_v6 import haar_dwt, haar_iwt


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


class MainPathBlend(nn.Module):
    """Learned, unbounded-in-training replacement of an EMMA fusion feature.

    Unlike V9's ``base + 0.25 * tanh(delta)``, this is a convex competition
    between the complete old and new paths.  A per-channel gate starts safely
    at ``initial_new`` but may move all the way to either path.
    """

    def __init__(self, channels: int, initial_new: float = 0.01) -> None:
        super().__init__()
        self.gate_logit = nn.Parameter(
            torch.full((1, channels, 1, 1), _logit(initial_new))
        )

    def forward(self, baseline: torch.Tensor, candidate: torch.Tensor):
        gate = torch.sigmoid(self.gate_logit)
        return baseline + gate * (candidate - baseline), gate


class SourcePrompt(nn.Module):
    """Embed a raw source image into a feature-scale conditioning map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, padding_mode="reflect"),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1,
                      padding_mode="reflect"),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 4), 1), nn.GELU(),
            nn.Conv2d(max(4, channels // 4), channels, 1), nn.Sigmoid(),
        )

    def forward(self, source: torch.Tensor, size: tuple[int, int]):
        source = F.interpolate(source, size=size, mode="bilinear",
                               align_corners=False)
        feature = self.body(source)
        return feature * self.channel(feature)


class ReliabilityHighBandFusion(nn.Module):
    """Source-conditioned, band- and channel-specific high-frequency routing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, 2 * channels)
        # IR bands + VI bands + absolute discrepancy + two source prompts.
        self.router = nn.Sequential(
            nn.Conv2d(11 * channels, hidden, 3, padding=1,
                      padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden,
                      padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(hidden, 6 * channels, 1),
        )
        # Equal source responsibility is a stable, non-suppressing start.
        nn.init.normal_(self.router[-1].weight, std=1e-3)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, ir_bands, vi_bands, prompt_ir, prompt_vi):
        height, width = ir_bands[0].shape[-2:]
        prompt_ir = F.interpolate(prompt_ir, (height, width), mode="bilinear",
                                  align_corners=False)
        prompt_vi = F.interpolate(prompt_vi, (height, width), mode="bilinear",
                                  align_corners=False)
        discrepancy = tuple(torch.abs(left - right)
                            for left, right in zip(ir_bands, vi_bands))
        router_input = torch.cat((*ir_bands, *vi_bands, *discrepancy,
                                  prompt_ir, prompt_vi), 1)
        logits = self.router(router_input)
        batch, _, height, width = logits.shape
        weights = logits.reshape(batch, 2, 3, -1, height, width).softmax(1)
        infrared = torch.stack(ir_bands, 1)
        visible = torch.stack(vi_bands, 1)
        fused = weights[:, 0] * infrared + weights[:, 1] * visible
        return tuple(fused[:, index] for index in range(3)), weights


class SourceWaveASSMFusion(nn.Module):
    """Deep frequency-decoupled fusion: ASSM for LL, reliability for details."""

    def __init__(self, channels: int, d_state: int, num_tokens: int) -> None:
        super().__init__()
        self.prompt_ir = SourcePrompt(channels)
        self.prompt_vi = SourcePrompt(channels)
        self.prompt_scale_ir = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.prompt_scale_vi = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.low_assm = PairedSemanticASSMFusion(
            channels, d_state, num_tokens
        )
        self.low_split = ExactCommonPrivateFusion(channels)
        self.high_fusion = ReliabilityHighBandFusion(channels)
        self.refine = Restormer_CNN_block(channels, channels)

    def forward(self, infrared, visible, raw_ir, raw_vi):
        prompt_ir = self.prompt_ir(raw_ir, infrared.shape[-2:])
        prompt_vi = self.prompt_vi(raw_vi, visible.shape[-2:])
        conditioned_ir = infrared + self.prompt_scale_ir * torch.tanh(prompt_ir)
        conditioned_vi = visible + self.prompt_scale_vi * torch.tanh(prompt_vi)

        ll_ir, *high_ir = haar_dwt(conditioned_ir)
        ll_vi, *high_vi = haar_dwt(conditioned_vi)
        joint, ll_ir_out, ll_vi_out, _, route = self.low_assm(ll_ir, ll_vi)
        split = self.low_split(ll_ir_out, ll_vi_out, joint)
        high, high_weights = self.high_fusion(
            tuple(high_ir), tuple(high_vi), prompt_ir, prompt_vi
        )
        candidate = haar_iwt(split["fused"], *high)
        candidate = self.refine(candidate)
        return candidate, route, split, high_weights


class DeepGuidedSkip(nn.Module):
    """Make every high-resolution skip conditional on the deeper decoder state."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 3, padding=1,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(2 * channels, hidden, 1),
            nn.GELU(), nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)
        nn.init.zeros_(self.channel[-1].weight)
        nn.init.zeros_(self.channel[-1].bias)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor):
        joined = torch.cat((deep, skip), 1)
        spatial = 2.0 * torch.sigmoid(self.spatial(joined))
        channel = 2.0 * torch.sigmoid(self.channel(joined))
        return skip * spatial * channel, spatial, channel


class UfuserSourceWaveASSMV10(nn.Module):
    """Four-scale main-path source/wavelet/ASSM successor to EMMA."""

    def __init__(self, d_state=16, num_tokens=9, num_classes=9,
                 initial_new=0.01) -> None:
        super().__init__()
        if num_tokens != num_classes:
            raise ValueError("V10 route supervision requires num_tokens == num_classes")
        self.emma = Ufuser()

        # SIBA-style raw-source queries replace shallow concatenation paths.
        self.shallow1 = SIBAShallowFusion(8)
        self.shallow2 = SIBAShallowFusion(16)
        self.deep3 = SourceWaveASSMFusion(32, d_state, num_tokens)
        self.deep4 = SourceWaveASSMFusion(32, d_state, num_tokens)
        self.blend1 = MainPathBlend(8, initial_new)
        self.blend2 = MainPathBlend(16, initial_new)
        self.blend3 = MainPathBlend(32, initial_new)
        self.blend4 = MainPathBlend(32, initial_new)

        self.skip3 = DeepGuidedSkip(32)
        self.skip2 = DeepGuidedSkip(16)
        self.skip1 = DeepGuidedSkip(8)
        self.semantic = DeepSemanticHead(32, num_classes)

    def load_emma(self, state) -> None:
        self.emma.load_state_dict(state, strict=True)

    def fusion_gates(self):
        return tuple(torch.sigmoid(module.gate_logit)
                     for module in (self.blend1, self.blend2,
                                    self.blend3, self.blend4))

    def forward_features(self, infrared, visible):
        b = self.emma
        i1 = b.I_en_1(infrared); i2 = b.I_en_2(b.I_down1(i1))
        i3 = b.I_en_3(b.I_down2(i2)); i4 = b.I_en_4(b.I_down3(i3))
        v1 = b.V_en_1(visible); v2 = b.V_en_2(b.V_down1(v1))
        v3 = b.V_en_3(b.V_down2(v2)); v4 = b.V_en_4(b.V_down3(v3))

        base1 = b.f_1(torch.cat((i1, v1), 1))
        base2 = b.f_2(torch.cat((i2, v2), 1))
        base3 = b.f_3(torch.cat((i3, v3), 1))
        base4 = b.f_4(torch.cat((i4, v4), 1))
        candidate1 = self.shallow1(infrared, visible, i1, v1)
        candidate2 = self.shallow2(infrared, visible, i2, v2)
        candidate3, route3, split3, high3 = self.deep3(
            i3, v3, infrared, visible
        )
        candidate4, route4, split4, high4 = self.deep4(
            i4, v4, infrared, visible
        )
        f1, gate1 = self.blend1(base1, candidate1)
        f2, gate2 = self.blend2(base2, candidate2)
        f3, gate3 = self.blend3(base3, candidate3)
        f4, gate4 = self.blend4(base4, candidate4)

        semantic_feature, semantic_logits = self.semantic(
            f3, f4, infrared.shape[-2:]
        )
        d4 = b.de_4(f4)
        up3 = b.up4(d4); guided3, spatial3, channel3 = self.skip3(up3, f3)
        d3 = b.de_3(torch.cat((up3, guided3), 1))
        up2 = b.up3(d3); guided2, spatial2, channel2 = self.skip2(up2, f2)
        d2 = b.de_2(torch.cat((up2, guided2), 1))
        up1 = b.up2(d2); guided1, spatial1, channel1 = self.skip1(up1, f1)
        d1 = b.de_1(torch.cat((up1, guided1), 1))
        logits = b.last[0](d1)
        fused = torch.sigmoid(logits)
        return {
            "fused": fused, "core": fused, "logit_delta": logits.new_zeros(()),
            "route3": route3, "route4": route4,
            "semantic_feature": semantic_feature,
            "semantic_logits": semantic_logits,
            "split3": split3, "split4": split4,
            "high_weights3": high3, "high_weights4": high4,
            "fusion_gates": (gate1, gate2, gate3, gate4),
            "skip_spatial": (spatial1, spatial2, spatial3),
            "skip_channel": (channel1, channel2, channel3),
            # Direct branch warm-up avoids both random-path shock and the
            # zero-output gradient delay observed in earlier versions.
            "base_features": (base1, base2, base3, base4),
            "candidate_features": (candidate1, candidate2,
                                   candidate3, candidate4),
        }

    def forward(self, infrared, visible, return_aux=False):
        outputs = self.forward_features(infrared, visible)
        return outputs if return_aux else outputs["fused"]


__all__ = [
    "UfuserSourceWaveASSMV10", "SourceWaveASSMFusion",
    "ReliabilityHighBandFusion", "DeepGuidedSkip", "MainPathBlend",
]
