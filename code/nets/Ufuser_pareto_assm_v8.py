"""Pareto-constrained, semantically routed EMMA-ASSM V8.

V8 intentionally keeps EMMA's high-resolution fusion and decoder topology.
Only scales 3/4 receive paired cross-modal ASSM corrections.  The routing is
trainable from MSRS semantic labels, while an exactly reconstructable
common/private split avoids the degenerate learned reconstruction used by V7.
An invertible Haar branch may add a bounded, zero-mean high-frequency logit
correction.  Every new output path is zero initialized, so V8 starts exactly
from the supplied EMMA checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Ufuser import Restormer_CNN_block, Ufuser
from nets.Ufuser_end2end_v7 import PairedSemanticASSMFusion
from nets.Ufuser_pareto_v6 import InvertibleHighFrequencyBranch


class ExactCommonPrivateFusion(nn.Module):
    """A convex common feature and algebraic private residuals.

    For either modality ``x = common + private_x`` by construction.  There is
    no learned restoration head that can minimize reconstruction loss without
    making the decomposition useful to the fusion path.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(), nn.Conv2d(channels, channels, 1), nn.Sigmoid(),
        )
        self.fuse = Restormer_CNN_block(4 * channels, channels)

    def forward(self, infrared, visible, joint):
        gate = self.gate(torch.cat((infrared, visible, infrared * visible,
                                    torch.abs(infrared - visible)), 1))
        common = gate * infrared + (1.0 - gate) * visible
        private_ir = infrared - common
        private_vi = visible - common
        fused = self.fuse(torch.cat((joint, common, private_ir, private_vi), 1))
        return {
            "fused": fused, "common": common,
            "private_ir": private_ir, "private_vi": private_vi, "gate": gate,
        }


class DeepSemanticHead(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1,
                      padding_mode="reflect"), nn.GELU(),
        )
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, scale3, scale4, output_size):
        scale4 = F.interpolate(scale4, scale3.shape[-2:], mode="bilinear",
                               align_corners=False)
        feature = self.body(torch.cat((scale3, scale4), 1))
        logits = F.interpolate(self.classifier(feature), output_size,
                               mode="bilinear", align_corners=False)
        return feature, logits


class UfuserParetoASSMV8(nn.Module):
    """EMMA U-Net with deep paired ASSM and bounded detail correction."""

    def __init__(self, d_state=16, num_tokens=9, num_classes=9,
                 max_feature_scale=0.25, max_detail_scale=0.20) -> None:
        super().__init__()
        if num_tokens != num_classes:
            raise ValueError("V8 uses class-supervised routes: num_tokens must equal num_classes")
        self.emma = Ufuser()
        self.assm3 = PairedSemanticASSMFusion(32, d_state, num_tokens)
        self.assm4 = PairedSemanticASSMFusion(32, d_state, num_tokens)
        self.split4 = ExactCommonPrivateFusion(32)
        self.semantic = DeepSemanticHead(32, num_classes)
        self.feature_scale3_logit = nn.Parameter(torch.zeros(1, 32, 1, 1))
        self.feature_scale4_logit = nn.Parameter(torch.zeros(1, 32, 1, 1))
        self.detail_branch = InvertibleHighFrequencyBranch()
        self.detail_scale_logit = nn.Parameter(torch.zeros(1))
        self.max_feature_scale = float(max_feature_scale)
        self.max_detail_scale = float(max_detail_scale)

    def load_emma(self, state) -> None:
        self.emma.load_state_dict(state, strict=True)

    def forward_features(self, infrared, visible):
        b = self.emma
        i1 = b.I_en_1(infrared); i2 = b.I_en_2(b.I_down1(i1))
        i3 = b.I_en_3(b.I_down2(i2)); i4 = b.I_en_4(b.I_down3(i3))
        v1 = b.V_en_1(visible); v2 = b.V_en_2(b.V_down1(v1))
        v3 = b.V_en_3(b.V_down2(v2)); v4 = b.V_en_4(b.V_down3(v3))

        # Preserve the high-resolution EMMA paths that carried most useful detail.
        f1 = b.f_1(torch.cat((i1, v1), 1))
        f2 = b.f_2(torch.cat((i2, v2), 1))
        base_f3 = b.f_3(torch.cat((i3, v3), 1))
        base_f4 = b.f_4(torch.cat((i4, v4), 1))

        joint3, i3j, v3j, _, route3 = self.assm3(i3, v3)
        joint4, i4j, v4j, _, route4 = self.assm4(i4, v4)
        split4 = self.split4(i4j, v4j, joint4)
        scale3 = self.max_feature_scale * torch.tanh(self.feature_scale3_logit)
        scale4 = self.max_feature_scale * torch.tanh(self.feature_scale4_logit)
        f3 = base_f3 + scale3 * torch.tanh(joint3)
        f4 = base_f4 + scale4 * torch.tanh(split4["fused"])

        semantic_feature, semantic_logits = self.semantic(
            f3, f4, infrared.shape[-2:]
        )

        # Keep EMMA's original decoder information flow; V7's skip modifiers were
        # empirically near zero and added optimization shortcuts.
        d4 = b.de_4(f4)
        d3 = b.de_3(torch.cat((b.up4(d4), f3), 1))
        d2 = b.de_2(torch.cat((b.up3(d3), f2), 1))
        d1 = b.de_1(torch.cat((b.up2(d2), f1), 1))
        core_logits = b.last[0](d1)

        detail, detail_weights = self.detail_branch(infrared, visible)
        detail_scale = self.max_detail_scale * torch.tanh(self.detail_scale_logit)
        detail_delta = detail_scale * torch.tanh(detail)
        fused = torch.sigmoid(core_logits + detail_delta)
        core = torch.sigmoid(core_logits)
        return {
            "fused": fused, "core": core, "logit_delta": detail_delta,
            "route3": route3, "route4": route4,
            "semantic_feature": semantic_feature,
            "semantic_logits": semantic_logits,
            "split4": split4, "feature_scale3": scale3,
            "feature_scale4": scale4, "detail_scale": detail_scale,
            "detail_weights": detail_weights,
        }

    def forward(self, infrared, visible, return_aux=False):
        outputs = self.forward_features(infrared, visible)
        return outputs if return_aux else outputs["fused"]


__all__ = ["UfuserParetoASSMV8", "ExactCommonPrivateFusion"]
