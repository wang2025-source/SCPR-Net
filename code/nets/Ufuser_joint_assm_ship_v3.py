"""EMMA V3: joint cross-modal ASSM and deep-guided U-shaped fusion.

This is an independent model file.  It does not replace the original EMMA or
the V2 experiment.  High-resolution EMMA blocks are retained, while scales 3
and 4 explicitly exchange modal tokens and separate common/private features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.assm_mambairv2 import ASSMResidual
from nets.joint_assm_v3 import CommonPrivateAdapter, JointCrossModalASSM
from nets.ship_high_order import SHIPHighOrderInteraction
from nets.Ufuser import Restormer_CNN_block
from nets.Ufuser_assm_ship_task_v2 import (
    ASSMCNNBlockV2,
    IdentityInitializedRefiner,
    LocalCNNBlockV2,
)


class PrivateSHIPFusion(nn.Module):
    """SHIP operates on private features and contributes a bounded branch."""

    def __init__(self, channels: int, order: int = 4) -> None:
        super().__init__()
        self.ship = SHIPHighOrderInteraction(channels, order=order)
        nn.init.constant_(self.ship.spatial.gamma, 5e-2)
        nn.init.constant_(self.ship.channel.gamma, 5e-2)
        self.base = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.ship_project = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # sigmoid(0)=0.5 -> effective contribution 0.55, bounded in [0.1, 1].
        self.gate_logit = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(
        self, private_ir: torch.Tensor, private_vi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ship_vi, ship_ir = self.ship(private_vi, private_ir)
        base = self.base(torch.cat((private_ir, private_vi), dim=1))
        ship_detail = self.ship_project(
            torch.cat((private_ir, private_vi, ship_ir, ship_vi), dim=1)
        )
        gate = 0.1 + 0.9 * torch.sigmoid(self.gate_logit)
        return self.out(base + gate * ship_detail), gate


class LightPrivateFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, private_ir: torch.Tensor, private_vi: torch.Tensor) -> torch.Tensor:
        return self.body(
            torch.cat(
                (
                    private_ir,
                    private_vi,
                    private_ir * private_vi,
                    torch.abs(private_ir - private_vi),
                ),
                dim=1,
            )
        )


class DeepGuidedSkip(nn.Module):
    """The decoder state gates shallow skips; the deep path is always present."""

    def __init__(
        self,
        channels: int,
        processor: nn.Module,
        skip_drop_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.skip_drop_probability = skip_drop_probability
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.processor = processor
        self.deep_project = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))

    def _drop_skip(self, skip: torch.Tensor) -> torch.Tensor:
        if not self.training or self.skip_drop_probability <= 0:
            return skip
        keep = 1.0 - self.skip_drop_probability
        mask = torch.empty(
            (skip.shape[0], 1, 1, 1), device=skip.device, dtype=skip.dtype
        ).bernoulli_(keep)
        return skip * mask / keep

    def forward(self, deep: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        gate = 0.1 + 0.9 * torch.sigmoid(self.gate(deep))
        shallow = gate * self._drop_skip(shallow)
        correction = self.processor(torch.cat((deep, shallow), dim=1))
        return self.deep_project(deep) + self.residual_scale * correction


class DualPathUpsample(nn.Module):
    """Bilinear structure path plus learnable PixelShuffle detail path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.structure = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.detail = nn.Sequential(
            nn.Conv2d(in_channels, 4 * out_channels, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.detail_scale = nn.Parameter(torch.full((1, out_channels, 1, 1), 0.1))
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        structure = self.structure(
            F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        )
        return self.activation(structure + self.detail_scale * self.detail(x))


class UfuserJointASSMSHIPV3(nn.Module):
    """Dual-stream U-shaped fusion with mandatory deep cross-modal interaction."""

    def __init__(
        self,
        d_state: int = 16,
        num_tokens: int = 8,
        ship_order: int = 4,
        layer_scale_init: float = 5e-2,
        num_classes: int = 9,
    ) -> None:
        super().__init__()
        channels = [8, 16, 32, 32]

        self.I_en_1 = Restormer_CNN_block(1, channels[0])
        self.I_en_2 = Restormer_CNN_block(channels[0], channels[1])
        self.V_en_1 = Restormer_CNN_block(1, channels[0])
        self.V_en_2 = Restormer_CNN_block(channels[0], channels[1])
        self.I_en_3_local = LocalCNNBlockV2(channels[1], channels[2])
        self.I_en_4_local = LocalCNNBlockV2(channels[2], channels[3])
        self.V_en_3_local = LocalCNNBlockV2(channels[1], channels[2])
        self.V_en_4_local = LocalCNNBlockV2(channels[2], channels[3])

        self.I_down1 = self._down(channels[0])
        self.I_down2 = self._down(channels[1])
        self.I_down3 = self._down(channels[2])
        self.V_down1 = self._down(channels[0])
        self.V_down2 = self._down(channels[1])
        self.V_down3 = self._down(channels[2])

        self.joint_assm_3 = JointCrossModalASSM(
            channels[2], d_state, num_tokens, layer_scale_init=layer_scale_init
        )
        self.joint_assm_4 = JointCrossModalASSM(
            channels[3], d_state, num_tokens, layer_scale_init=layer_scale_init
        )
        self.decompose_3 = CommonPrivateAdapter(channels[2])
        self.decompose_4 = CommonPrivateAdapter(channels[3])
        self.private_ship_3 = PrivateSHIPFusion(channels[2], order=ship_order)
        self.private_fuse_4 = LightPrivateFusion(channels[3])

        # Shallow fused features remain available, but cannot bypass the deep state.
        self.f_1 = Restormer_CNN_block(2 * channels[0], channels[0])
        self.f_2 = Restormer_CNN_block(2 * channels[1], channels[1])
        self.f_3 = LocalCNNBlockV2(2 * channels[2], channels[2])
        self.f_4 = ASSMCNNBlockV2(
            2 * channels[3], channels[3], d_state, num_tokens
        )

        self.de_4 = ASSMResidual(channels[3], d_state, num_tokens)
        self.skip_3 = DeepGuidedSkip(
            channels[2],
            ASSMCNNBlockV2(2 * channels[2], channels[2], d_state, num_tokens),
            skip_drop_probability=0.1,
        )
        self.skip_2 = DeepGuidedSkip(
            channels[1],
            Restormer_CNN_block(2 * channels[1], channels[1]),
            skip_drop_probability=0.2,
        )
        self.skip_1 = DeepGuidedSkip(
            channels[0],
            Restormer_CNN_block(2 * channels[0], channels[0]),
            skip_drop_probability=0.3,
        )
        self.up4 = DualPathUpsample(channels[3], channels[2])
        self.up3 = DualPathUpsample(channels[2], channels[1])
        self.up2 = DualPathUpsample(channels[1], channels[0])

        self.last = nn.Sequential(
            nn.Conv2d(channels[0], 1, 3, padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )
        self.deep_head = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels[1], 1, 1),
        )
        self.semantic_head = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels[3], num_classes, 1),
        )
        self.refiner = IdentityInitializedRefiner()

        for module in self.modules():
            if isinstance(module, ASSMResidual):
                nn.init.constant_(module.scale1, layer_scale_init)
                nn.init.constant_(module.scale2, layer_scale_init)

    @staticmethod
    def _down(channels: int) -> nn.Conv2d:
        return nn.Conv2d(
            channels,
            channels,
            3,
            stride=2,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )

    def forward_features(
        self, infrared: torch.Tensor, visible: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        i1 = self.I_en_1(infrared)
        i2 = self.I_en_2(self.I_down1(i1))
        v1 = self.V_en_1(visible)
        v2 = self.V_en_2(self.V_down1(v1))

        i3_local = self.I_en_3_local(self.I_down2(i2))
        v3_local = self.V_en_3_local(self.V_down2(v2))
        i3, v3, joint3 = self.joint_assm_3(i3_local, v3_local)
        split3 = self.decompose_3(i3, v3, joint3)
        split3["source_ir"] = i3
        split3["source_vi"] = v3
        detail3, ship_gate = self.private_ship_3(
            split3["private_ir"], split3["private_vi"]
        )
        f3 = self.f_3(torch.cat((split3["common"], detail3), dim=1))

        i4_local = self.I_en_4_local(self.I_down3(i3))
        v4_local = self.V_en_4_local(self.V_down3(v3))
        i4, v4, joint4 = self.joint_assm_4(i4_local, v4_local)
        split4 = self.decompose_4(i4, v4, joint4)
        split4["source_ir"] = i4
        split4["source_vi"] = v4
        detail4 = self.private_fuse_4(split4["private_ir"], split4["private_vi"])
        f4 = self.f_4(torch.cat((split4["common"], detail4), dim=1))

        f1 = self.f_1(torch.cat((i1, v1), dim=1))
        f2 = self.f_2(torch.cat((i2, v2), dim=1))
        d4 = self.de_4(f4)
        d3 = self.skip_3(self.up4(d4), f3)
        d2 = self.skip_2(self.up3(d3), f2)
        d1 = self.skip_1(self.up2(d2), f1)
        backbone = self.last(d1)
        fused = self.refiner(backbone)

        full_size = infrared.shape[-2:]
        deep_fused = torch.sigmoid(
            F.interpolate(
                self.deep_head(d3), size=full_size, mode="bilinear", align_corners=False
            )
        )
        semantic_logits = F.interpolate(
            self.semantic_head(split4["common"]),
            size=full_size,
            mode="bilinear",
            align_corners=False,
        )
        return {
            "fused": fused,
            "backbone": backbone,
            "deep_fused": deep_fused,
            "semantic_logits": semantic_logits,
            "scale3": split3,
            "scale4": split4,
            "ship_gate": ship_gate,
        }

    def forward(
        self,
        infrared: torch.Tensor,
        visible: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        outputs = self.forward_features(infrared, visible)
        return outputs if return_aux else outputs["fused"]
