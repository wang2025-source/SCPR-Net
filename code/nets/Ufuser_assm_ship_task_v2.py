"""Independent EMMA V2: deep ASSM, symmetric residual SHIP and safe decoder.

This file does not replace any previous EMMA variant. Shallow Restormer-CNN
blocks retain high-resolution detail; ASSM is restricted to deep stages; SHIP
is used only at scale 3; transposed convolutions are replaced by resize-conv.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.assm_mambairv2 import ASSMResidual
from nets.ship_high_order import SHIPHighOrderInteraction
from nets.Ufuser import LayerNorm, LocalFeatureExtraction, Restormer_CNN_block


class LocalCNNBlockV2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.embed = nn.Conv2d(
            in_dim, out_dim, 3, padding=1, bias=False, padding_mode="reflect"
        )
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN = nn.Conv2d(
            2 * out_dim, out_dim, 3, padding=1, bias=False, padding_mode="reflect"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        return self.FFN(torch.cat((x, self.LocalFeature(x)), dim=1))


class ASSMCNNBlockV2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, d_state: int, num_tokens: int) -> None:
        super().__init__()
        self.embed = nn.Conv2d(
            in_dim, out_dim, 3, padding=1, bias=False, padding_mode="reflect"
        )
        self.GlobalFeature = ASSMResidual(out_dim, d_state=d_state, num_tokens=num_tokens)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN = nn.Conv2d(
            2 * out_dim, out_dim, 3, padding=1, bias=False, padding_mode="reflect"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        return self.FFN(
            torch.cat((self.GlobalFeature(x), self.LocalFeature(x)), dim=1)
        )


class ResidualModalityAdapter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(x)


class SymmetricResidualSHIP(nn.Module):
    """Wrap the existing SHIP adaptation with protected residuals on both streams."""

    def __init__(self, channels: int, order: int = 4, scale_init: float = 1e-2) -> None:
        super().__init__()
        self.core = SHIPHighOrderInteraction(channels, order=order)
        self.visible_norm = LayerNorm(channels, "WithBias")
        self.infrared_norm = LayerNorm(channels, "WithBias")
        self.visible_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), scale_init)
        )
        self.infrared_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), scale_init)
        )

    def forward(
        self, visible: torch.Tensor, infrared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_visible, candidate_infrared = self.core(visible, infrared)
        visible_delta = self.visible_norm(candidate_visible - visible)
        infrared_delta = self.infrared_norm(candidate_infrared - infrared)
        return (
            visible + self.visible_scale * visible_delta,
            infrared + self.infrared_scale * infrared_delta,
        )


class ResizeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=False,
            padding_mode="reflect"
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.activation(self.conv(x))


class IdentityInitializedRefiner(nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1, padding_mode="reflect"),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, 1, 3, padding=1, padding_mode="reflect"),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.body(x))
        return torch.clamp(x + self.scale * delta, 0.0, 1.0)


class UfuserASSMSHIPTaskV2(nn.Module):
    def __init__(
        self,
        d_state: int = 16,
        num_tokens: int = 8,
        ship_order: int = 4,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        channels = [8, 16, 32, 32]

        # Preserve the original EMMA blocks where spatial detail is highest.
        self.I_en_1 = Restormer_CNN_block(1, channels[0])
        self.I_en_2 = Restormer_CNN_block(channels[0], channels[1])
        self.V_en_1 = Restormer_CNN_block(1, channels[0])
        self.V_en_2 = Restormer_CNN_block(channels[0], channels[1])
        self.I_en_3_local = LocalCNNBlockV2(channels[1], channels[2])
        self.I_en_4_local = LocalCNNBlockV2(channels[2], channels[3])
        self.V_en_3_local = LocalCNNBlockV2(channels[1], channels[2])
        self.V_en_4_local = LocalCNNBlockV2(channels[2], channels[3])

        self.I_adapter_3 = ResidualModalityAdapter(channels[2])
        self.I_adapter_4 = ResidualModalityAdapter(channels[3])
        self.V_adapter_3 = ResidualModalityAdapter(channels[2])
        self.V_adapter_4 = ResidualModalityAdapter(channels[3])
        self.shared_assm_3 = ASSMResidual(channels[2], d_state, num_tokens)
        self.shared_assm_4 = ASSMResidual(channels[3], d_state, num_tokens)

        self.f_1 = Restormer_CNN_block(2 * channels[0], channels[0])
        self.f_2 = Restormer_CNN_block(2 * channels[1], channels[1])
        self.ship_f3 = SymmetricResidualSHIP(channels[2], order=ship_order)
        self.f_3 = ASSMCNNBlockV2(2 * channels[2], channels[2], d_state, num_tokens)
        self.f_4 = ASSMCNNBlockV2(2 * channels[3], channels[3], d_state, num_tokens)

        self.I_down1 = self._down(channels[0])
        self.I_down2 = self._down(channels[1])
        self.I_down3 = self._down(channels[2])
        self.V_down1 = self._down(channels[0])
        self.V_down2 = self._down(channels[1])
        self.V_down3 = self._down(channels[2])

        self.de_4 = ASSMCNNBlockV2(channels[3], channels[3], d_state, num_tokens)
        self.de_3 = ASSMCNNBlockV2(2 * channels[2], channels[2], d_state, num_tokens)
        self.de_2 = Restormer_CNN_block(2 * channels[1], channels[1])
        self.de_1 = Restormer_CNN_block(2 * channels[0], channels[0])
        self.up4 = ResizeConv(channels[3], channels[2])
        self.up3 = ResizeConv(channels[2], channels[1])
        self.up2 = ResizeConv(channels[1], channels[0])
        self.last = nn.Sequential(
            nn.Conv2d(channels[0], 1, 3, padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )
        self.refiner = IdentityInitializedRefiner()

        # ASSM must not begin as an effectively disabled 1e-4 branch.
        for module in self.modules():
            if isinstance(module, ASSMResidual):
                nn.init.constant_(module.scale1, layer_scale_init)
                nn.init.constant_(module.scale2, layer_scale_init)

    @staticmethod
    def _down(channels: int) -> nn.Conv2d:
        return nn.Conv2d(
            channels, channels, 3, stride=2, padding=1,
            bias=False, padding_mode="reflect"
        )

    def forward_backbone(
        self, infrared: torch.Tensor, visible: torch.Tensor
    ) -> torch.Tensor:
        i1 = self.I_en_1(infrared)
        i2 = self.I_en_2(self.I_down1(i1))
        i3 = self.I_en_3_local(self.I_down2(i2))
        i3 = self.shared_assm_3(self.I_adapter_3(i3))
        i4 = self.I_en_4_local(self.I_down3(i3))
        i4 = self.shared_assm_4(self.I_adapter_4(i4))

        v1 = self.V_en_1(visible)
        v2 = self.V_en_2(self.V_down1(v1))
        v3 = self.V_en_3_local(self.V_down2(v2))
        v3 = self.shared_assm_3(self.V_adapter_3(v3))
        v4 = self.V_en_4_local(self.V_down3(v3))
        v4 = self.shared_assm_4(self.V_adapter_4(v4))

        f1 = self.f_1(torch.cat((i1, v1), dim=1))
        f2 = self.f_2(torch.cat((i2, v2), dim=1))
        v3_interacted, i3_interacted = self.ship_f3(v3, i3)
        f3 = self.f_3(torch.cat((i3_interacted, v3_interacted), dim=1))
        f4 = self.f_4(torch.cat((i4, v4), dim=1))

        out = self.up4(self.de_4(f4))
        out = self.up3(self.de_3(torch.cat((out, f3), dim=1)))
        out = self.up2(self.de_2(torch.cat((out, f2), dim=1)))
        return self.last(self.de_1(torch.cat((out, f1), dim=1)))

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        return self.refiner(self.forward_backbone(infrared, visible))
