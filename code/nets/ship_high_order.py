"""Compact SHIP high-order cross-modal interaction for EMMA features.

Adapted from the spatial/channel interaction definitions in SHIP
(Zheng et al., CVPR 2024). The two defining operations are retained:

* spatial interaction starts from the product of two 2-D FFT spectra and
  evolves the cross-modal feature through multiplicative high orders;
* channel interaction starts from pooled joint-modal channel responses and
  evolves both the response and feature through multiplicative high orders.

The original SHIP network uses an invertible post-processing backbone after
the interaction. Here a 1x1 projection is used because the interaction is a
front-end to EMMA's existing f3/f4 ASSM blocks, not a replacement backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nets.Ufuser import LayerNorm


def _spatial_refiner(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
        nn.ReLU(inplace=False),
        nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
    )


def _channel_refiner(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 1),
        nn.ReLU(inplace=False),
        nn.Conv2d(channels, channels, 1),
    )


class SHIPSpatialInteraction(nn.Module):
    """FFT-initialized fourth-order spatial interaction from SHIP."""

    def __init__(self, channels: int, order: int = 4) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("SHIP spatial order must be positive")
        self.order = order
        self.fused_refiners = nn.ModuleList(
            [_spatial_refiner(channels) for _ in range(order - 1)]
        )
        self.infrared_refiners = nn.ModuleList(
            [_spatial_refiner(channels) for _ in range(order - 1)]
        )
        self.norms = nn.ModuleList(
            [LayerNorm(channels, "WithBias") for _ in range(order)]
        )
        # Small residual scale stabilizes the multiplicative path at start.
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(
        self, visible: torch.Tensor, infrared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = visible.shape[-2:]
        dtype = visible.dtype

        visible_fft = torch.fft.rfft2(visible.float())
        infrared_fft = torch.fft.rfft2(infrared.float())
        interaction = torch.fft.irfft2(
            visible_fft * infrared_fft, s=(height, width)
        ).to(dtype=dtype)

        fused = self.norms[0](interaction) * infrared
        infrared_state = infrared
        for index, (fused_refine, infrared_refine) in enumerate(
            zip(self.fused_refiners, self.infrared_refiners), start=1
        ):
            fused = self.norms[index](fused_refine(fused))
            infrared_state = infrared_refine(infrared_state)
            fused = fused * infrared_state

        return visible + self.gamma * fused, infrared_state


class SHIPChannelInteraction(nn.Module):
    """Fourth-order channel-statistics interaction from SHIP."""

    def __init__(self, channels: int, order: int = 4) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("SHIP channel order must be positive")
        self.order = order
        joint_channels = 2 * channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attention = _channel_refiner(joint_channels)
        self.attention_refiners = nn.ModuleList(
            [_channel_refiner(joint_channels) for _ in range(order - 1)]
        )
        self.feature_refiners = nn.ModuleList(
            [_spatial_refiner(joint_channels) for _ in range(order - 1)]
        )
        self.project = nn.Conv2d(joint_channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(
        self, visible: torch.Tensor, infrared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat((visible, infrared), dim=1)
        response = torch.softmax(self.attention(self.pool(joint)), dim=1)
        fused = joint * response

        for feature_refine, attention_refine in zip(
            self.feature_refiners, self.attention_refiners
        ):
            fused = feature_refine(fused)
            response = torch.softmax(attention_refine(response), dim=1)
            fused = fused * response

        return visible + self.gamma * self.project(fused), infrared


class SHIPHighOrderInteraction(nn.Module):
    """Sequential spatial then channel interaction, matching SHIP's order."""

    def __init__(self, channels: int, order: int = 4) -> None:
        super().__init__()
        self.spatial = SHIPSpatialInteraction(channels, order=order)
        self.channel = SHIPChannelInteraction(channels, order=order)

    def forward(
        self, visible: torch.Tensor, infrared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible, infrared = self.spatial(visible, infrared)
        return self.channel(visible, infrared)
