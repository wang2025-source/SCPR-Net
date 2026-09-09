"""Fusion-first losses for baseline-preserving EMMA-SGRM V4."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nets.fusion_losses_v3 import source_selected_gradient_loss


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    kernel = x.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).reshape(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def multiscale_source_laplacian_loss(
    fused: torch.Tensor,
    infrared: torch.Tensor,
    visible: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """Select the signed Laplacian with greater source magnitude at each scale."""
    loss = fused.new_zeros(())
    for scale in scales:
        if scale == 1:
            fu, ir, vi = fused, infrared, visible
        else:
            fu = F.avg_pool2d(fused, scale, stride=scale)
            ir = F.avg_pool2d(infrared, scale, stride=scale)
            vi = F.avg_pool2d(visible, scale, stride=scale)
        lap_fu, lap_ir, lap_vi = _laplacian(fu), _laplacian(ir), _laplacian(vi)
        target = torch.where(lap_ir.abs() >= lap_vi.abs(), lap_ir, lap_vi)
        loss = loss + F.l1_loss(lap_fu, target)
    return loss / len(scales)


def _local_contrast(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    mean = F.avg_pool2d(x, window, stride=1, padding=window // 2)
    mean_square = F.avg_pool2d(x.square(), window, stride=1, padding=window // 2)
    return torch.sqrt((mean_square - mean.square()).clamp_min(1e-6))


def source_selected_contrast_loss(
    fused: torch.Tensor, infrared: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    contrast_fused = _local_contrast(fused)
    target = torch.maximum(_local_contrast(infrared), _local_contrast(visible))
    return F.l1_loss(contrast_fused, target)


def fusion_detail_objective(
    fused: torch.Tensor,
    infrared: torch.Tensor,
    visible: torch.Tensor,
    sensing_loss: torch.nn.Module,
    ai: torch.nn.Module,
    av: torch.nn.Module,
    lambda_gradient: float = 1.0,
    lambda_laplacian: float = 0.5,
    lambda_contrast: float = 0.2,
) -> dict[str, torch.Tensor]:
    sensing = sensing_loss(av(fused), visible) + sensing_loss(ai(fused), infrared)
    gradient = source_selected_gradient_loss(fused, infrared, visible)
    laplacian = multiscale_source_laplacian_loss(fused, infrared, visible)
    contrast = source_selected_contrast_loss(fused, infrared, visible)
    total = (
        sensing
        + lambda_gradient * gradient
        + lambda_laplacian * laplacian
        + lambda_contrast * contrast
    )
    return {
        "total": total,
        "sensing": sensing,
        "gradient": gradient,
        "laplacian": laplacian,
        "contrast": contrast,
    }


def baseline_ranking_loss(
    new_objective: torch.Tensor,
    baseline_objective: torch.Tensor,
    margin: float = 1e-3,
) -> torch.Tensor:
    return F.relu(new_objective - baseline_objective.detach() + margin)
