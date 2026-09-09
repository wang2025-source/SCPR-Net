"""Stable losses for the independent EMMA ASSM-SHIP Task V2 model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fixed_weight_intensity_loss(
    fused: torch.Tensor,
    infrared: torch.Tensor,
    visible: torch.Tensor,
    infrared_weight: float = 0.5,
) -> torch.Tensor:
    if not 0.0 <= infrared_weight <= 1.0:
        raise ValueError("infrared_weight must be in [0, 1]")
    visible_weight = 1.0 - infrared_weight
    return (
        infrared_weight * (infrared - fused).square()
        + visible_weight * (visible - fused).square()
    ).mean()


def _sobel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = x.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    return F.conv2d(x, kernel_x, padding=1), F.conv2d(x, kernel_y, padding=1)


def source_selected_gradient_loss(
    fused: torch.Tensor, infrared: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    """Use one source consistently for both gradient components at each pixel."""
    fused_x, fused_y = _sobel(fused)
    ir_x, ir_y = _sobel(infrared)
    vi_x, vi_y = _sobel(visible)
    ir_magnitude = ir_x.square() + ir_y.square()
    vi_magnitude = vi_x.square() + vi_y.square()
    choose_ir = ir_magnitude >= vi_magnitude
    target_x = torch.where(choose_ir, ir_x, vi_x)
    target_y = torch.where(choose_ir, ir_y, vi_y)
    return F.l1_loss(fused_x, target_x) + F.l1_loss(fused_y, target_y)


def foreground_dice_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    # Background dominates MSRS; Dice is averaged over foreground classes only.
    probabilities = probabilities[:, 1:]
    one_hot = one_hot[:, 1:]
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dimensions)
    denominator = probabilities.sum(dimensions) + one_hot.sum(dimensions)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()
