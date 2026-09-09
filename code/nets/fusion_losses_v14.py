"""Region-specific semantic fidelity losses for V14."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nets.fusion_losses_v2 import _sobel, foreground_dice_loss


def _masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def semantic_region_losses(outputs, infrared, visible, labels, class_weights):
    """Supervise semantics and distinct foreground/background image roles."""
    fused = outputs["fused"]
    logits = outputs["semantic_logits"]
    foreground = (labels != 0).to(fused.dtype)[:, None]
    foreground_logit = outputs["foreground_logit"]
    positives = foreground.sum().clamp_min(1.0)
    negatives = foreground.numel() - positives
    positive_weight = (negatives / positives).clamp(1.0, 10.0)
    binary = F.binary_cross_entropy_with_logits(
        foreground_logit, foreground, pos_weight=positive_weight)
    probability = foreground_logit.sigmoid()
    binary_dice = 1.0 - ((2.0 * (probability * foreground).sum() + 1.0)
                         / (probability.sum() + foreground.sum() + 1.0))
    foreground_semantic = binary + binary_dice
    semantic = (F.cross_entropy(logits, labels, weight=class_weights)
                + foreground_dice_loss(logits, labels)
                + foreground_semantic)
    # A soft rim avoids discontinuities at semantic boundaries.
    foreground_soft = F.avg_pool2d(foreground, 5, 1, 2)
    background_soft = 1.0 - foreground_soft
    thermal_target = torch.maximum(infrared, visible)
    foreground_intensity = _masked_mean(
        F.smooth_l1_loss(fused, thermal_target, beta=0.02,
                         reduction="none"), foreground_soft)

    fgx, fgy = _sobel(fused)
    igx, igy = _sobel(infrared)
    vgx, vgy = _sobel(visible)
    target_gx = torch.where(igx.abs() >= vgx.abs(), igx, vgx)
    target_gy = torch.where(igy.abs() >= vgy.abs(), igy, vgy)
    boundary = F.max_pool2d(foreground, 3, 1, 1) - (-F.max_pool2d(
        -foreground, 3, 1, 1))
    boundary_gradient = _masked_mean(
        (fgx - target_gx).abs() + (fgy - target_gy).abs(),
        boundary.clamp(0.0, 1.0))
    background_texture = _masked_mean(
        (fgx - vgx).abs() + (fgy - vgy).abs(), background_soft)
    return {
        "semantic": semantic,
        "foreground_semantic": foreground_semantic,
        "foreground_intensity": foreground_intensity,
        "boundary_gradient": boundary_gradient,
        "background_texture": background_texture,
    }


__all__ = ["semantic_region_losses"]
