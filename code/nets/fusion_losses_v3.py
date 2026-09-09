"""Losses for EMMA Joint-ASSM-SHIP V3."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nets.fusion_losses_v2 import _sobel, foreground_dice_loss


def source_saliency_weights(
    infrared: torch.Tensor,
    visible: torch.Tensor,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-trainable local-contrast weights for source-aware intensity targets."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    def saliency(x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(x, 5, stride=1, padding=2)
        gx, gy = _sobel(x)
        return torch.abs(x - mean) + 0.25 * torch.sqrt(gx.square() + gy.square() + 1e-6)

    with torch.no_grad():
        scores = torch.cat((saliency(infrared), saliency(visible)), dim=1)
        weights = torch.softmax(scores / temperature, dim=1)
    return weights[:, :1], weights[:, 1:]


def source_adaptive_intensity_loss(
    fused: torch.Tensor,
    infrared: torch.Tensor,
    visible: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    weight_ir, weight_vi = source_saliency_weights(infrared, visible, temperature)
    target = weight_ir * infrared + weight_vi * visible
    return F.l1_loss(fused, target)


def source_selected_gradient_loss(
    fused: torch.Tensor, infrared: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    fused_x, fused_y = _sobel(fused)
    ir_x, ir_y = _sobel(infrared)
    vi_x, vi_y = _sobel(visible)
    choose_ir = ir_x.square() + ir_y.square() >= vi_x.square() + vi_y.square()
    target_x = torch.where(choose_ir, ir_x, vi_x)
    target_y = torch.where(choose_ir, ir_y, vi_y)
    return F.l1_loss(fused_x, target_x) + F.l1_loss(fused_y, target_y)


def _absolute_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    return F.cosine_similarity(left, right, dim=1).abs().mean()


def common_private_losses(
    scale: dict[str, torch.Tensor],
    original_ir: torch.Tensor,
    original_vi: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Consistency, separation and reconstruction for one feature scale."""
    consistency = 1.0 - F.cosine_similarity(
        scale["common_ir"].flatten(1), scale["common_vi"].flatten(1), dim=1
    ).mean()
    private = _absolute_cosine(scale["private_ir"], scale["private_vi"])
    private = private + 0.5 * (
        _absolute_cosine(scale["common"], scale["private_ir"])
        + _absolute_cosine(scale["common"], scale["private_vi"])
    )
    reconstruction = F.l1_loss(scale["reconstructed_ir"], original_ir)
    reconstruction = reconstruction + F.l1_loss(
        scale["reconstructed_vi"], original_vi
    )
    return {
        "common": consistency,
        "private": private,
        "reconstruction": reconstruction,
    }


def semantic_auxiliary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    distillation_temperature: float = 2.0,
) -> torch.Tensor:
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    loss = loss + foreground_dice_loss(logits, labels)
    if teacher_logits is not None:
        temperature = distillation_temperature
        distillation = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature / (logits.shape[-2] * logits.shape[-1]))
        loss = loss + 0.25 * distillation
    return loss
