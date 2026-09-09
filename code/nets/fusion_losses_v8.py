"""Baseline-relative fusion objectives for Pareto ASSM V8."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nets.fusion_losses_v2 import _sobel, foreground_dice_loss
from nets.fusion_losses_v3 import source_selected_gradient_loss
from nets.fusion_losses_v4 import (
    multiscale_source_laplacian_loss, source_selected_contrast_loss,
)
from nets.fusion_losses_v6 import wavelet_highband_loss


VISUAL_NAMES = ("intensity", "gradient", "laplacian", "contrast", "ssim", "wavelet")


def _saliency(x):
    mean = F.avg_pool2d(x, 5, stride=1, padding=2)
    gx, gy = _sobel(x)
    return torch.abs(x - mean) + 0.25 * torch.sqrt(gx.square() + gy.square() + 1e-6)


def confident_intensity_target(infrared, visible, baseline, confidence_scale=0.05):
    """Hard-select active sources and retain EMMA in ambiguous regions."""
    with torch.no_grad():
        sal_ir, sal_vi = _saliency(infrared), _saliency(visible)
        selected = torch.where(sal_ir >= sal_vi, infrared, visible)
        confidence = 1.0 - torch.exp(-torch.abs(sal_ir - sal_vi) / confidence_scale)
        target = confidence * selected + (1.0 - confidence) * baseline
    return target, confidence


def _ssim_map(left, right, window=7):
    ml = F.avg_pool2d(left, window, 1, window // 2)
    mr = F.avg_pool2d(right, window, 1, window // 2)
    vl = F.avg_pool2d(left.square(), window, 1, window // 2) - ml.square()
    vr = F.avg_pool2d(right.square(), window, 1, window // 2) - mr.square()
    cov = F.avg_pool2d(left * right, window, 1, window // 2) - ml * mr
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * ml * mr + c1) * (2 * cov + c2)
            / ((ml.square() + mr.square() + c1) * (vl + vr + c2)))


def selected_ssim_loss(fused, infrared, visible):
    return 1.0 - torch.maximum(_ssim_map(fused, infrared),
                               _ssim_map(fused, visible)).mean()


def visual_terms(fused, infrared, visible, baseline):
    target, confidence = confident_intensity_target(infrared, visible, baseline)
    return {
        "intensity": F.smooth_l1_loss(fused, target, beta=0.02),
        "gradient": source_selected_gradient_loss(fused, infrared, visible),
        "laplacian": multiscale_source_laplacian_loss(fused, infrared, visible),
        "contrast": source_selected_contrast_loss(fused, infrared, visible),
        "ssim": selected_ssim_loss(fused, infrared, visible),
        "wavelet": wavelet_highband_loss(fused, infrared, visible),
        "selection_confidence": confidence.mean(),
    }


def normalized_pareto_guard(terms, baseline_terms, margin=0.0):
    guards = {
        name: F.relu(terms[name] / baseline_terms[name].detach().clamp_min(1e-6)
                     - (1.0 - margin))
        for name in VISUAL_NAMES
    }
    return sum(guards.values()), guards


def _local_std(x, window=9):
    mean = F.avg_pool2d(x, window, 1, window // 2)
    square = F.avg_pool2d(x.square(), window, 1, window // 2)
    return (square - mean.square()).clamp_min(1e-8).sqrt()


def statistic_gain_guard(fused, baseline, std_gain=1.01, detail_gain=1.005,
                         mean_tolerance=1.0 / 255.0):
    """Require controlled dispersion/detail gains without brightness drift."""
    dims = (-2, -1); base = baseline.detach()
    mean = F.relu((fused.mean(dims) - base.mean(dims)).abs() - mean_tolerance).mean()
    std_ratio = fused.std(dims, unbiased=False) / base.std(dims, unbiased=False).clamp_min(1e-4)
    local_ratio = _local_std(fused).mean(dims) / _local_std(base).mean(dims).clamp_min(1e-4)
    fgx, fgy = _sobel(fused); bgx, bgy = _sobel(base)
    grad_f = torch.sqrt(fgx.square() + fgy.square() + 1e-6).mean(dims)
    grad_b = torch.sqrt(bgx.square() + bgy.square() + 1e-6).mean(dims)
    grad_ratio = grad_f / grad_b.clamp_min(1e-4)
    std = F.relu(std_gain - std_ratio).mean()
    local = F.relu(detail_gain - local_ratio).mean()
    gradient = F.relu(detail_gain - grad_ratio).mean()
    saturation = (F.relu(0.01 - fused) + F.relu(fused - 0.99)).mean()
    total = mean + std + local + gradient + saturation
    return total, {"mean": mean, "std": std, "local": local,
                   "gradient_stat": gradient, "saturation": saturation,
                   "std_ratio": std_ratio.mean(), "local_ratio": local_ratio.mean(),
                   "gradient_ratio": grad_ratio.mean()}


def source_correlation_guard(fused, baseline, infrared, visible, margin=0.0):
    def correlation(left, right):
        left = left.flatten(1) - left.flatten(1).mean(1, keepdim=True)
        right = right.flatten(1) - right.flatten(1).mean(1, keepdim=True)
        return F.cosine_similarity(left, right, dim=1)
    current = correlation(fused, infrared) + correlation(fused, visible)
    reference = correlation(baseline.detach(), infrared) + correlation(baseline.detach(), visible)
    return F.relu(reference + margin - current).mean()


def supervised_route_loss(probabilities, labels):
    length = probabilities.shape[1]
    height = int(round(length ** 0.5))
    if height * height != length:
        raise ValueError("V8 route supervision expects square training crops")
    width = height
    target = F.interpolate(labels[:, None].float(), (height, width), mode="nearest")
    target = target[:, 0].long().flatten(1)
    return F.nll_loss(probabilities.clamp_min(1e-7).log().transpose(1, 2), target)


def semantic_and_route_loss(outputs, labels, class_weights):
    logits = outputs["semantic_logits"]
    semantic = F.cross_entropy(logits, labels, weight=class_weights)
    semantic = semantic + foreground_dice_loss(logits, labels)
    route3 = supervised_route_loss(outputs["route3"], labels)
    route4 = supervised_route_loss(outputs["route4"], labels)
    return semantic, 0.5 * (route3 + route4)


def decomposition_gate_loss(outputs, infrared, visible):
    with torch.no_grad():
        sal_ir, sal_vi = _saliency(infrared), _saliency(visible)
        target = (sal_ir >= sal_vi).to(infrared.dtype)
        target = F.interpolate(target, outputs["split4"]["gate"].shape[-2:], mode="nearest")
    gate = outputs["split4"]["gate"].mean(1, keepdim=True)
    return F.mse_loss(gate, target)


__all__ = ["VISUAL_NAMES", "visual_terms", "normalized_pareto_guard",
           "statistic_gain_guard", "source_correlation_guard",
           "semantic_and_route_loss", "decomposition_gate_loss"]
