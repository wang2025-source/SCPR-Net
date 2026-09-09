"""Fixed, baseline-guarded objectives for EMMA Pareto-Wavelet V6."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Ufuser_pareto_v6 import haar_dwt
from nets.fusion_losses_v3 import (
    source_adaptive_intensity_loss,
    source_selected_gradient_loss,
)
from nets.fusion_losses_v4 import (
    multiscale_source_laplacian_loss,
    source_selected_contrast_loss,
)


LOSS_NAMES = ("intensity", "gradient", "laplacian", "contrast", "wavelet", "sensing")


def _selected_band_loss(fused_band, infrared_band, visible_band):
    target = torch.where(
        infrared_band.abs() >= visible_band.abs(), infrared_band, visible_band
    )
    return F.smooth_l1_loss(fused_band, target, beta=0.02)


def wavelet_highband_loss(fused, infrared, visible):
    """Match only source-selected Haar detail; never impose a low-frequency target."""
    llf, *hf1 = haar_dwt(fused)
    lli, *hi1 = haar_dwt(infrared)
    llv, *hv1 = haar_dwt(visible)
    _, *hf2 = haar_dwt(llf)
    _, *hi2 = haar_dwt(lli)
    _, *hv2 = haar_dwt(llv)
    level1 = sum(_selected_band_loss(a, b, c) for a, b, c in zip(hf1, hi1, hv1)) / 3
    level2 = sum(_selected_band_loss(a, b, c) for a, b, c in zip(hf2, hi2, hv2)) / 3
    return level1 + 0.5 * level2


def fixed_objective(fused, infrared, visible, sensing_loss: nn.Module, ai, av):
    return {
        "intensity": source_adaptive_intensity_loss(fused, infrared, visible),
        "gradient": source_selected_gradient_loss(fused, infrared, visible),
        "laplacian": multiscale_source_laplacian_loss(fused, infrared, visible),
        "contrast": source_selected_contrast_loss(fused, infrared, visible),
        "wavelet": wavelet_highband_loss(fused, infrared, visible),
        "sensing": sensing_loss(av(fused), visible) + sensing_loss(ai(fused), infrared),
    }


def weighted_objective(terms, coefficients):
    return sum(coefficients[name] * terms[name] for name in LOSS_NAMES)


def objective_pareto_guard(terms, base_terms, margin=0.0):
    """Dimensionless hinge per objective, avoiding domination by loss scale."""
    guards = {
        name: F.relu(terms[name] / base_terms[name].detach().clamp_min(1e-6)
                     - (1.0 - margin))
        for name in LOSS_NAMES
    }
    return sum(guards.values()), guards


def _local_std(x, window=9):
    mean = F.avg_pool2d(x, window, stride=1, padding=window // 2)
    square = F.avg_pool2d(x.square(), window, stride=1, padding=window // 2)
    return (square - mean.square()).clamp_min(1e-8).sqrt()


def moment_guards(fused, base, mean_tolerance=2.0 / 255.0):
    """Protect brightness, global dispersion and local contrast of frozen EMMA."""
    dimensions = (-2, -1)
    mean_shift = (fused.mean(dimensions) - base.mean(dimensions)).abs()
    mean = F.relu(mean_shift - mean_tolerance).mean()
    std_fused = fused.std(dimensions, unbiased=False)
    std_base = base.std(dimensions, unbiased=False).detach()
    std = F.relu(1.0 - std_fused / std_base.clamp_min(1e-4)).mean()
    local_fused = _local_std(fused).mean(dimensions)
    local_base = _local_std(base).mean(dimensions).detach()
    local_std = F.relu(1.0 - local_fused / local_base.clamp_min(1e-4)).mean()
    return mean + std + local_std, {"mean": mean, "std": std, "local_std": local_std}
