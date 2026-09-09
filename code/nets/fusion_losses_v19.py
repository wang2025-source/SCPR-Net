"""Hard-target differentiable objectives for V19 region-MoE training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nets.fusion_losses_v2 import _sobel


def _corr(a, b):
    a = a.flatten(1); b = b.flatten(1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return F.cosine_similarity(a, b, dim=1)


def _local_std(x, k=9):
    m = F.avg_pool2d(x, k, 1, k // 2)
    q = F.avg_pool2d(x.square(), k, 1, k // 2)
    return (q - m.square()).clamp_min(1e-8).sqrt()


def differentiable_entropy(x, bins=64):
    """Smooth histogram entropy in bits, batch-wise."""
    x = x.float().flatten(1)
    centers = torch.linspace(0., 1., bins, device=x.device, dtype=x.dtype)
    assignment = torch.exp(-.5 * ((x[..., None] - centers) * bins).square())
    histogram = assignment.sum(1)
    probability = histogram / histogram.sum(1, keepdim=True).clamp_min(1e-8)
    return -(probability * torch.log2(probability.clamp_min(1e-8))).sum(1)


def _gaussian_kernel(size, device, dtype):
    sigma = size / 5.0
    axis = torch.arange(size, device=device, dtype=dtype) - (size - 1.) / 2.
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2. * sigma * sigma))
    return (kernel / kernel.sum())[None, None]


def differentiable_pair_vif(reference, distorted, noise_variance=2.0):
    """Torch equivalent of repro.evaluate_metrics.pair_vif."""
    reference = reference.float() * 255.; distorted = distorted.float() * 255.
    numerator = reference.new_zeros(reference.shape[0])
    denominator = reference.new_zeros(reference.shape[0]); eps = 1e-8
    for level, size in enumerate((17, 9, 5, 3)):
        kernel = _gaussian_kernel(size, reference.device, reference.dtype)
        if level:
            reference = F.conv2d(reference, kernel, stride=2)
            distorted = F.conv2d(distorted, kernel, stride=2)
        mr = F.conv2d(reference, kernel); md = F.conv2d(distorted, kernel)
        vr = (F.conv2d(reference.square(), kernel) - mr.square()).clamp_min(0.)
        vd = (F.conv2d(distorted.square(), kernel) - md.square()).clamp_min(0.)
        cov = F.conv2d(reference * distorted, kernel) - mr * md
        gain = cov / (vr + eps); residual = vd - gain * cov
        valid_r = vr >= eps; valid_d = vd >= eps
        gain = torch.where(valid_r & valid_d, gain, torch.zeros_like(gain))
        residual = torch.where(valid_r, residual, vd)
        residual = torch.where(valid_d, residual, torch.zeros_like(residual))
        negative = gain < 0
        residual = torch.where(negative, vd, residual).clamp_min(eps)
        gain = gain.clamp_min(0.); dims = (1, 2, 3)
        numerator += torch.log10(1. + gain.square() * vr / (residual + noise_variance)).sum(dims)
        denominator += torch.log10(1. + vr / noise_variance).sum(dims)
    return numerator / denominator.clamp_min(eps)


def metric_target_losses(fused, infrared, visible, baseline,
                         sd_gain=1.031, detail_gain=1.049,
                         correlation_gain=1.015, information_gain=1.033,
                         entropy_gain=1.037):
    """Six metric-aligned surrogates, expressed as baseline-relative hinges."""
    dims = (-2, -1); base = baseline.detach()
    # Entropy proxy: global and local dispersion with a saturation guard.
    std_f = fused.std(dims, unbiased=False)
    std_b = base.std(dims, unbiased=False).clamp_min(1e-4)
    local_f = _local_std(fused).mean(dims)
    local_b = _local_std(base).mean(dims).clamp_min(1e-4)
    entropy_f = differentiable_entropy(fused)
    with torch.no_grad(): entropy_b = differentiable_entropy(base)
    entropy = (F.relu(entropy_gain * entropy_b - entropy_f).mean()
               + F.relu(sd_gain - std_f / std_b).mean()
               + .5 * F.relu(1.02 - local_f / local_b).mean())
    saturation = (F.relu(.005 - fused) + F.relu(fused - .995)).mean()

    fx, fy = _sobel(fused); bx, by = _sobel(base)
    grad_f = torch.sqrt(fx.square() + fy.square() + 1e-6).mean(dims)
    grad_b = torch.sqrt(bx.square() + by.square() + 1e-6).mean(dims).clamp_min(1e-4)
    dx_f = (fused[..., 1:] - fused[..., :-1]).square().mean(dims).sqrt()
    dy_f = (fused[..., 1:, :] - fused[..., :-1, :]).square().mean(dims).sqrt()
    dx_b = (base[..., 1:] - base[..., :-1]).square().mean(dims).sqrt()
    dy_b = (base[..., 1:, :] - base[..., :-1, :]).square().mean(dims).sqrt()
    sf_f = torch.sqrt(dx_f.square() + dy_f.square() + 1e-8)
    sf_b = torch.sqrt(dx_b.square() + dy_b.square() + 1e-8).clamp_min(1e-4)
    detail_loss = (F.relu(detail_gain - grad_f / grad_b).mean()
                   + F.relu(detail_gain - sf_f / sf_b).mean())

    corr_f = _corr(fused, infrared) + _corr(fused, visible)
    corr_b = (_corr(base, infrared) + _corr(base, visible)).detach()
    correlation = F.relu(correlation_gain * corr_b - corr_f).mean()

    # Match the exact four-scale VIF definition used by formal evaluation.
    with torch.cuda.amp.autocast(enabled=False):
        current_vif = (differentiable_pair_vif(infrared, fused)
                       + differentiable_pair_vif(visible, fused))
        with torch.no_grad():
            baseline_vif = (differentiable_pair_vif(infrared, base)
                            + differentiable_pair_vif(visible, base))
        information = F.relu(information_gain * baseline_vif - current_vif).mean()
    return {"entropy": entropy + .25 * saturation,
            "detail_metric": detail_loss,
            "correlation_metric": correlation,
            "information": information}


class WorstObjectiveFirst:
    """DCEvo-inspired detached dynamic weighting of the hardest objectives."""
    def __init__(self, names, momentum=.95, temperature=.35,
                 minimum=(.30, .25, .15, .20)):
        self.names = tuple(names); self.momentum = momentum
        self.temperature = temperature; self.minimum = tuple(minimum); self.ema = None

    def __call__(self, losses):
        values = torch.stack([losses[n] for n in self.names])
        detached = values.detach().clamp_min(1e-6)
        if self.ema is None: self.ema = detached
        else: self.ema = self.momentum * self.ema + (1-self.momentum) * detached
        normalized = detached / self.ema.clamp_min(1e-6)
        dynamic = torch.softmax(normalized / self.temperature, 0)
        minimum = values.new_tensor(self.minimum)
        if minimum.numel() != values.numel() or float(minimum.sum()) >= 1.:
            raise ValueError("Invalid objective minimum weights")
        weights = (minimum + (1. - minimum.sum()) * dynamic).detach()
        return (weights * values).sum(), weights


__all__ = ["metric_target_losses", "differentiable_entropy",
           "differentiable_pair_vif",
           "WorstObjectiveFirst"]
