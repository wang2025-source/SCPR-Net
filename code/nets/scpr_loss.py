"""The four-term training objective in manuscript Section 3.4 (Eqs. 19--23).

The manuscript does not specify numerical loss weights or the exact saliency
mask estimator. These are explicit configuration choices: by default the
labelled foreground supplies Omega_I, its complement supplies Omega_V, and
all outer weights are 1. No reported benchmark metric enters this objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nets.scpr_net import haar_decompose


@dataclass(frozen=True)
class LossConfig:
    visual: float = 1.0
    frequency: float = 1.0
    semantic: float = 1.0
    route: float = 1.0
    omega_ir: float = 1.0
    omega_vi: float = 1.0
    omega_ssim: float = 1.0
    beta_foreground: float = 1.0
    beta_edge: float = 1.0
    rho: float = 0.08
    ignore_index: int = 255

    def to_dict(self):
        return asdict(self)


def masked_mean(values, mask):
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1)


def ssim(x, y):
    coordinates = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    gaussian = torch.exp(-(coordinates ** 2) / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None]
    def smooth(value):
        return F.conv2d(value, window, padding=5)
    mx, my = smooth(x), smooth(y)
    vx, vy = smooth(x * x) - mx * mx, smooth(y * y) - my * my
    covariance = smooth(x * y) - mx * my
    return (((2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) /
            ((mx * mx + my * my + 0.01 ** 2) * (vx + vy + 0.03 ** 2)).clamp_min(1e-8)).mean()


def sobel(image):
    kernel = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    return F.conv2d(image, kernel, padding=1), F.conv2d(image, kernel.transpose(-1, -2), padding=1)


class SCPRLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LossConfig()

    def forward(self, outputs, infrared, visible, labels):
        config = self.config
        fused = outputs["fused"]
        logits = outputs["semantic_logits"]
        valid = (labels >= 0) & (labels < logits.shape[1]) & (labels != config.ignore_index)
        if not valid.any():
            raise ValueError("A training batch must contain at least one valid semantic label.")
        foreground = ((labels != 0) & valid).to(fused.dtype)[:, None]
        background = ((labels == 0) & valid).to(fused.dtype)[:, None]
        visual = (config.omega_ir * masked_mean((fused - infrared).abs(), foreground)
                  + config.omega_vi * masked_mean((fused - visible).abs(), background)
                  + config.omega_ssim * (1 - ssim(fused, visible)))
        frequency = sum((left - right).abs().mean()
                        for left, right in zip(haar_decompose(fused)[1:], haar_decompose(visible)[1:]))

        target = labels.masked_fill(~valid, config.ignore_index)
        cross_entropy = F.cross_entropy(logits, target, ignore_index=config.ignore_index)
        safe_labels = labels.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_labels, logits.shape[1]).permute(0, 3, 1, 2).to(fused.dtype)
        mask = valid[:, None].to(fused.dtype)
        probability = logits.softmax(1) * mask
        one_hot = one_hot * mask
        dice = 1 - ((2 * (probability * one_hot).sum((0, 2, 3)) + 1e-6) /
                    (probability.sum((0, 2, 3)) + one_hot.sum((0, 2, 3)) + 1e-6)).mean()
        foreground_loss = masked_mean((fused - infrared).abs(), foreground)
        boundary = F.max_pool2d(foreground, 3, 1, 1) + F.max_pool2d(-foreground, 3, 1, 1)
        boundary = boundary.clamp(0, 1) * mask
        fx, fy = sobel(fused)
        ix, iy = sobel(infrared)
        vx, vy = sobel(visible)
        edge = masked_mean((fx - torch.where(ix.abs() >= vx.abs(), ix, vx)).abs()
                           + (fy - torch.where(iy.abs() >= vy.abs(), iy, vy)).abs(), boundary)
        semantic = cross_entropy + dice + config.beta_foreground * foreground_loss + config.beta_edge * edge
        route = sum(F.relu(config.rho - outputs[key].mean((0, 2, 3))[:2]).sum()
                    for key in ("low_weights", "coarse_weights", "fine_weights")) / 3
        total = config.visual * visual + config.frequency * frequency + config.semantic * semantic + config.route * route
        return {"total": total, "visual": visual, "frequency": frequency, "semantic": semantic, "route": route}
