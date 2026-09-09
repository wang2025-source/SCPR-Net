"""End-to-end semantic region-MoE successor to V12.

The head is not a bounded add-on gate.  It directly routes low-frequency and
two-scale structural components among infrared, visible and the trainable V12
backbone.  Semantic predictions enter the image path through the router.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Ufuser_v10_region_detail_v12 import UfuserV10RegionDetailV12


def lowpass(x, kernel=7):
    return F.avg_pool2d(x, kernel, 1, kernel // 2)


def detail(x, kernel=7):
    return x - lowpass(x, kernel)


class SemanticRegionMoE(nn.Module):
    """Region-adaptive source experts with semantic-conditioned routing."""
    def __init__(self, classes=9, channels=48, region=8, max_refine=.10):
        super().__init__()
        self.region = int(region)
        self.max_refine = float(max_refine)
        self.stem = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(6, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      padding_mode="reflect"),
            nn.GELU(), nn.Conv2d(channels, channels, 1), nn.GELU())
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 2, 1,
                      padding_mode="reflect"),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=2, dilation=2,
                      groups=channels * 2, padding_mode="reflect"),
            nn.GELU(), nn.Conv2d(channels * 2, channels, 1), nn.GELU())
        self.semantic = nn.Conv2d(channels, classes, 1)
        self.foreground = nn.Conv2d(channels, 1, 1)
        self.router = nn.Sequential(
            nn.Conv2d(2 * channels + 2, channels, 3, padding=1,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(channels, 9, 1))
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 3, padding=1,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(channels, 1, 1))
        # Start almost exactly at V12, while keeping direct router gradients.
        nn.init.zeros_(self.router[-1].weight)
        bias = torch.tensor([-4., -4., 4., -4., -4., 4., -4., -4., 4.])
        with torch.no_grad(): self.router[-1].bias.copy_(bias)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, base, infrared, visible):
        li, lv, lb = lowpass(infrared), lowpass(visible), lowpass(base)
        di, dv, db = detail(infrared), detail(visible), detail(base)
        l3i, l3v, l3b = lowpass(infrared, 3), lowpass(visible, 3), lowpass(base, 3)
        ci, cv, cb = l3i - li, l3v - lv, l3b - lb
        fi, fv, fb = infrared - l3i, visible - l3v, base - l3b
        stem = self.stem(torch.cat((
            infrared, visible, base, (infrared-visible).abs(),
            li, lv, lb, di, dv, db, (di-dv).abs(), (li-lv).abs()), 1))
        context = self.context(stem)
        logits = F.interpolate(self.semantic(context), base.shape[-2:],
                               mode="bilinear", align_corners=False)
        fg_logit = F.interpolate(self.foreground(context), base.shape[-2:],
                                 mode="bilinear", align_corners=False)
        fg = fg_logit.sigmoid()
        context = F.interpolate(context, base.shape[-2:], mode="bilinear",
                                align_corners=False)
        router_input = torch.cat((stem, context, fg, (di-dv).abs()), 1)
        r = min(self.region, *base.shape[-2:])
        route = self.router(F.avg_pool2d(router_input, r, r, ceil_mode=True))
        route = F.interpolate(route, base.shape[-2:], mode="bilinear",
                              align_corners=False)
        low_w = torch.softmax(route[:, 0:3], 1)
        coarse_w = torch.softmax(route[:, 3:6], 1)
        fine_w = torch.softmax(route[:, 6:9], 1)
        # Semantic priors affect routing but do not hard-code a class policy.
        low_bias = torch.cat((.35 * fg, .35 * (1. - fg), torch.zeros_like(fg)), 1)
        low_w = torch.softmax(torch.log(low_w.clamp_min(1e-6)) + low_bias, 1)
        low = (low_w * torch.cat((li, lv, lb), 1)).sum(1, keepdim=True)
        coarse = (coarse_w * torch.cat((ci, cv, cb), 1)).sum(1, keepdim=True)
        fine = (fine_w * torch.cat((fi, fv, fb), 1)).sum(1, keepdim=True)
        # Exact Laplacian reconstruction when all routes select one source.
        routed = low + coarse + fine
        correction = self.max_refine * torch.tanh(self.refine(torch.cat(
            (stem, routed, base, infrared, visible), 1)))
        fused = (routed + correction).clamp(0., 1.)
        return fused, {"semantic_logits": logits,
                       "foreground_logit": fg_logit, "foreground": fg,
                       "low_weights": low_w, "coarse_weights": coarse_w,
                       "fine_weights": fine_w, "routed": routed,
                       "moe_correction": correction}


class UfuserRegionMoEV19(nn.Module):
    """Fully trainable V12 backbone plus mandatory semantic region experts."""
    def __init__(self, v12: UfuserV10RegionDetailV12, classes=9,
                 channels=48, region=8, max_refine=.10):
        super().__init__()
        self.backbone = v12
        self.moe = SemanticRegionMoE(classes, channels, region, max_refine)

    def forward(self, infrared, visible, return_aux=False):
        base = self.backbone(infrared, visible)
        fused, aux = self.moe(base, infrared, visible)
        out = {"fused": fused, "base_v12": base, **aux}
        return out if return_aux else fused


__all__ = ["UfuserRegionMoEV19", "SemanticRegionMoE", "lowpass", "detail"]
