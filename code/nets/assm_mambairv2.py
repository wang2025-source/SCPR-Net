"""Attentive State-Space (ASSM) building blocks for EMMA.

This is a compact adaptation of the official MambaIRv2 ASSM implementation
(Guo et al., CVPR 2025, Apache-2.0).  It retains the two defining mechanisms:

* SGN: route pixels into semantic neighbourhoods and scan the sorted sequence.
* ASE: inject a learned semantic prompt into the selective-scan C term.

Unlike the previous EMMA-Mamba2 ablation, this module uses one semantic scan,
not four spatial scan directions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


def _reverse_index(index: torch.Tensor) -> torch.Tensor:
    reverse = torch.empty_like(index)
    values = torch.arange(index.shape[-1], device=index.device).expand_as(index)
    reverse.scatter_(1, index, values)
    return reverse


def _gather_tokens(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, index.unsqueeze(-1).expand_as(x))


class SelectiveScanASE(nn.Module):
    """Single selective scan with MambaIRv2's attentive C term."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError("mamba_ssm is required for ASSM")
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = math.ceil(dim / 16)

        x_proj = nn.Linear(dim, self.dt_rank + 2 * d_state, bias=False)
        self.x_proj_weight = nn.Parameter(x_proj.weight.unsqueeze(0))

        dt_proj = nn.Linear(self.dt_rank, dim, bias=True)
        nn.init.uniform_(dt_proj.weight, -(self.dt_rank ** -0.5), self.dt_rank ** -0.5)
        dt = torch.exp(
            torch.rand(dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp_min(1e-4)
        with torch.no_grad():
            dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        dt_proj.bias._no_reinit = True
        self.dt_proj_weight = nn.Parameter(dt_proj.weight.unsqueeze(0))
        self.dt_proj_bias = nn.Parameter(dt_proj.bias.unsqueeze(0))

        base = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_logs = nn.Parameter(torch.log(base).repeat(dim, 1))
        self.A_logs._no_weight_decay = True
        self.Ds = nn.Parameter(torch.ones(dim))
        self.Ds._no_weight_decay = True

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C], prompt: [B, L, d_state]
        batch, length, channels = x.shape
        xs = x.transpose(1, 2).unsqueeze(1)
        projected = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, bs, cs = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_proj_weight)

        ys = selective_scan_fn(
            xs.float().reshape(batch, channels, length),
            dts.float().reshape(batch, channels, length),
            -torch.exp(self.A_logs.float()),
            bs.float(),
            cs.float() + prompt.transpose(1, 2).unsqueeze(1).float(),
            self.Ds.float(),
            z=None,
            delta_bias=self.dt_proj_bias.float().reshape(-1),
            delta_softplus=True,
            return_last_state=False,
        )
        return ys.transpose(1, 2).to(dtype=x.dtype)


class ASSM2D(nn.Module):
    """MambaIRv2 SGN + ASE adapted to a BCHW feature map."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        num_tokens: int = 8,
        inner_rank: int = 16,
        expand: float = 2.0,
    ) -> None:
        super().__init__()
        hidden = int(dim * expand)
        self.dim = dim
        self.d_state = d_state
        self.in_proj = nn.Conv2d(dim, hidden, 1)
        self.cpe = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.route = nn.Sequential(
            nn.Linear(dim, max(dim // 3, 4)),
            nn.GELU(),
            nn.Linear(max(dim // 3, 4), num_tokens),
            nn.LogSoftmax(dim=-1),
        )
        self.embedding_b = nn.Embedding(num_tokens, inner_rank)
        self.embedding_a = nn.Embedding(inner_rank, d_state)
        nn.init.uniform_(self.embedding_b.weight, -1 / num_tokens, 1 / num_tokens)
        nn.init.uniform_(self.embedding_a.weight, -1 / inner_rank, 1 / inner_rank)
        self.scan = SelectiveScanASE(hidden, d_state=d_state)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        policy = F.gumbel_softmax(self.route(tokens), hard=True, dim=-1)
        dictionary = self.embedding_b.weight @ self.embedding_a.weight
        prompt = policy @ dictionary

        route_index = policy.detach().argmax(dim=-1)
        sort_index = torch.argsort(route_index, dim=-1, stable=False)
        reverse_index = _reverse_index(sort_index)

        features = self.in_proj(x)
        features = features * torch.sigmoid(self.cpe(features))
        features = features.flatten(2).transpose(1, 2)
        features = _gather_tokens(features, sort_index)
        prompt = _gather_tokens(prompt, sort_index)

        features = self.scan(features, prompt)
        features = self.out_proj(self.out_norm(features))
        features = _gather_tokens(features, reverse_index)
        return features.transpose(1, 2).reshape(batch, channels, height, width)


class ASSMResidual(nn.Module):
    """Pre-norm ASSM with a local convolutional feed-forward residual."""

    def __init__(self, dim: int, d_state: int = 16, num_tokens: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.assm = ASSM2D(dim, d_state=d_state, num_tokens=num_tokens)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
            nn.GELU(),
            nn.Conv2d(2 * dim, 2 * dim, 3, padding=1, groups=2 * dim),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.scale1 = nn.Parameter(torch.full((1, dim, 1, 1), 1e-4))
        self.scale2 = nn.Parameter(torch.full((1, dim, 1, 1), 1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale1 * self.assm(self.norm1(x))
        return x + self.scale2 * self.ffn(self.norm2(x))
