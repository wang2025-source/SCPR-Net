"""Joint cross-modal attentive state-space blocks for EMMA V3.

Unlike sharing one ASSM module between two independent forward calls, this
module interleaves infrared and visible tokens, routes them jointly, and runs
one selective scan over the joint sequence.  State propagation therefore
crosses the modality boundary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.assm_mambairv2 import SelectiveScanASE, _gather_tokens, _reverse_index


class JointCrossModalASSM(nn.Module):
    """One semantic scan over interleaved IR/VI tokens."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        num_tokens: int = 8,
        inner_rank: int = 16,
        expand: float = 2.0,
        layer_scale_init: float = 5e-2,
    ) -> None:
        super().__init__()
        hidden = int(dim * expand)
        self.dim = dim
        self.d_state = d_state
        self.num_tokens = num_tokens

        self.input_norm = nn.GroupNorm(1, dim)
        self.modality_embedding = nn.Parameter(torch.zeros(2, dim))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        self.route = nn.Sequential(
            nn.Linear(dim, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), num_tokens),
            nn.LogSoftmax(dim=-1),
        )
        self.embedding_b = nn.Embedding(num_tokens, inner_rank)
        self.embedding_a = nn.Embedding(inner_rank, d_state)
        nn.init.uniform_(self.embedding_b.weight, -1 / num_tokens, 1 / num_tokens)
        nn.init.uniform_(self.embedding_a.weight, -1 / inner_rank, 1 / inner_rank)

        self.in_proj = nn.Conv2d(dim, hidden, 1)
        self.cpe = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.scan = SelectiveScanASE(hidden, d_state=d_state)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

        self.cross_gate_ir = nn.Conv2d(3 * dim, dim, 1)
        self.cross_gate_vi = nn.Conv2d(3 * dim, dim, 1)
        self.common = nn.Sequential(
            nn.Conv2d(4 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.ir_scale = nn.Parameter(
            torch.full((1, dim, 1, 1), layer_scale_init)
        )
        self.vi_scale = nn.Parameter(
            torch.full((1, dim, 1, 1), layer_scale_init)
        )

    @staticmethod
    def _interleave(ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, 2L, C] with IR/VI adjacent at every location.
        return torch.stack((ir, vi), dim=2).flatten(1, 2)

    def forward(
        self, infrared: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if infrared.shape != visible.shape:
            raise ValueError("Joint ASSM inputs must have identical shapes")
        batch, channels, height, width = infrared.shape
        length = height * width

        ir_norm = self.input_norm(infrared)
        vi_norm = self.input_norm(visible)
        ir_route = ir_norm.flatten(2).transpose(1, 2) + self.modality_embedding[0]
        vi_route = vi_norm.flatten(2).transpose(1, 2) + self.modality_embedding[1]
        route_tokens = self._interleave(ir_route, vi_route)
        route_logits = self.route(route_tokens)
        if self.training:
            policy = F.gumbel_softmax(route_logits, hard=True, dim=-1)
        else:
            policy = F.one_hot(
                route_logits.argmax(dim=-1), num_classes=self.num_tokens
            ).to(dtype=route_logits.dtype)
        dictionary = self.embedding_b.weight @ self.embedding_a.weight
        prompt = policy @ dictionary

        route_index = policy.detach().argmax(dim=-1)
        sort_index = torch.argsort(route_index, dim=-1, stable=False)
        reverse_index = _reverse_index(sort_index)

        ir_feature = self.in_proj(ir_norm)
        vi_feature = self.in_proj(vi_norm)
        ir_feature = ir_feature * torch.sigmoid(self.cpe(ir_feature))
        vi_feature = vi_feature * torch.sigmoid(self.cpe(vi_feature))
        ir_feature = ir_feature.flatten(2).transpose(1, 2)
        vi_feature = vi_feature.flatten(2).transpose(1, 2)
        features = self._interleave(ir_feature, vi_feature)
        features = _gather_tokens(features, sort_index)
        prompt = _gather_tokens(prompt, sort_index)
        features = self.scan(features, prompt)
        features = self.out_proj(self.out_norm(features))
        features = _gather_tokens(features, reverse_index)

        features = features.reshape(batch, length, 2, channels)
        ir_scan = features[:, :, 0].transpose(1, 2).reshape(
            batch, channels, height, width
        )
        vi_scan = features[:, :, 1].transpose(1, 2).reshape(
            batch, channels, height, width
        )

        preliminary_common = 0.5 * (ir_scan + vi_scan)
        gate_input = torch.cat((ir_scan, vi_scan, preliminary_common), dim=1)
        ir_delta = ir_scan + torch.sigmoid(self.cross_gate_ir(gate_input)) * vi_scan
        vi_delta = vi_scan + torch.sigmoid(self.cross_gate_vi(gate_input)) * ir_scan
        infrared_out = infrared + self.ir_scale * ir_delta
        visible_out = visible + self.vi_scale * vi_delta
        common = self.common(
            torch.cat(
                (
                    infrared_out,
                    visible_out,
                    infrared_out * visible_out,
                    torch.abs(infrared_out - visible_out),
                ),
                dim=1,
            )
        )
        return infrared_out, visible_out, common


class CommonPrivateAdapter(nn.Module):
    """Explicit common/private representation with reconstructable outputs."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.common_ir = nn.Conv2d(channels, channels, 3, padding=1)
        self.common_vi = nn.Conv2d(channels, channels, 3, padding=1)
        self.common_fuse = nn.Sequential(
            nn.Conv2d(5 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.private_ir = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.private_vi = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.restore_common_ir = nn.Conv2d(channels, channels, 1)
        self.restore_common_vi = nn.Conv2d(channels, channels, 1)
        self.restore_private_ir = nn.Conv2d(channels, channels, 1)
        self.restore_private_vi = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        infrared: torch.Tensor,
        visible: torch.Tensor,
        joint_common: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        common_ir = self.common_ir(infrared)
        common_vi = self.common_vi(visible)
        common = self.common_fuse(
            torch.cat(
                (
                    common_ir,
                    common_vi,
                    joint_common,
                    common_ir * common_vi,
                    torch.abs(common_ir - common_vi),
                ),
                dim=1,
            )
        )
        private_ir = self.private_ir(
            torch.cat((infrared, infrared - self.restore_common_ir(common)), dim=1)
        )
        private_vi = self.private_vi(
            torch.cat((visible, visible - self.restore_common_vi(common)), dim=1)
        )
        reconstructed_ir = self.restore_common_ir(common) + self.restore_private_ir(
            private_ir
        )
        reconstructed_vi = self.restore_common_vi(common) + self.restore_private_vi(
            private_vi
        )
        return {
            "common": common,
            "common_ir": common_ir,
            "common_vi": common_vi,
            "private_ir": private_ir,
            "private_vi": private_vi,
            "reconstructed_ir": reconstructed_ir,
            "reconstructed_vi": reconstructed_vi,
        }
