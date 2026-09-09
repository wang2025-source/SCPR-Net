"""End-to-end EMMA V7 with source-query fusion and paired attentive SSM.

The pretrained EMMA is used as initialization, never as a frozen output
teacher.  Its four concatenation fusion blocks are replaced by SIBA-inspired
source-query blocks at shallow scales and paired semantic-routing ASSM blocks
at deep scales.  MRFS-inspired interactive gated skips and DCEvo-inspired
low/high discriminative enhancement replace passive decoder skip concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nets.Ufuser import LayerNorm, Mlp, Restormer_CNN_block, Ufuser
from nets.assm_mambairv2 import SelectiveScanASE, _gather_tokens, _reverse_index
from nets.joint_assm_v3 import CommonPrivateAdapter


def _heads(channels: int) -> int:
    return 4 if channels % 4 == 0 else 1


class SourceBoostSuppress(nn.Module):
    """Compact CBSM adapted from the official SIBA implementation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(4, channels // 4)
        self.body = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, padding_mode="reflect"),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, source, size):
        source = F.interpolate(source, size=size, mode="bilinear", align_corners=False)
        feature = self.body(source)
        return F.prelu(feature * self.channel(feature), feature.new_tensor(0.25))


class SourceQueryAttention(nn.Module):
    """Official SIBA convention: source map is Q; opposite feature is K/V."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = _heads(channels)
        self.scale = nn.Parameter(torch.ones(self.heads, 1, 1))
        self.kv = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1, bias=False),
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1,
                      groups=2 * channels, bias=False),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, feature, query):
        batch, channels, height, width = feature.shape
        key, value = self.kv(feature).chunk(2, 1)
        query = rearrange(query, "b (h c) x y -> b h c (x y)", h=self.heads)
        key = rearrange(key, "b (h c) x y -> b h c (x y)", h=self.heads)
        value = rearrange(value, "b (h c) x y -> b h c (x y)", h=self.heads)
        query = F.normalize(query, dim=-1); key = F.normalize(key, dim=-1)
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(-1)
        output = attention @ value
        output = rearrange(output, "b h c (x y) -> b (h c) x y", x=height, y=width)
        return self.project(output)


class SourceQueryBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm(channels, "WithBias")
        self.attention = SourceQueryAttention(channels)
        self.norm2 = LayerNorm(channels, "WithBias")
        self.ffn = Mlp(channels, channels, ffn_expansion_factor=2)

    def forward(self, feature, query):
        feature = feature + self.attention(self.norm1(feature), query)
        return feature + self.ffn(self.norm2(feature))


class SIBAShallowFusion(nn.Module):
    """Four raw/inverted source-query paths from SIBA, used at scales 1/2."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q_ir = SourceBoostSuppress(channels)
        self.q_ir_inverse = SourceBoostSuppress(channels)
        self.q_vi = SourceBoostSuppress(channels)
        self.q_vi_inverse = SourceBoostSuppress(channels)
        self.ir_to_vi = SourceQueryBlock(channels)
        self.iri_to_vi = SourceQueryBlock(channels)
        self.vi_to_ir = SourceQueryBlock(channels)
        self.vii_to_ir = SourceQueryBlock(channels)
        self.fuse = Restormer_CNN_block(4 * channels, channels)

    def forward(self, raw_ir, raw_vi, infrared, visible):
        size = infrared.shape[-2:]
        q_ir = self.q_ir(raw_ir, size)
        q_iri = self.q_ir_inverse(1.0 - raw_ir, size)
        q_vi = self.q_vi(raw_vi, size)
        q_vii = self.q_vi_inverse(1.0 - raw_vi, size)
        paths = (
            self.ir_to_vi(visible, q_ir), self.iri_to_vi(visible, q_iri),
            self.vi_to_ir(infrared, q_vi), self.vii_to_ir(infrared, q_vii),
        )
        return self.fuse(torch.cat(paths, 1))


class PairedSemanticASSMFusion(nn.Module):
    """Deterministic paired SGN + forward/reverse ASE scans for two modalities."""

    def __init__(self, channels=32, d_state=16, num_tokens=8, expand=2.0) -> None:
        super().__init__()
        hidden = int(channels * expand)
        self.channels = channels; self.hidden = hidden
        self.num_tokens = num_tokens; self.d_state = d_state
        self.norm_ir = nn.GroupNorm(1, channels)
        self.norm_vi = nn.GroupNorm(1, channels)
        self.router = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1), nn.GELU(),
            nn.Conv2d(channels, num_tokens, 1),
        )
        self.embedding_b = nn.Embedding(num_tokens, 16)
        self.embedding_a = nn.Embedding(16, d_state)
        self.in_ir = nn.Conv2d(channels, hidden, 1)
        self.in_vi = nn.Conv2d(channels, hidden, 1)
        self.cpe_ir = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.cpe_vi = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.scan = SelectiveScanASE(hidden, d_state)
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, channels)
        self.ir_scale = nn.Parameter(torch.full((1, channels, 1, 1), 5e-2))
        self.vi_scale = nn.Parameter(torch.full((1, channels, 1, 1), 5e-2))
        self.common = nn.Sequential(
            nn.Conv2d(4 * channels, 2 * channels, 1), nn.GELU(),
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1,
                      groups=2 * channels), nn.GELU(),
            nn.Conv2d(2 * channels, channels, 1),
        )
        self.fuse = Restormer_CNN_block(3 * channels, channels)
        nn.init.uniform_(self.embedding_b.weight, -1 / num_tokens, 1 / num_tokens)
        nn.init.uniform_(self.embedding_a.weight, -1 / 16, 1 / 16)

    def forward(self, infrared, visible):
        batch, _, height, width = infrared.shape
        ir = self.norm_ir(infrared); vi = self.norm_vi(visible)
        route_input = torch.cat((ir, vi, ir * vi, torch.abs(ir - vi)), 1)
        probabilities = self.router(route_input).flatten(2).transpose(1, 2).softmax(-1)
        route_index = probabilities.detach().argmax(-1)
        order = torch.argsort(route_index, dim=-1, stable=True)
        reverse = _reverse_index(order)
        dictionary = self.embedding_b.weight @ self.embedding_a.weight
        prompt = _gather_tokens(probabilities @ dictionary, order)
        prompt = torch.stack((prompt, prompt), 2).flatten(1, 2)

        ir_tokens = self.in_ir(ir); ir_tokens = ir_tokens * torch.sigmoid(self.cpe_ir(ir_tokens))
        vi_tokens = self.in_vi(vi); vi_tokens = vi_tokens * torch.sigmoid(self.cpe_vi(vi_tokens))
        ir_tokens = _gather_tokens(ir_tokens.flatten(2).transpose(1, 2), order)
        vi_tokens = _gather_tokens(vi_tokens.flatten(2).transpose(1, 2), order)
        sequence = torch.stack((ir_tokens, vi_tokens), 2).flatten(1, 2)
        forward = self.scan(sequence, prompt)
        backward = self.scan(sequence.flip(1), prompt.flip(1)).flip(1)
        sequence = self.out(self.out_norm(0.5 * (forward + backward)))
        sequence = sequence.reshape(batch, height * width, 2, self.channels)
        ir_delta = _gather_tokens(sequence[:, :, 0], reverse).transpose(1, 2).reshape_as(ir)
        vi_delta = _gather_tokens(sequence[:, :, 1], reverse).transpose(1, 2).reshape_as(vi)
        ir_out = infrared + self.ir_scale * ir_delta
        vi_out = visible + self.vi_scale * vi_delta
        common = self.common(torch.cat((ir_out, vi_out, ir_out * vi_out,
                                        torch.abs(ir_out - vi_out)), 1))
        fused = self.fuse(torch.cat((ir_out, vi_out, common), 1))
        return fused, ir_out, vi_out, common, probabilities


class InteractiveGatedMix(nn.Module):
    """MRFS IGM-Att: channel/spatial reliability plus reciprocal completion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(4, channels // 2)
        self.channel = nn.Sequential(
            nn.Linear(4 * channels, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2 * channels), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 16, 1), nn.ReLU(True), nn.Conv2d(16, 2, 1), nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels, 1), nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, deep, skip):
        batch, channels, _, _ = deep.shape
        joined = torch.cat((deep, skip), 1)
        pooled = torch.cat((F.adaptive_avg_pool2d(joined, 1).flatten(1),
                            F.adaptive_max_pool2d(joined, 1).flatten(1)), 1)
        channel = self.channel(pooled).reshape(batch, 2, channels, 1, 1)
        statistics = torch.cat((deep.mean(1, True), deep.amax(1, True),
                                skip.mean(1, True), skip.amax(1, True)), 1)
        spatial = self.spatial(statistics).unsqueeze(2)
        mixed = channel * spatial
        gate = self.gate(joined)
        deep_out = deep + self.scale * (1.0 - gate) * mixed[:, 1] * skip
        skip_out = skip + self.scale * gate * mixed[:, 0] * deep
        return deep_out, skip_out


class DiscriminativeEnhancer(nn.Module):
    """DCEvo-inspired explicit low/high-frequency decoder enhancement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.low = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.high = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.select = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(2 * channels, channels, 1),
            nn.GELU(), nn.Conv2d(channels, 2 * channels, 1),
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        low_seed = F.avg_pool2d(x, 5, stride=1, padding=2)
        low = self.low(low_seed)
        high = self.high(x - low_seed)
        weights = self.select(torch.cat((low, high), 1))
        weights = weights.reshape(x.shape[0], 2, x.shape[1], 1, 1).softmax(1)
        enhanced = weights[:, 0] * low + weights[:, 1] * high
        return x + self.scale * self.project(enhanced)


class SemanticFeedback(nn.Module):
    """Cross-dimensional task feature modulation inspired by DCEvo CDE."""

    def __init__(self, semantic_channels: int, target_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(semantic_channels, target_channels, 1), nn.GELU(),
            nn.Conv2d(target_channels, 2 * target_channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, feature, semantic):
        semantic = F.interpolate(semantic, feature.shape[-2:], mode="bilinear",
                                 align_corners=False)
        gamma, beta = self.project(semantic).chunk(2, 1)
        return feature * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * beta


class UfuserEnd2EndV7(nn.Module):
    def __init__(self, d_state=16, num_tokens=8, num_classes=9) -> None:
        super().__init__()
        self.emma = Ufuser()
        self.fuse1 = SIBAShallowFusion(8)
        self.fuse2 = SIBAShallowFusion(16)
        self.fuse3 = PairedSemanticASSMFusion(32, d_state, num_tokens)
        self.fuse4 = PairedSemanticASSMFusion(32, d_state, num_tokens)
        self.fusion_scale1 = nn.Parameter(torch.zeros(1, 8, 1, 1))
        self.fusion_scale2 = nn.Parameter(torch.zeros(1, 16, 1, 1))
        self.fusion_scale3 = nn.Parameter(torch.zeros(1, 32, 1, 1))
        self.fusion_scale4 = nn.Parameter(torch.zeros(1, 32, 1, 1))
        self.decompose4 = CommonPrivateAdapter(32)
        self.deep_fuse4 = Restormer_CNN_block(3 * 32, 32)

        self.skip3 = InteractiveGatedMix(32)
        self.skip2 = InteractiveGatedMix(16)
        self.skip1 = InteractiveGatedMix(8)
        self.enhance4 = DiscriminativeEnhancer(32)
        self.enhance3 = DiscriminativeEnhancer(32)
        self.enhance2 = DiscriminativeEnhancer(16)
        self.enhance1 = DiscriminativeEnhancer(8)

        self.semantic_context = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, padding_mode="reflect"), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode="reflect"), nn.GELU(),
        )
        self.semantic_head = nn.Conv2d(32, num_classes, 1)
        self.feedback3 = SemanticFeedback(32, 32)
        self.feedback2 = SemanticFeedback(32, 16)
        self.feedback1 = SemanticFeedback(32, 8)
        self.aux3 = nn.Conv2d(32, 1, 1)
        self.aux2 = nn.Conv2d(16, 1, 1)

    def load_emma(self, state):
        self.emma.load_state_dict(state, strict=True)

    def forward_features(self, infrared, visible):
        b = self.emma
        i1 = b.I_en_1(infrared); i2 = b.I_en_2(b.I_down1(i1))
        i3 = b.I_en_3(b.I_down2(i2)); i4 = b.I_en_4(b.I_down3(i3))
        v1 = b.V_en_1(visible); v2 = b.V_en_2(b.V_down1(v1))
        v3 = b.V_en_3(b.V_down2(v2)); v4 = b.V_en_4(b.V_down3(v3))

        base_f1 = b.f_1(torch.cat((i1, v1), 1))
        base_f2 = b.f_2(torch.cat((i2, v2), 1))
        base_f3 = b.f_3(torch.cat((i3, v3), 1))
        base_f4 = b.f_4(torch.cat((i4, v4), 1))
        new_f1 = self.fuse1(infrared, visible, i1, v1)
        new_f2 = self.fuse2(infrared, visible, i2, v2)
        new_f3, i3j, v3j, c3, route3 = self.fuse3(i3, v3)
        f4_raw, i4j, v4j, c4, route4 = self.fuse4(i4, v4)
        f1 = base_f1 + self.fusion_scale1 * torch.tanh(new_f1)
        f2 = base_f2 + self.fusion_scale2 * torch.tanh(new_f2)
        f3 = base_f3 + self.fusion_scale3 * torch.tanh(new_f3)
        split4 = self.decompose4(i4j, v4j, c4)
        new_f4 = self.deep_fuse4(torch.cat((f4_raw, split4["common"],
                                            split4["private_ir"] + split4["private_vi"]), 1))
        f4 = base_f4 + self.fusion_scale4 * torch.tanh(new_f4)

        semantic_seed = F.interpolate(f4, f3.shape[-2:], mode="bilinear", align_corners=False)
        semantic = self.semantic_context(torch.cat((f3, semantic_seed), 1))
        semantic_logits = F.interpolate(self.semantic_head(semantic), infrared.shape[-2:],
                                        mode="bilinear", align_corners=False)

        d4 = self.enhance4(b.de_4(f4))
        up3, skip3 = self.skip3(b.up4(d4), f3)
        d3 = self.enhance3(b.de_3(torch.cat((up3, skip3), 1)))
        d3 = self.feedback3(d3, semantic)
        up2, skip2 = self.skip2(b.up3(d3), f2)
        d2 = self.enhance2(b.de_2(torch.cat((up2, skip2), 1)))
        d2 = self.feedback2(d2, semantic)
        up1, skip1 = self.skip1(b.up2(d2), f1)
        d1 = self.enhance1(b.de_1(torch.cat((up1, skip1), 1)))
        d1 = self.feedback1(d1, semantic)
        fused = b.last(d1)
        aux3 = torch.sigmoid(F.interpolate(self.aux3(d3), infrared.shape[-2:],
                                           mode="bilinear", align_corners=False))
        aux2 = torch.sigmoid(F.interpolate(self.aux2(d2), infrared.shape[-2:],
                                           mode="bilinear", align_corners=False))
        return {"fused": fused, "aux3": aux3, "aux2": aux2,
                "semantic_logits": semantic_logits, "split4": split4,
                "source_ir4": i4j, "source_vi4": v4j,
                "route3": route3, "route4": route4}

    def forward(self, infrared, visible, return_aux=False):
        outputs = self.forward_features(infrared, visible)
        return outputs if return_aux else outputs["fused"]
