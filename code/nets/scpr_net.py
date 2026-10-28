"""SCPR-Net: the architecture in Sections 3.1--3.3 of the manuscript.

The forward path is encoder -> hierarchical collaboration -> guided decoder
-> region-adaptive detail enhancement -> spatial-semantic frequency routing.
The local/global hybrid block is inherited from the attributed EMMA source.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nets.hybrid_blocks import Restormer_CNN_block


@dataclass(frozen=True)
class SCPRConfig:
    channels: tuple[int, ...] = (8, 16, 32, 64)
    classes: int = 9
    semantic_slots: int = 8
    state_dim: int = 16
    region_size: int = 8
    router_channels: int = 48

    def __post_init__(self):
        if tuple(self.channels) != (8, 16, 32, 64):
            raise ValueError("The manuscript requires channels (8, 16, 32, 64).")
        if self.region_size != 8:
            raise ValueError("The manuscript requires 8 x 8 reconstruction regions.")
        if min(self.classes, self.semantic_slots, self.state_dim) < 1:
            raise ValueError("Class, semantic-slot and state dimensions must be positive.")
        if self.router_channels < 1 or self.router_channels % 24:
            raise ValueError("Router channels must be a positive multiple of 24.")

    def to_dict(self):
        return asdict(self)


def haar_decompose(x):
    """Orthonormal LL, horizontal, vertical and diagonal subbands (Eq. 7)."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2


def haar_reconstruct(ll, horizontal, vertical, diagonal):
    values = torch.stack((ll + horizontal + vertical + diagonal,
                          ll - horizontal + vertical - diagonal,
                          ll + horizontal - vertical - diagonal,
                          ll - horizontal - vertical + diagonal), 2) / 2
    batch, channels, _, height, width = values.shape
    return F.pixel_shuffle(values.reshape(batch, 4 * channels, height, width), 2)


def three_bands(x):
    """Exactly reconstructable L/C/H decomposition (Eq. 15)."""
    low = F.avg_pool2d(x, 7, 1, 3)
    medium = F.avg_pool2d(x, 3, 1, 1)
    return low, medium - low, x - medium


class SourceQuery(nn.Module):
    """Original/inverted source query and channel recalibration (Eq. 5)."""

    def __init__(self, channels):
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1, padding_mode="reflect"),
                                       nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"))
        self.recalibration = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                          nn.Conv2d(channels, max(4, channels // 4), 1), nn.ReLU(),
                                          nn.Conv2d(max(4, channels // 4), channels, 1), nn.Sigmoid())

    def forward(self, source, size):
        feature = self.embedding(F.interpolate(source, size, mode="bilinear", align_corners=False))
        return feature * self.recalibration(feature)


class SourceCrossAttention(nn.Module):
    """Normalized transposed channel attention with positive temperature (Eq. 6)."""

    def __init__(self, channels):
        super().__init__()
        self.heads = 4
        self.raw_temperature = nn.Parameter(torch.full((self.heads, 1, 1), math.log(math.expm1(1.0))))
        self.key_value = nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False),
                                       nn.Conv2d(channels * 2, channels * 2, 3, padding=1,
                                                 groups=channels * 2, bias=False))
        self.projection = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, query, opposite_feature):
        batch, channels, height, width = opposite_feature.shape
        key, value = self.key_value(opposite_feature).chunk(2, 1)
        def split(x):
            return x.reshape(batch, self.heads, channels // self.heads, height * width)
        query, key, value = split(query), split(key), split(value)
        temperature = F.softplus(self.raw_temperature) + 1e-6
        attention = (F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-1, -2) / temperature).softmax(-1)
        return self.projection((attention @ value).reshape(batch, channels, height, width))


class ShallowSourceInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.queries = nn.ModuleList(SourceQuery(channels) for _ in range(4))
        self.attention = nn.ModuleList(SourceCrossAttention(channels) for _ in range(4))
        self.compress = Restormer_CNN_block(4 * channels, channels)

    def forward(self, infrared, visible, ir_feature, vi_feature):
        sources = (infrared, 1 - infrared, visible, 1 - visible)
        targets = (vi_feature, vi_feature, ir_feature, ir_feature)
        paths = [attention(query(source, target.shape[-2:]), target)
                 for query, attention, source, target in zip(self.queries, self.attention, sources, targets)]
        return self.compress(torch.cat(paths, 1))


class AttentiveSelectiveScan(nn.Module):
    """Selective SSM with semantic prompts added to the C term.

    Uses the Mamba CUDA kernel when available; the differentiable PyTorch
    recurrence implements the same discretization for CPU functional checks.
    """

    def __init__(self, channels, state_dim, backend="auto"):
        super().__init__()
        if backend not in ("auto", "torch", "mamba"):
            raise ValueError("Unknown scan backend")
        self.backend = backend
        self.rank = math.ceil(channels / 16)
        self.state_dim = state_dim
        self.project = nn.Linear(channels, self.rank + state_dim * 2, bias=False)
        self.delta = nn.Linear(self.rank, channels)
        self.A_log = nn.Parameter(torch.arange(1, state_dim + 1, dtype=torch.float).log().repeat(channels, 1))
        self.skip = nn.Parameter(torch.ones(channels))
        nn.init.uniform_(self.delta.weight, -self.rank ** -0.5, self.rank ** -0.5)
        dt = torch.exp(torch.empty(channels).uniform_(math.log(0.001), math.log(0.1)))
        with torch.no_grad():
            self.delta.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, tokens, prompt):
        raw_dt, b, c = torch.split(self.project(tokens), (self.rank, self.state_dim, self.state_dim), -1)
        raw_dt = F.linear(raw_dt, self.delta.weight)
        c = c + prompt
        kernel = None
        if tokens.is_cuda and self.backend != "torch":
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                kernel = selective_scan_fn
            except ImportError:
                if self.backend == "mamba":
                    raise RuntimeError("Install mamba-ssm for the requested CUDA scan backend.") from None
        if self.backend == "mamba" and not tokens.is_cuda:
            raise RuntimeError("The mamba scan backend requires a CUDA device.")
        if kernel is not None:
            result = kernel(tokens.transpose(1, 2).float().contiguous(), raw_dt.transpose(1, 2).float().contiguous(),
                            -self.A_log.float().exp(), b.transpose(1, 2).float().contiguous(),
                            c.transpose(1, 2).float().contiguous(), self.skip.float(),
                            delta_bias=self.delta.bias.float(), delta_softplus=True)
            return result.transpose(1, 2).to(tokens.dtype)
        # h_t = exp(dt_t A) h_(t-1) + dt_t B_t x_t; y_t = C_t h_t + D x_t.
        dt = F.softplus(raw_dt.float() + self.delta.bias.float())
        x, b, c = tokens.float(), b.float(), c.float()
        state = x.new_zeros(x.shape[0], x.shape[2], self.state_dim)
        outputs = []
        a = -self.A_log.float().exp()
        for index in range(x.shape[1]):
            step = dt[:, index, :, None]
            state = (step * a).exp() * state + step * b[:, index, None, :] * x[:, index, :, None]
            outputs.append((state * c[:, index, None, :]).sum(-1) + self.skip.float() * x[:, index])
        return torch.stack(outputs, 1).to(tokens.dtype)


class SemanticStateInteraction(nn.Module):
    """Interleave first, route each token with a shared MLP, sort and scan (Eq. 8)."""

    def __init__(self, channels, slots, state_dim, backend):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, slots))
        self.dictionary = nn.Parameter(torch.randn(slots, state_dim) * 0.02)
        self.scan = AttentiveSelectiveScan(channels, state_dim, backend)
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, channels)
        self.common_gate = nn.Conv2d(2 * channels, channels, 1)
        self.fuse = Restormer_CNN_block(3 * channels, channels)

    def forward(self, infrared, visible):
        batch, channels, height, width = infrared.shape
        tokens = torch.stack((infrared.flatten(2).transpose(1, 2),
                              visible.flatten(2).transpose(1, 2)), 2).flatten(1, 2)
        probabilities = self.router(tokens).softmax(-1)
        order = probabilities.detach().argmax(-1).argsort(dim=-1, stable=True)
        def gather(x, index):
            return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))
        scanned = self.scan(gather(tokens, order), gather(probabilities @ self.dictionary, order))
        recovered = gather(self.output(self.norm(scanned)), order.argsort(-1)) + tokens
        ir, vi = recovered.reshape(batch, height * width, 2, channels).unbind(2)
        ir = ir.transpose(1, 2).reshape_as(infrared)
        vi = vi.transpose(1, 2).reshape_as(visible)
        gate = self.common_gate(torch.cat((ir, vi), 1)).sigmoid()
        common = gate * ir + (1 - gate) * vi
        private_ir, private_vi = ir - common, vi - common
        return self.fuse(torch.cat((common, private_ir, private_vi), 1))


class SourceFrequencyCollaboration(nn.Module):
    def __init__(self, channels, config, backend):
        super().__init__()
        self.prompt_ir, self.prompt_vi = SourceQuery(channels), SourceQuery(channels)
        self.amplitude_ir = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.amplitude_vi = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.low = SemanticStateInteraction(channels, config.semantic_slots, config.state_dim, backend)
        hidden = 2 * channels
        self.high_router = nn.Sequential(nn.Conv2d(11 * channels, hidden, 3, padding=1, padding_mode="reflect"),
                                         nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                                         nn.GELU(), nn.Conv2d(hidden, 6 * channels, 1))
        self.refine = Restormer_CNN_block(channels, channels)

    def forward(self, infrared, visible, ir_feature, vi_feature):
        pi = self.prompt_ir(infrared, ir_feature.shape[-2:])
        pv = self.prompt_vi(visible, vi_feature.shape[-2:])
        li, *hi = haar_decompose(ir_feature + self.amplitude_ir * pi)
        lv, *hv = haar_decompose(vi_feature + self.amplitude_vi * pv)
        size = li.shape[-2:]
        evidence = torch.cat((*hi, *hv, *((a - b).abs() for a, b in zip(hi, hv)),
                              F.interpolate(pi, size, mode="bilinear", align_corners=False),
                              F.interpolate(pv, size, mode="bilinear", align_corners=False)), 1)
        count = ir_feature.shape[1]
        logits = self.high_router(evidence)
        batch, _, height, width = logits.shape
        weights = logits.reshape(batch, 2, 3, count, height, width).softmax(1)
        high = weights[:, 0] * torch.stack(hi, 1) + weights[:, 1] * torch.stack(hv, 1)
        return self.refine(haar_reconstruct(self.low(li, lv), *high.unbind(1)))


class GuidedSkip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1, padding_mode="reflect"),
                                     nn.GELU(), nn.Conv2d(channels, 1, 1))
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2 * channels, channels, 1),
                                     nn.GELU(), nn.Conv2d(channels, channels, 1))
        for branch in (self.spatial, self.channel):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, decoding, skip):
        evidence = torch.cat((decoding, skip), 1)
        return skip * (2 * self.spatial(evidence).sigmoid()) * (2 * self.channel(evidence).sigmoid())


class RegionAdaptiveDetailEnhancement(nn.Module):
    """Source selection, consistency/confidence and bounded regional gains (Eq. 14)."""

    def __init__(self, region_size=8):
        super().__init__()
        self.region_size = region_size
        self.router = nn.Sequential(nn.Conv2d(10, 32, 3, padding=1, padding_mode="reflect"), nn.GELU(),
                                    nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(), nn.Conv2d(32, 3, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, baseline, infrared, visible):
        di, dv, db = (x - F.avg_pool2d(x, 3, 1, 1) for x in (infrared, visible, baseline))
        choose_ir = di.abs() >= dv.abs()
        selected_detail = torch.where(choose_ir, di, dv)
        selected_source = torch.where(choose_ir, infrared, visible)
        agreement = (1 - (di - dv).abs() / (di.abs() + dv.abs() + 1 / 255)).clamp(0, 1)
        confidence = (di.abs() - dv.abs()).abs()
        confidence = confidence / confidence.mean((-2, -1), keepdim=True).clamp_min(1e-4)
        evidence = torch.cat((infrared, visible, baseline, di, dv, db, agreement, confidence, di.abs(), dv.abs()), 1)
        control = self.router(F.avg_pool2d(evidence, self.region_size, self.region_size, ceil_mode=True))
        source, fused, intensity = F.interpolate(control, baseline.shape[-2:], mode="bilinear", align_corners=False).tanh().chunk(3, 1)
        return (baseline + (0.05 + 0.15 * agreement + 0.05 * source) * selected_detail
                + (0.1 + 0.1 * fused) * db + 0.01 * intensity * (selected_source - baseline)).clamp(0, 1)


class SpatialSemanticFrequencyRouting(nn.Module):
    """Twelve-channel evidence, three source experts per band (Eqs. 15--18)."""

    def __init__(self, config):
        super().__init__()
        channels = config.router_channels
        self.region_size = config.region_size
        self.stem = nn.Sequential(nn.Conv2d(12, channels, 3, padding=1, padding_mode="reflect"),
                                  nn.GroupNorm(6, channels), nn.GELU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.GELU())
        self.context = nn.Sequential(nn.Conv2d(channels, 2 * channels, 3, 2, 1, padding_mode="reflect"),
                                     nn.GroupNorm(8, 2 * channels), nn.GELU(),
                                     nn.Conv2d(2 * channels, channels, 3, padding=2, dilation=2), nn.GELU())
        self.semantic = nn.Conv2d(channels, config.classes, 1)
        self.foreground = nn.Conv2d(channels, 1, 1)
        self.router = nn.Sequential(nn.Conv2d(2 * channels + 2, channels, 3, padding=1, padding_mode="reflect"),
                                    nn.GELU(), nn.Conv2d(channels, 9, 1))
        self.refinement = nn.Sequential(nn.Conv2d(channels + 4, channels, 3, padding=1, padding_mode="reflect"),
                                        nn.GELU(), nn.Conv2d(channels, 1, 1))
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(self, baseline, infrared, visible):
        bands = [three_bands(x) for x in (infrared, visible, baseline)]
        low = [x[0] for x in bands]
        residual = [x[1] + x[2] for x in bands]
        stem = self.stem(torch.cat((infrared, visible, baseline, (infrared - visible).abs(),
                                    *low, *residual, (residual[0] - residual[1]).abs(), (low[0] - low[1]).abs()), 1))
        context = F.interpolate(self.context(stem), baseline.shape[-2:], mode="bilinear", align_corners=False)
        semantic_logits = self.semantic(context)
        foreground = self.foreground(context).sigmoid()
        evidence = torch.cat((stem, context, foreground, (residual[0] - residual[1]).abs()), 1)
        logits = self.router(F.avg_pool2d(evidence, self.region_size, self.region_size, ceil_mode=True))
        logits = F.interpolate(logits, baseline.shape[-2:], mode="bilinear", align_corners=False)
        wl, wc, wh = (x.softmax(1) for x in logits.chunk(3, 1))
        bias = torch.cat((0.35 * foreground, 0.35 * (1 - foreground), torch.zeros_like(foreground)), 1)
        wl = ((wl + 1e-6).log() + bias).softmax(1)
        routed = sum((weight * torch.cat([source[index] for source in bands], 1)).sum(1, keepdim=True)
                     for index, weight in enumerate((wl, wc, wh)))
        correction = 0.1 * self.refinement(torch.cat((stem, routed, baseline, infrared, visible), 1)).tanh()
        return {"fused": (routed + correction).clamp(0, 1), "semantic_logits": semantic_logits,
                "foreground": foreground, "low_weights": wl, "coarse_weights": wc, "fine_weights": wh,
                "routed": routed, "correction": correction}


class SCPRNet(nn.Module):
    """Single, end-to-end network matching Fig. 2 and Section 3 of the manuscript."""

    def __init__(self, config=None, scan_backend="auto"):
        super().__init__()
        self.config = config or SCPRConfig()
        channels = self.config.channels
        def encoder():
            return nn.ModuleList(Restormer_CNN_block(1 if i == 0 else channels[i - 1], c)
                                 for i, c in enumerate(channels))
        def downsample():
            return nn.ModuleList(nn.Conv2d(c, c, 3, 2, 1, bias=False, padding_mode="reflect") for c in channels[:-1])
        self.encoder_ir, self.encoder_vi = encoder(), encoder()
        self.down_ir, self.down_vi = downsample(), downsample()
        self.base_fusion = nn.ModuleList(Restormer_CNN_block(c * 2, c) for c in channels)
        self.collaboration = nn.ModuleList([ShallowSourceInteraction(c) for c in channels[:2]] +
                                           [SourceFrequencyCollaboration(c, self.config, scan_backend) for c in channels[2:]])
        self.injection = nn.ParameterList(nn.Parameter(torch.full((1, c, 1, 1), math.log(0.01 / 0.99))) for c in channels)
        self.deep_decoder = Restormer_CNN_block(channels[-1], channels[-1])
        self.upsample = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(channels[i + 1], channels[i], 4, 2, 1), nn.ReLU())
                                     for i in range(3))
        self.skip = nn.ModuleList(GuidedSkip(c) for c in channels[:-1])
        self.decoder = nn.ModuleList(Restormer_CNN_block(2 * c, c) for c in channels[:-1])
        self.output = nn.Sequential(nn.Conv2d(channels[0], 1, 3, padding=1, padding_mode="reflect"), nn.Sigmoid())
        self.detail_enhancement = RegionAdaptiveDetailEnhancement(self.config.region_size)
        self.frequency_routing = SpatialSemanticFrequencyRouting(self.config)

    def forward(self, infrared, visible, return_aux=False):
        if infrared.shape != visible.shape or infrared.ndim != 4 or infrared.shape[1] != 1:
            raise ValueError("Expected aligned infrared/visible tensors of shape B x 1 x H x W.")
        if min(infrared.shape[-2:]) < 32 or any(x % 16 for x in infrared.shape[-2:]):
            raise ValueError("Pad image dimensions to multiples of 16, with both dimensions at least 32.")
        ir, vi, fused_scales = infrared, visible, []
        for index in range(4):
            if index:
                ir, vi = self.down_ir[index - 1](ir), self.down_vi[index - 1](vi)
            ir, vi = self.encoder_ir[index](ir), self.encoder_vi[index](vi)
            base = self.base_fusion[index](torch.cat((ir, vi), 1))
            candidate = self.collaboration[index](infrared, visible, ir, vi)
            gate = self.injection[index].sigmoid()
            fused_scales.append((1 - gate) * base + gate * candidate)
        decoded = self.deep_decoder(fused_scales[-1])
        for index in (2, 1, 0):
            decoded = self.upsample[index](decoded)
            decoded = self.decoder[index](torch.cat((decoded, self.skip[index](decoded, fused_scales[index])), 1))
        baseline = self.output(decoded)
        enhanced = self.detail_enhancement(baseline, infrared, visible)
        result = self.frequency_routing(enhanced, infrared, visible)
        result.update(baseline=baseline, enhanced=enhanced)
        return result if return_aux else result["fused"]
