#!/usr/bin/env python3
"""End-to-end, baseline-constrained training for Pareto ASSM V8."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nets.Ufuser import Ufuser  # noqa: E402
from nets.Ufuser_pareto_assm_v8 import UfuserParetoASSMV8  # noqa: E402
from nets.fusion_losses_v8 import (  # noqa: E402
    VISUAL_NAMES, decomposition_gate_loss, normalized_pareto_guard,
    semantic_and_route_loss, source_correlation_guard, statistic_gain_guard,
    visual_terms,
)
from repro.evaluate_metrics import (  # noqa: E402
    average_gradient, entropy, scd, spatial_frequency, standard_deviation, vif,
)
from repro.msrs_v2_data import MSRSAlignedV2, aligned_names  # noqa: E402

METRICS = ("EN", "SD", "SF", "AG", "SCD", "VIF")


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "datasets/MSRS-main")
    p.add_argument("--split-checkpoint", type=Path,
                   default=ROOT / "model/MSRS_LRASPP_teacher_v2_best.ckpt")
    p.add_argument("--init-emma", type=Path, default=ROOT / "model/EMMA_trained.pth")
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "model/EMMA_PARETO_ASSM_V8.ckpt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples-per-epoch", type=int, default=12000)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr-new", type=float, default=1e-4)
    p.add_argument("--lr-decoder", type=float, default=2e-6)
    p.add_argument("--lr-encoder", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--lambda-intensity", type=float, default=0.4)
    p.add_argument("--lambda-gradient", type=float, default=1.2)
    p.add_argument("--lambda-laplacian", type=float, default=0.5)
    p.add_argument("--lambda-contrast", type=float, default=0.5)
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--lambda-wavelet", type=float, default=1.0)
    p.add_argument("--lambda-pareto", type=float, default=1.0)
    p.add_argument("--lambda-statistics", type=float, default=2.0)
    p.add_argument("--lambda-correlation", type=float, default=0.5)
    p.add_argument("--lambda-anchor", type=float, default=0.1)
    p.add_argument("--lambda-delta", type=float, default=1e-3)
    p.add_argument("--lambda-semantic", type=float, default=5e-3)
    p.add_argument("--lambda-route", type=float, default=1e-2)
    p.add_argument("--lambda-decomposition", type=float, default=5e-3)
    p.add_argument("--semantic-start-epoch", type=int, default=3)
    p.add_argument("--std-gain", type=float, default=1.03)
    p.add_argument("--detail-gain", type=float, default=1.02)
    p.add_argument("--pareto-margin", type=float, default=0.0)
    p.add_argument("--metric-val-images", type=int, default=48)
    p.add_argument("--pareto-min-ratio", type=float, default=0.995)
    p.add_argument("--pareto-min-wins", type=int, default=4)
    p.add_argument("--regression-floor", type=float, default=0.985)
    p.add_argument("--regression-patience", type=int, default=3)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--num-tokens", type=int, default=9)
    p.add_argument("--num-classes", type=int, default=9)
    p.add_argument("--max-feature-scale", type=float, default=0.25)
    p.add_argument("--max-detail-scale", type=float, default=0.20)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def raw_state(checkpoint):
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def pad8(x):
    height, width = x.shape[-2:]
    return F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode="reflect"), height, width


def schedule_factor(step, warmup, total):
    if step < warmup:
        return max((step + 1) / max(1, warmup), 1e-3)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def exact_metrics(fused, infrared, visible):
    def image(x):
        return np.rint(x.detach().cpu().numpy()[0, 0] * 255).clip(0, 255).astype(np.float64)
    fu, ir, vi = image(fused), image(infrared), image(visible)
    return {"EN": entropy(fu), "SD": standard_deviation(fu),
            "SF": spatial_frequency(fu), "AG": average_gradient(fu),
            "SCD": scd(fu, ir, vi), "VIF": vif(fu, ir, vi)}


@torch.inference_mode()
def validate(model, baseline, loader, device):
    model.eval(); baseline.eval()
    totals = {name: 0.0 for name in METRICS}
    totals.update({f"base_{name}": 0.0 for name in METRICS})
    count = 0
    for infrared, visible, _, _ in loader:
        infrared = infrared.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)
        irp, height, width = pad8(infrared); vip, _, _ = pad8(visible)
        fused = model(irp, vip)[..., :height, :width]
        base = baseline(irp, vip)[..., :height, :width]
        for name, value in exact_metrics(fused, infrared, visible).items(): totals[name] += value
        for name, value in exact_metrics(base, infrared, visible).items(): totals[f"base_{name}"] += value
        count += 1
    totals = {key: value / count for key, value in totals.items()}
    ratios = {name: totals[name] / max(totals[f"base_{name}"], 1e-8) for name in METRICS}
    totals["quality"] = math.exp(sum(math.log(max(0.5, min(2.0, x)))
                                      for x in ratios.values()) / len(ratios))
    totals["wins"] = sum(x > 1.0 for x in ratios.values())
    totals["min_ratio"] = min(ratios.values())
    totals.update({f"ratio_{name}": value for name, value in ratios.items()})
    totals["images"] = count
    return totals


def optimizer_groups(model, opt):
    decoder_prefixes = ("emma.de_", "emma.up", "emma.last")
    new, decoder, encoder = [], [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        if not name.startswith("emma."): new.append(parameter)
        elif name.startswith(decoder_prefixes): decoder.append(parameter)
        else: encoder.append(parameter)
    return [{"params": new, "lr": opt.lr_new, "name": "new"},
            {"params": decoder, "lr": opt.lr_decoder, "name": "decoder"},
            {"params": encoder, "lr": opt.lr_encoder, "name": "encoder"}]


def make_emma_reference(model):
    return {name: parameter.detach().clone()
            for name, parameter in model.emma.named_parameters()}


def emma_anchor(model, reference):
    values = []
    for name, parameter in model.emma.named_parameters():
        scale = reference[name].square().mean().clamp_min(1e-6)
        values.append((parameter - reference[name]).square().mean() / scale)
    return torch.stack(values).mean()


def weighted_visual(terms, opt):
    return sum(float(getattr(opt, f"lambda_{name}")) * terms[name]
               for name in VISUAL_NAMES)


def save(path, epoch, model, optimizer, scheduler, metrics, opt):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "metrics": metrics, "config": vars(opt)}, path)


def main():
    opt = arguments()
    if opt.crop_size % 8 or opt.crop_size <= 0:
        raise ValueError("crop-size must be a positive multiple of 8")
    if opt.num_tokens != opt.num_classes:
        raise ValueError("num-tokens must equal num-classes for supervised routing")
    for path in (opt.data_root, opt.split_checkpoint, opt.init_emma):
        if not path.exists(): raise FileNotFoundError(path)
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(opt.device)

    split = torch.load(opt.split_checkpoint, map_location="cpu")
    all_val = list(split["val_names"]); val_set = set(all_val)
    train_names = [name for name in aligned_names(opt.data_root, "train") if name not in val_set]
    if 0 < opt.metric_val_images < len(all_val):
        indices = np.linspace(0, len(all_val) - 1, opt.metric_val_images).round().astype(int)
        val_names = [all_val[index] for index in dict.fromkeys(indices)]
    else: val_names = all_val
    train_data = MSRSAlignedV2(opt.data_root, train_names, crop_size=opt.crop_size,
                               augment=True, samples_per_epoch=opt.samples_per_epoch)
    val_data = MSRSAlignedV2(opt.data_root, val_names, crop_size=None, augment=False)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.workers, pin_memory=device.type == "cuda",
                              drop_last=True, persistent_workers=opt.workers > 0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=max(1, opt.workers // 2),
                            pin_memory=device.type == "cuda")

    emma_state = raw_state(torch.load(opt.init_emma, map_location=device))
    model = UfuserParetoASSMV8(opt.d_state, opt.num_tokens, opt.num_classes,
                               opt.max_feature_scale, opt.max_detail_scale).to(device)
    model.load_emma(emma_state)
    baseline = Ufuser().to(device); baseline.load_state_dict(emma_state, strict=True)
    baseline.eval()
    for parameter in baseline.parameters(): parameter.requires_grad_(False)
    class_weights = split["class_weights"].to(device)

    model.eval()
    with torch.inference_mode():
        ir = torch.rand(1, 1, 32, 32, device=device); vi = torch.rand_like(ir)
        difference = (model(ir, vi) - baseline(ir, vi)).abs().max()
    print(f"Zero-init max |model-EMMA|={float(difference):.3e}")
    if float(difference) > 1e-6: raise RuntimeError("V8 initialization does not preserve EMMA")

    reference = make_emma_reference(model)
    groups = optimizer_groups(model, opt)
    optimizer = torch.optim.AdamW(groups, weight_decay=opt.weight_decay)
    total_steps = opt.epochs * len(train_loader); warmup = opt.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lambda step: schedule_factor(step, warmup, total_steps)] * len(groups))
    start_epoch = 1
    if opt.resume:
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    scaler = GradScaler(enabled=opt.amp and device.type == "cuda", init_scale=1024.0, growth_interval=2000)
    candidate_path = opt.output.with_name(f"{opt.output.stem}_best_candidate{opt.output.suffix}")
    pareto_path = opt.output.with_name(f"{opt.output.stem}_best_pareto{opt.output.suffix}")
    strict_path = opt.output.with_name(f"{opt.output.stem}_best_strict{opt.output.suffix}")
    history_path = opt.output.with_suffix(".csv"); history_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if opt.resume and history_path.exists() else "w"
    best_candidate = -float("inf"); best_pareto = 1.0; best_strict = 1.0
    regression_count = 0
    print(f"Model parameters={sum(p.numel() for p in model.parameters()):,}; all EMMA parameters "
          f"trainable; train/metric-val={len(train_names)}/{len(val_names)}; batches={len(train_loader)}")

    keys = ("total", *VISUAL_NAMES, "pareto", "statistics", "correlation",
            "anchor", "semantic", "route", "decomposition", "std_ratio",
            "local_ratio", "gradient_ratio")
    columns = ["epoch", *keys, "quality", "wins", "min_ratio", *METRICS,
               *(f"{name}_r" for name in METRICS), "scale3", "scale4", "detail_scale",
               "lr_new", "lr_decoder", "lr_encoder"]
    with history_path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if mode == "w": writer.writerow(columns)
        for epoch in range(start_epoch, opt.epochs + 1):
            model.train(); baseline.eval(); sums = {key: 0.0 for key in keys}
            semantic_factor = 0.0 if epoch < opt.semantic_start_epoch else min(
                1.0, (epoch - opt.semantic_start_epoch + 1) / 2.0)
            for batch_index, (infrared, visible, labels, _) in enumerate(train_loader, 1):
                infrared = infrared.to(device, non_blocking=True)
                visible = visible.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler.is_enabled()):
                    with torch.no_grad(): baseline_output = baseline(infrared, visible)
                    outputs = model(infrared, visible, True)
                    terms = visual_terms(outputs["fused"], infrared, visible, baseline_output)
                    with torch.no_grad():
                        base_terms = visual_terms(baseline_output, infrared, visible, baseline_output)
                    pareto, _ = normalized_pareto_guard(terms, base_terms, opt.pareto_margin)
                    statistics, stat = statistic_gain_guard(
                        outputs["fused"], baseline_output, opt.std_gain, opt.detail_gain)
                    correlation = source_correlation_guard(
                        outputs["fused"], baseline_output, infrared, visible)
                    semantic, route = semantic_and_route_loss(outputs, labels, class_weights)
                    decomposition = decomposition_gate_loss(outputs, infrared, visible)
                    anchor = emma_anchor(model, reference)
                    delta = outputs["logit_delta"].abs().mean()
                    loss = (weighted_visual(terms, opt) + opt.lambda_pareto * pareto
                            + opt.lambda_statistics * statistics
                            + opt.lambda_correlation * correlation
                            + opt.lambda_anchor * anchor + opt.lambda_delta * delta
                            + semantic_factor * (opt.lambda_semantic * semantic
                                                 + opt.lambda_route * route)
                            + opt.lambda_decomposition * decomposition)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                values = {"total": loss, **{name: terms[name] for name in VISUAL_NAMES},
                          "pareto": pareto, "statistics": statistics,
                          "correlation": correlation, "anchor": anchor,
                          "semantic": semantic, "route": route,
                          "decomposition": decomposition,
                          "std_ratio": stat["std_ratio"], "local_ratio": stat["local_ratio"],
                          "gradient_ratio": stat["gradient_ratio"]}
                for key in keys: sums[key] += float(values[key].detach())
                if opt.log_interval and batch_index % opt.log_interval == 0:
                    print(f"epoch {epoch}/{opt.epochs} batch {batch_index}/{len(train_loader)} "
                          f"loss={float(loss):.5f} guard={float(pareto):.4f} "
                          f"std={float(stat['std_ratio']):.4f} route={float(route):.3f}", flush=True)
            means = {key: value / len(train_loader) for key, value in sums.items()}
            metrics = validate(model, baseline, val_loader, device)
            scale3 = float((model.max_feature_scale * torch.tanh(model.feature_scale3_logit)).abs().mean())
            scale4 = float((model.max_feature_scale * torch.tanh(model.feature_scale4_logit)).abs().mean())
            detail_scale = float(model.max_detail_scale * torch.tanh(model.detail_scale_logit))
            row = [epoch, *(means[key] for key in keys), metrics["quality"], metrics["wins"],
                   metrics["min_ratio"], *(metrics[name] for name in METRICS),
                   *(metrics[f"ratio_{name}"] for name in METRICS), scale3, scale4,
                   detail_scale, *(group["lr"] for group in optimizer.param_groups)]
            writer.writerow(row); stream.flush()
            ratios = "/".join(f"{metrics[f'ratio_{name}']:.4f}" for name in METRICS)
            print(f"epoch {epoch}/{opt.epochs} total={means['total']:.5f} "
                  f"quality={metrics['quality']:.5f} wins={metrics['wins']}/6 "
                  f"min={metrics['min_ratio']:.4f} ratios={ratios}", flush=True)
            save(opt.output, epoch, model, optimizer, scheduler, metrics, opt)
            if metrics["quality"] > best_candidate:
                best_candidate = metrics["quality"]
                save(candidate_path, epoch, model, optimizer, scheduler, metrics, opt)
            qualifies = (metrics["wins"] >= opt.pareto_min_wins
                         and metrics["min_ratio"] >= opt.pareto_min_ratio
                         and metrics["quality"] > 1.0)
            if qualifies and metrics["quality"] > best_pareto:
                best_pareto = metrics["quality"]
                save(pareto_path, epoch, model, optimizer, scheduler, metrics, opt)
                print(f"Saved Pareto-qualified model: {pareto_path}")
            strict = metrics["min_ratio"] >= 1.0 and metrics["quality"] > best_strict
            if strict:
                best_strict = metrics["quality"]
                save(strict_path, epoch, model, optimizer, scheduler, metrics, opt)
                print(f"Saved all-six-wins model: {strict_path}")
            if opt.save_every > 0 and epoch % opt.save_every == 0:
                save(opt.output.parent / f"{opt.output.stem}_checkpoints/epoch_{epoch:03d}.ckpt",
                     epoch, model, optimizer, scheduler, metrics, opt)
            regression_count = regression_count + 1 if metrics["quality"] < opt.regression_floor else 0
            if opt.regression_patience > 0 and regression_count >= opt.regression_patience:
                print("Stopped early: validation remained below the EMMA regression floor.")
                break
    print(f"Finished model latest={opt.output}")
    if pareto_path.exists(): print(f"Use Pareto checkpoint: {pareto_path}")
    else: print("No checkpoint passed the Pareto gate; do not claim improvement.")


if __name__ == "__main__":
    main()
