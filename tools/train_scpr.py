#!/usr/bin/env python3
"""Train the manuscript SCPR-Net with its four-term end-to-end objective."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from nets.scpr_net import SCPRConfig, SCPRNet
from nets.scpr_loss import LossConfig, SCPRLoss
from repro.scpr_data import MSRSAlignedDataset, aligned_names, grouped_train_val_split
from repro.scpr_checkpoint import FORMAT, load_checkpoint, restore_model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Optional clean interruption for smoke checks; 0 completes the schedule")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="0 uses every training pair once")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--classes", type=int, default=9)
    parser.add_argument("--semantic-slots", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-backend", choices=("auto", "torch", "mamba"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    # Values within the grouped-LR range specified in the implementation section.
    for name, value in (("encoder", 5e-6), ("decoder", 1e-5), ("collaboration", 3e-5),
                        ("detail", 5e-5), ("routing", 2e-4)):
        parser.add_argument(f"--lr-{name}", type=float, default=value)
    defaults = LossConfig()
    for name in ("visual", "frequency", "semantic", "route", "omega_ir", "omega_vi", "omega_ssim",
                 "beta_foreground", "beta_edge"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name),
                            help="Explicit coefficient; the manuscript does not specify its numeric value")
    options = parser.parse_args()
    if options.crop_size < 32 or options.crop_size % 16:
        parser.error("--crop-size must be a multiple of 16 and at least 32")
    if options.epochs < 1 or options.batch_size < 1 or options.workers < 0 or options.samples_per_epoch < 0:
        parser.error("Invalid epoch, batch, worker or sample count")
    return options


def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def parameter_groups(model, options):
    groups = {name: [] for name in ("encoder", "decoder", "collaboration", "detail", "routing")}
    for name, parameter in model.named_parameters():
        if name.startswith(("encoder_", "down_")):
            group = "encoder"
        elif name.startswith(("collaboration.", "injection.")):
            group = "collaboration"
        elif name.startswith("detail_enhancement."):
            group = "detail"
        elif name.startswith("frequency_routing."):
            group = "routing"
        else:
            group = "decoder"
        groups[group].append(parameter)
    return [{"params": values, "lr": getattr(options, "lr_" + name), "name": name}
            for name, values in groups.items()]


def schedule_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def pack_checkpoint(model, optimizer, scheduler, scaler, epoch, best, split, generator, options, loss_config):
    return {"format": FORMAT, "model_config": model.config.to_dict(), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_val_loss": best, "split": split, "loss_config": loss_config.to_dict(),
            "training_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()},
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_rng": generator.get_state()}


def main():
    options = arguments()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(options.device)
    checkpoint = load_checkpoint(options.resume) if options.resume else None
    if checkpoint:
        # Resume the same training protocol; only device/data/output paths may change.
        for name, value in checkpoint["training_config"].items():
            if name not in {"data_root", "output_dir", "resume", "device", "scan_backend", "workers", "stop_after_epoch"}:
                setattr(options, name, value)
        model = restore_model(checkpoint, device, options.scan_backend)
        loss_config = LossConfig(**checkpoint["loss_config"])
        split = checkpoint["split"]
        existing_names = set(aligned_names(options.data_root, "train"))
        if set(split["train_names"]) | set(split["val_names"]) != existing_names:
            raise ValueError("The resume dataset differs from the saved training/validation split.")
    else:
        model = SCPRNet(SCPRConfig(classes=options.classes, semantic_slots=options.semantic_slots,
                                   state_dim=options.state_dim), options.scan_backend).to(device)
        loss_config = LossConfig(**{name: getattr(options, name) for name in (
            "visual", "frequency", "semantic", "route", "omega_ir", "omega_vi", "omega_ssim", "beta_foreground", "beta_edge")})
        train_names, val_names = grouped_train_val_split(aligned_names(options.data_root, "train"),
                                                        options.val_fraction, options.seed)
        split = {"train_names": train_names, "val_names": val_names}
    objective = SCPRLoss(loss_config)
    generator = torch.Generator().manual_seed(options.seed)
    train_data = MSRSAlignedDataset(options.data_root, split["train_names"], crop_size=options.crop_size, augment=True,
                               samples_per_epoch=options.samples_per_epoch or None)
    val_data = MSRSAlignedDataset(options.data_root, split["val_names"], crop_size=options.crop_size, augment=False)
    train_loader = DataLoader(train_data, options.batch_size, shuffle=True, num_workers=options.workers,
                              generator=generator, worker_init_fn=seed_worker, drop_last=False)
    val_loader = DataLoader(val_data, options.batch_size, shuffle=False, num_workers=options.workers, worker_init_fn=seed_worker)
    optimizer = torch.optim.AdamW(parameter_groups(model, options), weight_decay=options.weight_decay)
    total = options.epochs * len(train_loader)
    warmup = options.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_factor(step, warmup, total))
    scaler = torch.amp.GradScaler("cuda", enabled=options.amp and device.type == "cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=options.amp and device.type == "cuda")
    start, best = 1, float("inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_val_loss"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        generator.set_state(checkpoint["loader_rng"])
    options.output_dir.mkdir(parents=True, exist_ok=True)
    history = options.output_dir / "training.csv"
    if history.exists() and not checkpoint:
        raise FileExistsError(f"Training output already exists: {history}; use --resume or a new --output-dir")
    names = ("total", "visual", "frequency", "semantic", "route")
    with history.open("a" if checkpoint else "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(("epoch", *("train_" + name for name in names), "val_loss"))
        for epoch in range(start, options.epochs + 1):
            model.train()
            sums = {name: 0.0 for name in names}
            seen = 0
            for ir, vi, labels, _ in train_loader:
                ir, vi, labels = ir.to(device), vi.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    terms = objective(model(ir, vi, return_aux=True), ir, vi, labels)
                if not torch.isfinite(terms["total"]):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(terms["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                seen += ir.shape[0]
                for name in names:
                    sums[name] += float(terms[name].detach()) * ir.shape[0]
            model.eval()
            val_sum, val_count = 0.0, 0
            with torch.inference_mode():
                for ir, vi, labels, _ in val_loader:
                    ir, vi, labels = ir.to(device), vi.to(device), labels.to(device)
                    val_sum += float(objective(model(ir, vi, return_aux=True), ir, vi, labels)["total"]) * ir.shape[0]
                    val_count += ir.shape[0]
            val_loss = val_sum / val_count
            if not math.isfinite(val_loss):
                raise FloatingPointError("Non-finite validation loss")
            improved = val_loss < best
            best = min(best, val_loss)
            state = pack_checkpoint(model, optimizer, scheduler, scaler, epoch, best, split, generator, options, loss_config)
            torch.save(state, options.output_dir / "SCPR-Net_last.ckpt")
            if improved:
                torch.save(state, options.output_dir / "SCPR-Net_best.ckpt")
            writer.writerow((epoch, *(sums[name] / seen for name in names), val_loss))
            stream.flush()
            print(f"Epoch {epoch}/{options.epochs}: train={sums['total'] / seen:.6f} val={val_loss:.6f}", flush=True)
            if options.stop_after_epoch and epoch >= options.stop_after_epoch:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
