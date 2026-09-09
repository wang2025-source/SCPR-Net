#!/usr/bin/env python3
"""Portable SCPR inference without modifying the preserved source snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPOSITORY_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from nets.Ufuser_region_moe_v19 import UfuserRegionMoEV19  # noqa: E402
from nets.Ufuser_v10_region_detail_v12 import UfuserV10RegionDetailV12  # noqa: E402

EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable SCPR-Net IR/visible fusion")
    parser.add_argument("--model", type=Path, required=True, help="V19 or V22 checkpoint")
    parser.add_argument("--v12-model", type=Path, required=True)
    parser.add_argument("--v10-model", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--vi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(options: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    v10_checkpoint = load_checkpoint(options.v10_model)
    v12_checkpoint = load_checkpoint(options.v12_model)
    scpr_checkpoint = load_checkpoint(options.model)

    v10_config = v10_checkpoint.get("config", {})
    v12_config = v12_checkpoint.get("config", {})
    v12 = UfuserV10RegionDetailV12(
        int(v10_config.get("d_state", 16)),
        int(v10_config.get("num_tokens", 9)),
        int(v10_config.get("num_classes", 9)),
        float(v10_config.get("initial_new", 0.01)),
        int(v12_config.get("region_size", 16)),
        float(v12_config.get("source_gain", 0.05)),
        float(v12_config.get("agreement_gain", 0.15)),
        float(v12_config.get("fused_gain", 0.10)),
        float(v12_config.get("max_source_delta", 0.05)),
        float(v12_config.get("max_fused_gain", 0.10)),
        float(v12_config.get("max_intensity_gain", 0.01)),
    )
    v12.load_state_dict(v12_checkpoint["model"], strict=True)

    scpr_config = scpr_checkpoint.get("config", {})
    model = UfuserRegionMoEV19(
        v12,
        int(scpr_config.get("classes", 9)),
        int(scpr_config.get("channels", 48)),
        int(scpr_config.get("region", 8)),
        float(scpr_config.get("max_refine", 0.10)),
    ).to(device)
    model.load_state_dict(scpr_checkpoint["model"], strict=True)
    model.eval()
    return model


def read_pair(ir_path: Path, vi_path: Path) -> tuple[np.ndarray, np.ndarray]:
    infrared = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    visible_bgr = cv2.imread(str(vi_path), cv2.IMREAD_COLOR)
    if infrared is None:
        raise RuntimeError(f"Cannot read infrared image: {ir_path}")
    if visible_bgr is None:
        raise RuntimeError(f"Cannot read visible image: {vi_path}")
    visible = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    if infrared.shape != visible.shape:
        raise ValueError(f"Shape mismatch for {ir_path.name}: {infrared.shape} vs {visible.shape}")
    return infrared, visible


def main() -> int:
    options = arguments()
    if not options.ir_dir.is_dir() or not options.vi_dir.is_dir():
        raise FileNotFoundError("Both --ir-dir and --vi-dir must be directories")

    device = torch.device(options.device)
    model = build_model(options, device)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    ir_paths = sorted(path for path in options.ir_dir.iterdir() if path.suffix.lower() in EXTENSIONS)
    if not ir_paths:
        raise RuntimeError(f"No supported images under {options.ir_dir}")

    with torch.inference_mode():
        for ir_path in tqdm(ir_paths, desc="SCPR-Net inference"):
            vi_path = options.vi_dir / ir_path.name
            if not vi_path.is_file():
                raise FileNotFoundError(f"Visible pair is missing: {vi_path}")
            infrared, visible = read_pair(ir_path, vi_path)
            height, width = infrared.shape
            ir = torch.from_numpy(infrared).float()[None, None].to(device) / 255.0
            vi = torch.from_numpy(visible).float()[None, None].to(device) / 255.0
            pad_h, pad_w = (-height) % 16, (-width) % 16
            if pad_h or pad_w:
                mode = "reflect" if height > pad_h and width > pad_w else "replicate"
                ir = F.pad(ir, (0, pad_w, 0, pad_h), mode=mode)
                vi = F.pad(vi, (0, pad_w, 0, pad_h), mode=mode)
            fused = model(ir, vi)[0, 0, :height, :width]
            image = np.rint(fused.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            output_path = options.output_dir / f"{ir_path.stem}.png"
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Failed to write {output_path}")

    print(f"Done: {len(ir_paths)} image(s) -> {options.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
