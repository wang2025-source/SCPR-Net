#!/usr/bin/env python3
"""Test the manuscript SCPR-Net from a single self-contained checkpoint."""

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

from repro.scpr_checkpoint import load_checkpoint, restore_model  # noqa: E402

EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable SCPR-Net IR/visible fusion")
    parser.add_argument("--model", type=Path, default=REPOSITORY_ROOT / "outputs/checkpoints/SCPR-Net_best.ckpt")
    parser.add_argument("--ir-dir", type=Path, default=CODE_ROOT / "test_img/ir")
    parser.add_argument("--vi-dir", type=Path, default=CODE_ROOT / "test_img/vi")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs/scpr-net")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan-backend", choices=("auto", "torch", "mamba"), default="auto")
    return parser.parse_args()


def build_model(options: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    return restore_model(load_checkpoint(options.model), device, options.scan_backend).eval()


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
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model = build_model(options, device)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    ir_paths = sorted(path for path in options.ir_dir.iterdir() if path.suffix.lower() in EXTENSIONS)
    if not ir_paths:
        raise RuntimeError(f"No supported images under {options.ir_dir}")
    stems = [path.stem for path in ir_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate input stems would overwrite output PNG files.")
    if options.output_dir.resolve() in (options.ir_dir.resolve(), options.vi_dir.resolve()):
        raise ValueError("Use a separate output directory to preserve source images.")

    with torch.inference_mode():
        for ir_path in tqdm(ir_paths, desc="SCPR-Net inference"):
            vi_path = options.vi_dir / ir_path.name
            if not vi_path.is_file():
                raise FileNotFoundError(f"Visible pair is missing: {vi_path}")
            infrared, visible = read_pair(ir_path, vi_path)
            height, width = infrared.shape
            ir = torch.from_numpy(infrared).float()[None, None].to(device) / 255.0
            vi = torch.from_numpy(visible).float()[None, None].to(device) / 255.0
            pad_h = max(32, (height + 15) // 16 * 16) - height
            pad_w = max(32, (width + 15) // 16 * 16) - width
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
