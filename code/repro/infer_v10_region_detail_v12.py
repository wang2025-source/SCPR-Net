#!/usr/bin/env python3
"""MSRS inference for the V10-based region detail V12."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nets.Ufuser_v10_region_detail_v12 import UfuserV10RegionDetailV12  # noqa: E402


def build(checkpoint, device):
    cfg = checkpoint["config"]
    reference = torch.load(cfg["reference_v10"], map_location="cpu")
    base_cfg = reference["config"]
    model = UfuserV10RegionDetailV12(
        int(base_cfg.get("d_state", 16)), int(base_cfg.get("num_tokens", 9)),
        int(base_cfg.get("num_classes", 9)), float(base_cfg.get("initial_new", 0.01)),
        int(cfg.get("region_size", 16)), float(cfg.get("source_gain", 0.05)),
        float(cfg.get("agreement_gain", 0.15)), float(cfg.get("fused_gain", 0.10)),
        float(cfg.get("max_source_delta", 0.05)),
        float(cfg.get("max_fused_gain", 0.10)),
        float(cfg.get("max_intensity_gain", 0.01)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_head_strength(1.0); model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ir-dir", type=Path, required=True)
    p.add_argument("--vi-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opt = p.parse_args(); device = torch.device(opt.device)
    model = build(torch.load(opt.model, map_location="cpu"), device)
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(opt.ir_dir.glob("*.png"))
    if not paths: raise RuntimeError(f"No PNG images under {opt.ir_dir}")
    with torch.inference_mode():
        for path in tqdm(paths, desc="V12 inference"):
            infrared = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            visible_bgr = cv2.imread(str(opt.vi_dir / path.name), cv2.IMREAD_COLOR)
            if infrared is None or visible_bgr is None: raise FileNotFoundError(path.name)
            visible = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
            height, width = infrared.shape
            ir = torch.from_numpy(infrared).float()[None, None].to(device) / 255.0
            vi = torch.from_numpy(visible).float()[None, None].to(device) / 255.0
            ir = F.pad(ir, (0, (-width) % 16, 0, (-height) % 16), mode="reflect")
            vi = F.pad(vi, (0, (-width) % 16, 0, (-height) % 16), mode="reflect")
            output = model(ir, vi)[0, 0, :height, :width]
            image = np.rint(output.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if not cv2.imwrite(str(opt.output_dir / path.name), image):
                raise RuntimeError(f"Failed to write {path.name}")
    print(f"Saved {len(paths)} V12 images to {opt.output_dir.resolve()}")


if __name__ == "__main__": main()
