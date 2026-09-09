#!/usr/bin/env python3
"""Process-parallel wrapper using the exact evaluate_metrics formulas."""

from __future__ import annotations

import argparse, csv, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.evaluate_metrics import (EXTENSIONS, average_gradient, entropy,
    read_gray, scd, spatial_frequency, standard_deviation, vif)  # noqa: E402


def evaluate_one(task):
    name, ir_path, vi_path, fu_path = task
    cv2.setNumThreads(1)
    ir, vi, fu = read_gray(ir_path), read_gray(vi_path), read_gray(fu_path)
    if ir.shape != vi.shape or ir.shape != fu.shape:
        raise ValueError(f"Shape mismatch for {name}: {ir.shape}, {vi.shape}, {fu.shape}")
    return {"image": name, "EN": entropy(fu), "SD": standard_deviation(fu),
            "SF": spatial_frequency(fu), "AG": average_gradient(fu),
            "SCD": scd(fu, ir, vi), "VIF": vif(fu, ir, vi)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ir-dir',type=Path,required=True);p.add_argument('--vi-dir',type=Path,required=True)
    p.add_argument('--fused-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);opt=p.parse_args()
    maps=[]
    for directory in (opt.ir_dir,opt.vi_dir,opt.fused_dir):
        if not directory.is_dir():raise FileNotFoundError(directory)
        maps.append({x.stem:x for x in directory.iterdir() if x.suffix.lower() in EXTENSIONS})
    infrared,visible,fused=maps
    if set(infrared)!=set(visible) or set(infrared)!=set(fused):
        raise RuntimeError(f'Filename mismatch: IR={len(infrared)}, VI={len(visible)}, fused={len(fused)}')
    tasks=[(n,infrared[n],visible[n],fused[n]) for n in sorted(infrared)]
    with ProcessPoolExecutor(max_workers=opt.workers) as pool:
        rows=list(tqdm(pool.map(evaluate_one,tasks),total=len(tasks),desc='Evaluating'))
    names=('EN','SD','SF','AG','SCD','VIF');summary={'images':len(rows)}
    summary.update({n:float(np.mean([r[n] for r in rows])) for n in names})
    summary.update({f'{n}_std':float(np.std([r[n] for r in rows])) for n in names})
    opt.output_dir.mkdir(parents=True,exist_ok=True)
    with (opt.output_dir/'per_image_metrics.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=('image',)+names);w.writeheader();w.writerows(rows)
    with (opt.output_dir/'summary_metrics.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow(('metric','mean','std'))
        for n in names:w.writerow((n,summary[n],summary[f'{n}_std']))
    with (opt.output_dir/'summary_metrics.json').open('w') as h:json.dump(summary,h,indent=2)
    print(f'Images: {len(rows)}');print('       EN       SD       SF       AG      SCD      VIF')
    print(' '.join(f'{summary[n]:8.4f}' for n in names));print(f'Saved metrics to {opt.output_dir.resolve()}')


if __name__=='__main__':main()
