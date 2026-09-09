#!/usr/bin/env python3
"""Evaluate fused images with the metrics used by the EMMA paper."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm

EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--vi-dir", type=Path, required=True)
    parser.add_argument("--fused-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image.astype(np.float64)


def entropy(image):
    values = np.uint8(np.round(image)).ravel()
    histogram = np.bincount(values, minlength=256) / values.size
    return float(-np.sum(histogram * np.log2(histogram + (histogram == 0))))


def standard_deviation(image):
    return float(np.std(image))


def spatial_frequency(image):
    row = np.mean((image[:, 1:] - image[:, :-1]) ** 2)
    column = np.mean((image[1:, :] - image[:-1, :]) ** 2)
    return float(np.sqrt(row + column))


def average_gradient(image):
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    gy[0, :] = image[1, :] - image[0, :]
    gy[-1, :] = image[-1, :] - image[-2, :]
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
    return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2)))


def correlation(left, right):
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    if denominator == 0:
        return 0.0
    return float(np.sum(left * right) / denominator)


def scd(fused, infrared, visible):
    return correlation(infrared, fused - visible) + correlation(visible, fused - infrared)


def pair_vif(reference, distorted):
    sigma_nsq = 2
    eps = 1e-10
    numerator = 0.0
    denominator = 0.0
    for scale in range(1, 5):
        size = 2 ** (4 - scale + 1) + 1
        sigma = size / 5.0
        middle = (size - 1.0) / 2.0
        y, x = np.ogrid[-middle:middle + 1, -middle:middle + 1]
        kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
        kernel /= kernel.sum()

        if scale > 1:
            reference = convolve2d(reference, np.rot90(kernel, 2), mode="valid")[::2, ::2]
            distorted = convolve2d(distorted, np.rot90(kernel, 2), mode="valid")[::2, ::2]

        mu_reference = convolve2d(reference, np.rot90(kernel, 2), mode="valid")
        mu_distorted = convolve2d(distorted, np.rot90(kernel, 2), mode="valid")
        reference_variance = (
            convolve2d(reference * reference, np.rot90(kernel, 2), mode="valid")
            - mu_reference * mu_reference
        )
        distorted_variance = (
            convolve2d(distorted * distorted, np.rot90(kernel, 2), mode="valid")
            - mu_distorted * mu_distorted
        )
        covariance = (
            convolve2d(reference * distorted, np.rot90(kernel, 2), mode="valid")
            - mu_reference * mu_distorted
        )
        reference_variance[reference_variance < 0] = 0
        distorted_variance[distorted_variance < 0] = 0
        gain = covariance / (reference_variance + eps)
        residual_variance = distorted_variance - gain * covariance
        gain[reference_variance < eps] = 0
        residual_variance[reference_variance < eps] = distorted_variance[reference_variance < eps]
        reference_variance[reference_variance < eps] = 0
        gain[distorted_variance < eps] = 0
        residual_variance[distorted_variance < eps] = 0
        residual_variance[gain < 0] = distorted_variance[gain < 0]
        gain[gain < 0] = 0
        residual_variance[residual_variance <= eps] = eps
        numerator += np.sum(
            np.log10(1 + gain * gain * reference_variance / (residual_variance + sigma_nsq))
        )
        denominator += np.sum(np.log10(1 + reference_variance / sigma_nsq))
    value = numerator / denominator
    return 1.0 if np.isnan(value) else float(value)


def vif(fused, infrared, visible):
    return pair_vif(infrared, fused) + pair_vif(visible, fused)


def main():
    opt = arguments()
    for directory in (opt.ir_dir, opt.vi_dir, opt.fused_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    infrared = {p.stem: p for p in opt.ir_dir.iterdir() if p.suffix.lower() in EXTENSIONS}
    visible = {p.stem: p for p in opt.vi_dir.iterdir() if p.suffix.lower() in EXTENSIONS}
    fused = {p.stem: p for p in opt.fused_dir.iterdir() if p.suffix.lower() in EXTENSIONS}
    if set(infrared) != set(visible) or set(infrared) != set(fused):
        raise RuntimeError(
            f"Filename mismatch: IR={len(infrared)}, VI={len(visible)}, fused={len(fused)}"
        )

    rows = []
    for name in tqdm(sorted(infrared), desc="Evaluating"):
        ir = read_gray(infrared[name])
        vi = read_gray(visible[name])
        fu = read_gray(fused[name])
        if ir.shape != vi.shape or ir.shape != fu.shape:
            raise ValueError(f"Shape mismatch for {name}: {ir.shape}, {vi.shape}, {fu.shape}")
        rows.append({
            "image": name,
            "EN": entropy(fu),
            "SD": standard_deviation(fu),
            "SF": spatial_frequency(fu),
            "AG": average_gradient(fu),
            "SCD": scd(fu, ir, vi),
            "VIF": vif(fu, ir, vi),
        })

    names = ("EN", "SD", "SF", "AG", "SCD", "VIF")
    summary = {"images": len(rows)}
    summary.update({name: float(np.mean([row[name] for row in rows])) for name in names})
    summary.update({f"{name}_std": float(np.std([row[name] for row in rows])) for name in names})

    opt.output_dir.mkdir(parents=True, exist_ok=True)
    with (opt.output_dir / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image",) + names)
        writer.writeheader()
        writer.writerows(rows)
    with (opt.output_dir / "summary_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "mean", "std"))
        for name in names:
            writer.writerow((name, summary[name], summary[f"{name}_std"]))
    with (opt.output_dir / "summary_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Images: {len(rows)}")
    print("       EN       SD       SF       AG      SCD      VIF")
    print(" ".join(f"{summary[name]:8.4f}" for name in names))
    print(f"Saved metrics to {opt.output_dir.resolve()}")


if __name__ == "__main__":
    main()
