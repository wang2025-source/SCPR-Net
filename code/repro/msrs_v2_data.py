"""Data utilities shared by the independent EMMA V2 experiments.

The split is made from contiguous filename groups instead of individual frames.
MSRS contains neighbouring video frames, so grouping reduces train/validation
leakage compared with a plain random image split.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def aligned_names(root: Path, split: str = "train") -> list[str]:
    directory = Path(root) / split
    names = sorted(path.name for path in (directory / "ir").glob("*.png"))
    if not names:
        raise RuntimeError(f"No MSRS images under {directory / 'ir'}")
    for name in names:
        for folder in ("vi", "Segmentation_labels"):
            path = directory / folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
    return names


def _sequence_group(name: str, group_size: int) -> tuple[str, int]:
    stem = Path(name).stem
    match = re.match(r"(\d+)(.*)", stem)
    if match is None:
        return stem, 0
    frame = int(match.group(1))
    condition = match.group(2)
    return condition, frame // group_size


def grouped_train_val_split(
    names: Sequence[str],
    val_fraction: float = 0.1,
    seed: int = 3407,
    group_size: int = 50,
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[tuple[str, int], list[str]] = {}
    for name in names:
        groups.setdefault(_sequence_group(name, group_size), []).append(name)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(names) * val_fraction))
    val_keys: list[tuple[str, int]] = []
    count = 0
    for key in keys:
        if count >= target and val_keys:
            break
        val_keys.append(key)
        count += len(groups[key])
    val_set = {name for key in val_keys for name in groups[key]}
    train_names = [name for name in names if name not in val_set]
    val_names = [name for name in names if name in val_set]
    if not train_names or not val_names:
        raise RuntimeError("Grouped split produced an empty partition")
    return train_names, val_names


class MSRSAlignedV2(Dataset):
    """Aligned IR/Y/label samples with optional repeated random crops."""

    def __init__(
        self,
        root: Path,
        names: Sequence[str],
        split: str = "train",
        crop_size: int | None = 128,
        augment: bool = False,
        samples_per_epoch: int | None = None,
    ) -> None:
        self.directory = Path(root) / split
        self.names = list(names)
        self.crop_size = crop_size
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        if not self.names:
            raise ValueError("Dataset names cannot be empty")

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index % len(self.names)]
        ir = cv2.imread(str(self.directory / "ir" / name), cv2.IMREAD_GRAYSCALE)
        vi_bgr = cv2.imread(str(self.directory / "vi" / name), cv2.IMREAD_COLOR)
        label = cv2.imread(
            str(self.directory / "Segmentation_labels" / name), cv2.IMREAD_GRAYSCALE
        )
        if ir is None or vi_bgr is None or label is None:
            raise RuntimeError(f"Failed to read aligned sample {name}")
        vi = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2YCrCb)[..., 0]

        if self.crop_size is not None:
            size = self.crop_size
            height, width = ir.shape
            if min(height, width) < size:
                raise ValueError(f"{name} is smaller than crop size {size}")
            if self.augment:
                top = random.randint(0, height - size)
                left = random.randint(0, width - size)
            else:
                top = (height - size) // 2
                left = (width - size) // 2
            selection = np.s_[top : top + size, left : left + size]
            ir, vi, label = ir[selection], vi[selection], label[selection]

        if self.augment and random.random() < 0.5:
            ir, vi, label = np.fliplr(ir), np.fliplr(vi), np.fliplr(label)
        if self.augment and random.random() < 0.5:
            ir, vi, label = np.flipud(ir), np.flipud(vi), np.flipud(label)

        ir_tensor = torch.from_numpy(np.ascontiguousarray(ir)).float()[None] / 255.0
        vi_tensor = torch.from_numpy(np.ascontiguousarray(vi)).float()[None] / 255.0
        label_tensor = torch.from_numpy(np.ascontiguousarray(label)).long()
        return ir_tensor, vi_tensor, label_tensor, name


def segmentation_class_weights(
    root: Path, names: Iterable[str], num_classes: int = 9
) -> torch.Tensor:
    """Inverse-square-root weights, normalized and clipped for stability."""
    counts = np.zeros(num_classes, dtype=np.int64)
    label_dir = Path(root) / "train" / "Segmentation_labels"
    for name in names:
        label = cv2.imread(str(label_dir / name), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise RuntimeError(f"Failed to read {label_dir / name}")
        counts += np.bincount(label.ravel(), minlength=num_classes)[:num_classes]
    frequencies = counts / max(counts.sum(), 1)
    weights = 1.0 / np.sqrt(np.maximum(frequencies, 1e-8))
    weights /= weights.mean()
    weights = np.clip(weights, 0.1, 5.0)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def update_confusion(
    confusion: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    valid = (target >= 0) & (target < num_classes)
    indices = target[valid] * num_classes + prediction[valid]
    confusion += torch.bincount(
        indices.cpu(), minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def confusion_metrics(confusion: torch.Tensor) -> dict[str, object]:
    matrix = confusion.double()
    intersection = matrix.diag()
    union = matrix.sum(1) + matrix.sum(0) - intersection
    iou = intersection / union.clamp_min(1)
    valid = union > 0
    accuracy = intersection.sum() / matrix.sum().clamp_min(1)
    return {
        "pixel_accuracy": float(accuracy),
        "miou": float(iou[valid].mean()),
        "class_iou": [float(value) for value in iou],
    }
