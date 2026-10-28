"""Strict, self-contained checkpoint loading shared by training and testing."""

from pathlib import Path

import torch

from nets.scpr_net import SCPRConfig, SCPRNet

FORMAT = "scpr-net-paper"


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run tools/train_scpr.py first or supply --model.")
    # Training checkpoints contain optimizer/RNG state. Only load trusted files.
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_model(checkpoint, device="cpu", scan_backend="auto"):
    if checkpoint.get("format") != FORMAT:
        raise ValueError("This checkpoint does not describe the manuscript architecture. "
                         "Supply a SCPR-Net checkpoint produced by tools/train_scpr.py; "
                         "incompatible tensors are never silently reshaped or skipped.")
    model = SCPRNet(SCPRConfig(**checkpoint["model_config"]), scan_backend=scan_backend).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
