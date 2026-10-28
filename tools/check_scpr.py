#!/usr/bin/env python3
"""CPU architecture, equation, gradient and train/resume/test smoke checks."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from nets.scpr_net import (SCPRConfig, SCPRNet, AttentiveSelectiveScan,
                           haar_decompose, haar_reconstruct, three_bands)
from nets.scpr_loss import LossConfig, SCPRLoss
from repro.scpr_checkpoint import FORMAT, restore_model

torch.set_num_threads(2)


class ArchitectureChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)

    def test_exact_frequency_reconstruction(self):
        image = torch.rand(2, 3, 16, 20)
        torch.testing.assert_close(haar_reconstruct(*haar_decompose(image)), image)
        torch.testing.assert_close(sum(three_bands(image)), image)

    def test_scan_discretization(self):
        scan = AttentiveSelectiveScan(4, 3, backend="torch")
        tokens, prompt = torch.rand(2, 5, 4), torch.rand(2, 5, 3)
        actual = scan(tokens, prompt)
        raw_dt, bs, cs = torch.split(scan.project(tokens), (scan.rank, 3, 3), -1)
        dt = F.softplus(scan.delta(raw_dt))
        a = -scan.A_log.exp()
        expected = torch.zeros_like(actual)
        for batch in range(2):
            for channel in range(4):
                state = torch.zeros(3)
                for step in range(5):
                    state = (dt[batch, step, channel] * a[channel]).exp() * state + (
                        dt[batch, step, channel] * bs[batch, step] * tokens[batch, step, channel])
                    expected[batch, step, channel] = (state * (cs[batch, step] + prompt[batch, step])).sum() + (
                        scan.skip[channel] * tokens[batch, step, channel])
        torch.testing.assert_close(actual, expected)

    def test_four_scales_and_all_parameter_gradients(self):
        model = SCPRNet(scan_backend="torch")
        shapes = []
        handles = [module.register_forward_hook(lambda _m, _x, y: shapes.append(tuple(y.shape)))
                   for module in model.encoder_ir]
        ir, vi = torch.rand(1, 1, 32, 48), torch.rand(1, 1, 32, 48)
        output = model(ir, vi, return_aux=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes, [(1, 8, 32, 48), (1, 16, 16, 24), (1, 32, 8, 12), (1, 64, 4, 6)])
        self.assertEqual(model.detail_enhancement.region_size, 8)
        self.assertEqual(model.frequency_routing.region_size, 8)
        for key in ("low_weights", "coarse_weights", "fine_weights"):
            torch.testing.assert_close(output[key].sum(1), torch.ones(1, 32, 48))
        self.assertTrue(bool((output["fused"] >= 0).all() and (output["fused"] <= 1).all()))
        self.assertLessEqual(float(output["correction"].detach().abs().max()), 0.1)
        labels = torch.randint(0, 9, (1, 32, 48))
        labels[:, 0, 0] = 255
        losses = SCPRLoss()(output, ir, vi, labels)
        self.assertEqual(set(losses), {"total", "visual", "frequency", "semantic", "route"})
        torch.testing.assert_close(losses["total"], sum(losses[k] for k in ("visual", "frequency", "semantic", "route")))
        losses["total"].backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_normal_training_crop(self):
        model = SCPRNet(scan_backend="torch").eval()
        with torch.inference_mode():
            output = model(torch.rand(1, 1, 128, 128), torch.rand(1, 1, 128, 128))
        self.assertEqual(tuple(output.shape), (1, 1, 128, 128))

    def test_checkpoint_roundtrip_and_incompatible_guard(self):
        model = SCPRNet(scan_backend="torch").eval()
        checkpoint = {"format": FORMAT, "model_config": model.config.to_dict(), "model": model.state_dict()}
        recovered = restore_model(checkpoint, scan_backend="torch").eval()
        ir, vi = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(model(ir, vi), recovered(ir, vi), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "manuscript architecture"):
            restore_model({"model": model.state_dict()})
        with self.assertRaises(ValueError):
            SCPRConfig(channels=(8, 16, 32, 32))

    def test_routing_balance_formula(self):
        image = torch.rand(1, 1, 32, 32)
        masks = torch.zeros(1, 3, 32, 32)
        masks[:, 2] = 1
        out = {"fused": image, "semantic_logits": torch.rand(1, 9, 32, 32),
               "low_weights": masks, "coarse_weights": masks, "fine_weights": masks}
        loss = SCPRLoss()(out, image, image, torch.zeros(1, 32, 32, dtype=torch.long))
        torch.testing.assert_close(loss["route"], torch.tensor(0.16))
        with self.assertRaises(ValueError):
            SCPRLoss()(out, image, image, torch.full((1, 32, 32), 255, dtype=torch.long))


class CommandLineChecks(unittest.TestCase):
    def run_command(self, *arguments):
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_training_resume_and_padded_inference(self):
        with tempfile.TemporaryDirectory(prefix="scpr-check-") as temporary:
            work = Path(temporary)
            data_root = work / "dataset"
            rng = np.random.default_rng(3407)
            for name in ("00001D.png", "00061D.png", "00121D.png", "00181D.png"):
                for modality in ("ir", "vi", "Segmentation_labels"):
                    directory = data_root / "train" / modality
                    directory.mkdir(parents=True, exist_ok=True)
                    image = rng.integers(0, 9 if modality == "Segmentation_labels" else 256, (32, 32), dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(directory / name), image))
            arguments = [ROOT / "tools/train_scpr.py", "--data-root", data_root, "--epochs", "2",
                         "--batch-size", "1", "--crop-size", "32", "--workers", "0", "--samples-per-epoch", "2",
                         "--device", "cpu", "--scan-backend", "torch", "--no-amp"]
            interrupted = work / "interrupted"
            uninterrupted = work / "uninterrupted"
            self.run_command(*arguments, "--output-dir", interrupted, "--stop-after-epoch", "1")
            saved = torch.load(interrupted / "SCPR-Net_last.ckpt", weights_only=False)
            self.assertEqual(saved["epoch"], 1)
            self.run_command(ROOT / "tools/train_scpr.py", "--data-root", data_root, "--workers", "0",
                             "--device", "cpu", "--scan-backend", "torch", "--resume", interrupted / "SCPR-Net_last.ckpt",
                             "--output-dir", interrupted)
            self.run_command(*arguments, "--output-dir", uninterrupted)
            resumed = torch.load(interrupted / "SCPR-Net_last.ckpt", weights_only=False)
            complete = torch.load(uninterrupted / "SCPR-Net_last.ckpt", weights_only=False)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["split"], complete["split"])
            for key, value in resumed["model"].items():
                torch.testing.assert_close(value, complete["model"][key], rtol=0, atol=0)
            for modality in ("ir", "vi"):
                directory = work / modality
                directory.mkdir()
                self.assertTrue(cv2.imwrite(str(directory / "sample.png"), rng.integers(0, 256, (17, 35), dtype=np.uint8)))
            self.run_command(ROOT / "tools/infer_scpr.py", "--model", interrupted / "SCPR-Net_best.ckpt",
                             "--ir-dir", work / "ir", "--vi-dir", work / "vi", "--output-dir", work / "fused",
                             "--device", "cpu", "--scan-backend", "torch")
            image = cv2.imread(str(work / "fused/sample.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(image.shape, (17, 35))


if __name__ == "__main__":
    unittest.main(verbosity=2)
