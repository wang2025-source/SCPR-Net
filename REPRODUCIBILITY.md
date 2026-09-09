# Reproducibility Guide

## 1. Supported setup

SCPR-Net V22/ASSM experiments are intended for Linux with an NVIDIA GPU. The pinned reference environment is Python 3.10, PyTorch 2.2.2, torchvision 0.17.2 and CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate scpr-net
python -m pip install --no-build-isolation -r requirements-mamba.txt
python tools/verify_snapshot.py
```

`mamba-ssm` compiles CUDA extensions. Install it only after PyTorch and the CUDA toolkit are available. If compilation fails, confirm that `nvcc --version`, `python -c "import torch; print(torch.version.cuda)"`, and the CUDA version selected by conda agree.

## 2. V22 smoke test

The repository contains three aligned IR/visible test pairs and the complete V22 inference checkpoint chain:

```bash
python tools/infer_scpr.py \
  --model code/model/EMMA_REGION_MOE_V22.ckpt \
  --v12-model code/model/EMMA_V10_REGION_DETAIL_V12_FIXED_STRONG.ckpt \
  --v10-model code/model/EMMA_SOURCE_WAVE_ASSM_V10_SCREEN_best_pareto.ckpt \
  --ir-dir code/test_img/ir \
  --vi-dir code/test_img/vi \
  --output-dir outputs/v22-smoke \
  --device cuda
```

Expected outcome: three PNG files under `outputs/v22-smoke/`. This checks I/O, V10 → V12 → V22 construction, padding, weight loading and inference wiring; it is not a benchmark claim.

## 3. V22 checkpoint inference

The training checkpoints store paths from the machine on which they were produced. `tools/infer_scpr.py` reconstructs the V10 → V12 → V22 region-MoE chain from explicit local paths without editing the checkpoints or the original source files.

```bash
python tools/infer_scpr.py \
  --model code/model/EMMA_REGION_MOE_V22.ckpt \
  --v12-model code/model/EMMA_V10_REGION_DETAIL_V12_FIXED_STRONG.ckpt \
  --v10-model code/model/EMMA_SOURCE_WAVE_ASSM_V10_SCREEN_best_pareto.ckpt \
  --ir-dir /path/to/MSRS/test/ir \
  --vi-dir /path/to/MSRS/test/vi \
  --output-dir outputs/scpr-v22 \
  --device cuda
```

## 4. Dataset layout

The V8/V10/V12/V19 training path uses the MSRS directory layout below:

```text
datasets/MSRS-main/
├── train/
│   ├── ir/<name>.png
│   ├── vi/<name>.png
│   └── Segmentation_labels/<name>.png
└── test/
    ├── ir/<name>.png
    ├── vi/<name>.png
    └── Segmentation_labels/<name>.png
```

Every modality and label file must have the same filename. The loader groups adjacent frame names before creating the train/validation split to reduce video-frame leakage. The default random seed is `3407`.

## 5. Training entry points

Run commands from the repository root. The source files below are preserved unchanged.

Pareto/ASSM V8 training with explicit portable paths:

```bash
python code/repro/train_pareto_assm_v8.py \
  --data-root datasets/MSRS-main \
  --split-checkpoint code/model/MSRS_LRASPP_teacher_v2_best.ckpt \
  --init-emma code/model/EMMA_trained.pth \
  --output outputs/checkpoints/EMMA_PARETO_ASSM_V8.ckpt \
  --seed 3407
```

V22 was produced through the staged V10 → V12 → region-MoE continuation represented by `code/repro/train_region_moe_v19.py`. The preserved class and script retain the internal V19 name, while `EMMA_REGION_MOE_V22.ckpt` is the selected final model. Saved configurations contain the original training-machine paths. To resume the exact stage without changing source, reproduce those referenced paths or launch from the original training layout. Portable inference is provided; portable training-path rewriting is deliberately not performed because the supplied source must remain byte-identical.

## 6. Evaluation

```bash
python code/repro/evaluate_metrics_parallel.py \
  --ir-dir datasets/MSRS-main/test/ir \
  --vi-dir datasets/MSRS-main/test/vi \
  --fused-dir outputs/scpr-v22 \
  --output outputs/scpr-v22-metrics.json
```

Check the script's `--help` before a benchmark run because output flags and supported formats are defined by the preserved source. Use the same image set, resize policy and grayscale/Y-channel conversion for every compared method.

## 7. Files intentionally excluded

| Item | Reason | Restore location |
|---|---|---|
| Ai/Av and EMMA baseline files | Not used by the selected V22 inference chain | Not required |
| Old V19 and other experimental checkpoints | V22 is the selected best model | Not required |
| `outputs/`, `results/`, logs and temporary files | Generated artifacts, not source | Recreate with the commands above |
| Third-party repository snapshots and nested `.git/` directories | Not imported by the V22 runtime path | Not required |

## 8. Reproducibility checklist

- Record GPU model, driver, CUDA, PyTorch and `mamba-ssm` versions.
- Verify `SNAPSHOT_SHA256SUMS` before running.
- Use seed `3407` and retain the saved split checkpoint.
- Keep paired filenames and resolutions identical.
- Report exact checkpoint SHA-256 and command line.
- Do not compare metrics across different crops, color conversions or resized inputs.
