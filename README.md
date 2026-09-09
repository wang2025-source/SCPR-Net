<div align="center">

# SCPR-Net

### Spatial–Semantic Cross-Modal Prompt Routing Network

**Source-aware · Frequency-decoupled · Region-adaptive infrared–visible image fusion**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](environment.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](environment.yml)
[![Model Zoo](https://img.shields.io/badge/model-V22-7c3aed?style=for-the-badge)](MODEL_ZOO.md)
[![Reproducibility](https://img.shields.io/badge/reproducibility-SHA--256_verified-0f766e?style=for-the-badge)](SNAPSHOT_SHA256SUMS)

[Architecture](ARCHITECTURE.md) · [Reproducibility](REPRODUCIBILITY.md) · [Model Zoo](MODEL_ZOO.md) · [Datasets](#datasets)

</div>

<p align="center">
  <img src="assets/figures/图片10.png" width="100%" alt="SCPR-Net qualitative and quantitative overview">
</p>

## Architecture

<p align="center">
  <img src="assets/figures/图片1.png" width="100%" alt="SCPR-Net full network architecture">
</p>

<p align="center">
  <img src="assets/figures/图片3.png" width="100%" alt="Source-query and frequency-collaboration modules">
</p>

The full dependency closure of the network source is preserved in [`code/nets/`](code/nets/); per-module class names, forward paths, and weight-construction relationships are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick Start

### 1. Environment

Linux with an NVIDIA GPU and CUDA 12.1 is recommended.

```bash
conda env create -f environment.yml
conda activate scpr-net
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

`mamba-ssm` compiles CUDA extensions; install it after PyTorch and the CUDA toolkit are confirmed available.

### 2. Inference

Run fusion on aligned IR/visible pairs:

```bash
python tools/infer_scpr.py \
  --model code/model/EMMA_REGION_MOE_V22.ckpt \
  --v12-model code/model/EMMA_V10_REGION_DETAIL_V12_FIXED_STRONG.ckpt \
  --v10-model code/model/EMMA_SOURCE_WAVE_ASSM_V10_SCREEN_best_pareto.ckpt \
  --ir-dir code/test_img/ir \
  --vi-dir code/test_img/vi \
  --output-dir outputs/scpr-net \
  --device cuda
```

The original entry point [`code/repro/infer_region_moe_v19.py`](code/repro/infer_region_moe_v19.py) requires only `--model` and resolves the V12 stage from the path recorded in the checkpoint; `tools/infer_scpr.py` reconstructs the same V10 → V12 → V22 chain from explicit local paths, making the snapshot fully portable. Infrared and visible images must share the same filename and resolution. The visible image is converted to its Y channel and the output is a grayscale fused image. PNG, JPG, JPEG, BMP, TIF, and TIFF are supported.

## Datasets

The model is evaluated on five public infrared–visible benchmark datasets. All data are obtained from the official sources below, which include the standard test splits.

| Dataset | Modalities | Pairs | Evaluation | Official source |
|---|---|---|---|---|
| **MSRS** | IR + Visible | 1,444 | Quantitative, semantic segmentation | [github.com/Linfeng-Tang/MSRS](https://github.com/Linfeng-Tang/MSRS) |
| **TNO** | IR + Visible | multi-scene night-time | Quantitative | [figshare.com (TNO Image Fusion Dataset)](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) |
| **RoadScene** | IR + Visible | 221 | Qualitative | [github.com/hanna-xu/RoadScene](https://github.com/hanna-xu/RoadScene) |
| **M³FD** | IR + Visible | 4,200 | Qualitative, object detection | [github.com/JinyuanLiu-CV/TarDAL](https://github.com/JinyuanLiu-CV/TarDAL) |
| **LLVIP** | IR + Visible | 15,488 | Qualitative | [bupt-ai-cz.github.io/LLVIP](https://bupt-ai-cz.github.io/LLVIP/) |

Refer to [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact directory layout and evaluation protocol.

## Experimental Results

### Quantitative comparison on MSRS

<p align="center">
  <img src="assets/figures/MSRS_quantitative_comparison_roman.png" width="100%" alt="MSRS quantitative comparison">
</p>

### Qualitative comparison across five datasets

<p align="center">
  <img src="assets/figures/图片6.png" width="100%" alt="Qualitative comparison on MSRS, TNO, and RoadScene">
</p>

<p align="center">
  <img src="assets/figures/图片5.png" width="100%" alt="Qualitative comparison on M3FD and LLVIP">
</p>

### Analysis

<p align="center">
  <img src="assets/figures/00004N_frequency_routing_enhancement.png" width="100%" alt="Frequency-routing response visualization">
</p>

<p align="center">
  <img src="assets/figures/feature_evolution_00581D.png" width="100%" alt="Feature evolution across the network">
</p>

<p align="center">
  <img src="assets/figures/图片7.png" width="100%" alt="Prototype and frequency branch ablation">
</p>

<p align="center">
  <img src="assets/figures/图片11.png" width="100%" alt="Semantic segmentation comparison">
</p>

<p align="center">
  <img src="assets/figures/图片12.png" width="100%" alt="Detection-oriented comparison">
</p>

## Repository Layout

```text
SCPR-Net/
├── code/
│   ├── nets/                  # V22 complete network dependency closure
│   ├── repro/                 # training, inference, and evaluation sources
│   ├── model/                 # V10 → V12 → V22 checkpoint chain
│   └── test_img/              # three aligned IR/visible test pairs
├── assets/figures/            # figures used in this README
├── tools/infer_scpr.py        # portable V22 constructor and inference
├── tools/verify_snapshot.py   # SHA-256 verification
├── ARCHITECTURE.md            # implementation-level architecture map
├── REPRODUCIBILITY.md         # environment, data, training, and evaluation
├── MODEL_ZOO.md               # checkpoint roles and sizes
└── SNAPSHOT_SHA256SUMS        # immutable snapshot manifest
```

## Reproducibility

- Pinned environment: Python 3.10, PyTorch 2.2.2, CUDA 12.1.
- Default random seed: `3407`.
- Final weights: `EMMA_REGION_MOE_V22.ckpt`.
- The portable entry point loads the V10 → V12 → V22 chain explicitly, bypassing absolute paths baked into the training checkpoints.
- Snapshot hashes and Python syntax are checked automatically in CI.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full protocol.
