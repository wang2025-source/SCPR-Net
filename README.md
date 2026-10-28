<div align="center">

# SCPR-Net

**Source-Conditioned Hierarchical Collaboration and Progressive Spatial-Frequency Reconstruction for Infrared and Visible Image Fusion**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](environment.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](environment.yml)
[![Model](https://img.shields.io/badge/Model-SCPR--Net-7c3aed?style=flat-square)](code/nets/scpr_net.py)
</div>

<p align="center">
  <img src="assets/figures/overview_comparison.png" width="100%" alt="SCPR-Net qualitative and quantitative overview">
</p>

## Architecture

<p align="center">
  <img src="assets/figures/network_architecture.png" width="100%" alt="SCPR-Net full network architecture">
</p>

<p align="center">
  <img src="assets/figures/source_frequency_collaboration.png" width="100%" alt="Source-query and frequency-collaboration modules">
</p>

## Quick Start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate scpr-net
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

### 2. Testing

#### Samples

```bash
python tools/infer_scpr.py --model outputs/checkpoints/SCPR-Net_best.ckpt
```

#### Benchmark

```bash
python tools/infer_scpr.py \
  --model outputs/checkpoints/SCPR-Net_best.ckpt \
  --ir-dir datasets/MSRS-main/test/ir \
  --vi-dir datasets/MSRS-main/test/vi \
  --output-dir outputs/MSRS
```

#### Evaluation

```bash
python code/repro/evaluate_metrics_parallel.py \
  --ir-dir datasets/MSRS-main/test/ir \
  --vi-dir datasets/MSRS-main/test/vi \
  --fused-dir outputs/MSRS \
  --output-dir outputs/metrics/MSRS
```

## Training

```bash
python tools/train_scpr.py \
  --data-root datasets/MSRS-main \
  --output-dir outputs/checkpoints
```

### Resume

```bash
python tools/train_scpr.py --data-root datasets/MSRS-main \
  --resume outputs/checkpoints/SCPR-Net_last.ckpt
```

### Check

```bash
python tools/check_scpr.py
```

## Datasets

The model is evaluated on five public infrared–visible benchmark datasets. All data are obtained from the official sources below, which include the standard test splits.

| Dataset | Modalities | Pairs | Evaluation | Official source |
|---|---|---|---|---|
| **MSRS** | IR + Visible | 1,444 | Quantitative, semantic segmentation | [github.com/Linfeng-Tang/MSRS](https://github.com/Linfeng-Tang/MSRS) |
| **TNO** | IR + Visible | multi-scene night-time | Quantitative | [figshare.com (TNO Image Fusion Dataset)](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) |
| **RoadScene** | IR + Visible | 221 | Qualitative | [github.com/hanna-xu/RoadScene](https://github.com/hanna-xu/RoadScene) |
| **M³FD** | IR + Visible | 4,200 | Qualitative, object detection | [github.com/JinyuanLiu-CV/TarDAL](https://github.com/JinyuanLiu-CV/TarDAL) |
| **LLVIP** | IR + Visible | 15,488 | Qualitative | [bupt-ai-cz.github.io/LLVIP](https://bupt-ai-cz.github.io/LLVIP/) |

## Experimental Results

### Quantitative comparison on MSRS

<p align="center">
  <img src="assets/figures/MSRS_quantitative_comparison_roman.png" width="100%" alt="MSRS quantitative comparison">
</p>

### Qualitative comparison across five datasets

<p align="center">
  <img src="assets/figures/qualitative_main.png" width="100%" alt="Qualitative comparison on MSRS, TNO, and RoadScene">
</p>

<p align="center">
  <img src="assets/figures/qualitative_cross_domain.png" width="100%" alt="Qualitative comparison on M3FD and LLVIP">
</p>

### Analysis

<p align="center">
  <img src="assets/figures/00004N_frequency_routing_enhancement.png" width="100%" alt="Frequency-routing response visualization">
</p>

<p align="center">
  <img src="assets/figures/feature_evolution_00581D.png" width="100%" alt="Feature evolution across the network">
</p>

<p align="center">
  <img src="assets/figures/ablation_analysis.png" width="100%" alt="Prototype and frequency branch ablation">
</p>

<p align="center">
  <img src="assets/figures/segmentation_comparison.png" width="100%" alt="Semantic segmentation comparison">
</p>

<p align="center">
  <img src="assets/figures/detection_comparison.png" width="100%" alt="Detection-oriented comparison">
</p>
