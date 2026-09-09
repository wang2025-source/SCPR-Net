# SCPR-Net

SCPR-Net 是一个面向红外–可见光图像融合的可复现发布仓库。当前发布版本以 **V22** 为唯一主模型，围绕源条件提示、频率路由、语义区域专家与细节增强组织模型和实验材料。

> 源代码保护：`code/` 是从原工作目录复制的只读发布快照，未改写任何原有源码。仓库外层仅新增发布文档、校验工具和便携推理入口。

![Frequency-routing visualization](assets/figures/00004N_frequency_routing_enhancement.png)

## 仓库结构

```text
SCPR-Net/
├── code/                    # V22 所需的原样代码、测试样例和权重链
├── assets/figures/          # 原“图片”目录中的全部 PNG/PDF
├── tools/infer_scpr.py      # 不改原源码的便携 SCPR 推理入口
├── tools/verify_snapshot.py # SHA-256 完整性校验
├── REPRODUCIBILITY.md       # 环境、数据、训练、推理和评测说明
├── MODEL_ZOO.md             # 权重清单与依赖关系
└── SNAPSHOT_SHA256SUMS      # 代码和图片的固定哈希
```

## 快速开始

### 1. 创建环境

推荐 Linux、NVIDIA GPU、CUDA 12.1：

```bash
conda env create -f environment.yml
conda activate scpr-net
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

只复现原始 EMMA 基线推理时不需要 `mamba-ssm`。

### 2. 校验发布快照

```bash
python tools/verify_snapshot.py
```

### 3. V22 推理（推荐 GPU）

```bash
python tools/infer_scpr.py \
  --model code/model/EMMA_REGION_MOE_V22.ckpt \
  --v12-model code/model/EMMA_V10_REGION_DETAIL_V12_FIXED_STRONG.ckpt \
  --v10-model code/model/EMMA_SOURCE_WAVE_ASSM_V10_SCREEN_best_pareto.ckpt \
  --ir-dir code/test_img/ir \
  --vi-dir code/test_img/vi \
  --output-dir outputs/scpr-net
```

完整的数据布局、训练入口、评测方式和复现边界见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。权重说明见 [MODEL_ZOO.md](MODEL_ZOO.md)。

## 输入约定

红外图和可见光图必须同名、同尺寸：

```text
my_data/
├── ir/001.png
└── vi/001.png
```

可见光输入会转换到 Y 通道，输出为灰度融合图。支持 PNG、JPG、JPEG、BMP、TIF 和 TIFF。

## 完整性与可追溯性

- `SNAPSHOT_SHA256SUMS` 固定了 `code/` 与 `assets/figures/` 中每个发布文件的 SHA-256。
- GitHub Actions 会检查源码语法并验证哈希。
- 不参与 V22 的 Ai/Av、EMMA 基线、旧实验权重、生成结果、训练日志和第三方仓库副本均未提交。
- V22 仍沿用源码中的 `UfuserRegionMoEV19` 类名；为保证源码不变，没有做表面改名。

## 来源与引用

本项目代码以 EMMA（CVPR 2024）为基础继续实验。请同时阅读 [NOTICE.md](NOTICE.md) 并引用原工作：

```bibtex
@InProceedings{Zhao_2024_CVPR,
  author    = {Zhao, Zixiang and Bai, Haowen and Zhang, Jiangshe and Zhang, Yulun and Zhang, Kai and Xu, Shuang and Chen, Dongdong and Timofte, Radu and Van Gool, Luc},
  title     = {Equivariant Multi-Modality Image Fusion},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```

## 许可说明

本快照没有擅自新增开源许可证。使用、再发布或商业化前，请分别核对上游 EMMA 与 `code/third_party/` 中各项目的许可条款。
