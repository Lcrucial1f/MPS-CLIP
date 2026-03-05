下面给你一份“可翻页（用锚点导航模拟翻页）+ 更美观”的 README 版本，直接替换你当前的 `README.md` 即可。
说明：GitHub README 不支持真正的轮播组件（carousel），但可以用 **目录/缩略图 + 上一页/下一页锚点** 达到“翻页浏览”的体验。

> 你当前有个文件名是 `visualiazation.png`（拼写带 `a`），我在 README 里保持不改名以免 404；如果你愿意改成 `visualization.png`，记得同步改 README 路径。

---

```markdown
<div align="center">

<!-- Optional logo -->
<!-- <img src="tools/docs/logo.png" alt="MPS-CLIP Logo" width="180" /> -->

# MPS-CLIP

**Multi-Scale Prompting for Remote Sensing Image-Text Retrieval with CLIP**

[![arXiv](https://img.shields.io/badge/arXiv-2601.18190-b31b1b.svg)](https://arxiv.org/abs/2601.18190)
[![Project Page](https://img.shields.io/badge/Project%20Page-MPS--CLIP-blue)](https://lcrucial1f.github.io/)
[![HuggingFace Datasets](https://img.shields.io/badge/HuggingFace-Datasets-orange)](https://huggingface.co/datasets/lcrucial1f/MPS-CLIP_Data/tree/main)

**Model Pipeline (PDF):** [assets/pipeline.pdf](assets/pipeline.pdf)

</div>

---

## Table of Contents
- [Overview](#overview)
  - [Pipeline](#pipeline)
  - [Visualization Gallery](#visualization-gallery)
- [Installation](#installation)
- [Prepare Data](#prepare-data)
- [Training](#training)
- [Testing](#testing)

---

## Overview

### Pipeline

GitHub cannot preview PDF inline in README.
Use the image preview below (click to open the PDF):

<p align="center">
  <a href="assets/pipeline.pdf">
    <!-- Recommended: export the first page of pipeline.pdf to assets/pipeline.png -->
    <img src="assets/pipeline.png" alt="Pipeline (click to open PDF)" width="92%" />
  </a>
</p>

<p align="center">
  <a href="assets/pipeline.pdf"><b>Open PDF: assets/pipeline.pdf</b></a>
</p>

---

### Visualization Gallery

A paged gallery for qualitative results.
Click a thumbnail, then use **Prev/Next** links to flip pages.

<p align="center">
  <a href="#vis-1"><img src="assets/visualiazation.png" width="22%" alt="Vis 1 thumbnail" /></a>
  <a href="#vis-2"><img src="assets/visual_2.png" width="22%" alt="Vis 2 thumbnail" /></a>
  <a href="#vis-3"><img src="assets/visual_3.png" width="22%" alt="Vis 3 thumbnail" /></a>
  <a href="#vis-4"><img src="assets/visual_4.png" width="22%" alt="Vis 4 thumbnail" /></a>
</p>

---

<a id="vis-1"></a>
#### Page 1 / 4 — Visualization Summary

<p align="center">
  <a href="assets/visualiazation.png">
    <img src="assets/visualiazation.png" alt="Visualization Summary" width="92%" />
  </a>
</p>

<p align="center">
  <a href="#vis-4">Prev</a> |
  <a href="#vis-2">Next</a> |
  <a href="#visualization-gallery">Back to Gallery</a>
</p>

---

<a id="vis-2"></a>
#### Page 2 / 4 — visual_2

<p align="center">
  <a href="assets/visual_2.png">
    <img src="assets/visual_2.png" alt="visual_2" width="92%" />
  </a>
</p>

<p align="center">
  <a href="#vis-1">Prev</a> |
  <a href="#vis-3">Next</a> |
  <a href="#visualization-gallery">Back to Gallery</a>
</p>

---

<a id="vis-3"></a>
#### Page 3 / 4 — visual_3

<p align="center">
  <a href="assets/visual_3.png">
    <img src="assets/visual_3.png" alt="visual_3" width="92%" />
  </a>
</p>

<p align="center">
  <a href="#vis-2">Prev</a> |
  <a href="#vis-4">Next</a> |
  <a href="#visualization-gallery">Back to Gallery</a>
</p>

---

<a id="vis-4"></a>
#### Page 4 / 4 — visual_4

<p align="center">
  <a href="assets/visual_4.png">
    <img src="assets/visual_4.png" alt="visual_4" width="92%" />
  </a>
</p>

<p align="center">
  <a href="#vis-3">Prev</a> |
  <a href="#vis-1">Next</a> |
  <a href="#visualization-gallery">Back to Gallery</a>
</p>

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Prepare Data

All experiments are based on **RSITMD** and **RSICD**.

- Datasets (HuggingFace): https://huggingface.co/datasets/lcrucial1f/MPS-CLIP_Data/tree/main

After downloading and organizing the datasets, modify the corresponding `configs/yaml` file:

- For **RSICD**:
```yaml
image_root: 'YOUR_OWN_PATH/rsicd'
```

- For **RSITMD**:
```yaml
image_root: 'YOUR_OWN_PATH/rsitmd'
```

Annotation files are in `data/finetune`.

---

## Training

### Step 1: Download Pretrained Weights (GeoRSCLIP)

Download GeoRSCLIP from:
- https://huggingface.co/Zilun/GeoRSCLIP/blob/main/ckpt/RS5M_ViT-B-32_RET-2.pt

Place the checkpoint into:
```text
models/pretrain/
```

### Step 2: (Optional) Multi-GPU / Distributed Setup

If needed, modify `get_dist_launch` in `run.py`. Example for 2 GPUs:

```python
elif args.dist == 'f2':
        return "CUDA_VISIBLE_DEVICES=8,9 WORLD_SIZE=2 YOUR_OWN_PYTHON_PATH -W ignore -m torch.distributed.launch --master_port 25903 --nproc_per_node=2 " \
               "--nnodes=1 "
```

Replace:
- `CUDA_VISIBLE_DEVICES=8,9` with your GPU IDs
- `YOUR_OWN_PYTHON_PATH` with your python path (e.g., `/root/miniconda3/bin/python`)

### Step 3: Start Training

```bash
python run.py --task 'itr_rsitmd_vit' --dist "f2" --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/full_rsitmd_vit'

python run.py --task 'itr_rsicd_vit' --dist "f2" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/full_rsicd_vit'
```

---

## Testing

Set `if_evaluation: True` in the corresponding `configs/yaml`, then run:

```bash
python run.py --task 'itr_rsitmd_vit' --dist "f2" --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/test' --checkpoint './checkpoints/MPS-CLIP/full_rsitmd_vit/checkpoint_best.pth' --evaluate

python run.py --task 'itr_rsicd_vit' --dist "f2" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/test' --checkpoint './checkpoints/MPS-CLIP/full_rsicd_vit/checkpoint_best.pth' --evaluate
```
```

---
