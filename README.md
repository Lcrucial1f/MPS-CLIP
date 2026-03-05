<div align="center">

<!-- You can add a logo here later -->
<!-- <img src="tools/docs/logo.png" alt="MPS-CLIP Logo" width="180" /> -->

# MPS-CLIP

Multi-Scale Prompting for Remote Sensing Image-Text Retrieval with CLIP

[![arXiv](https://img.shields.io/badge/arXiv-2601.18190-b31b1b.svg)](https://arxiv.org/abs/2601.18190)
[![Project Page](https://img.shields.io/badge/Project%20Page-MPS--CLIP-blue)](https://lcrucial1f.github.io/)
[![HuggingFace Datasets](https://img.shields.io/badge/🤗HuggingFace-Datasets-orange)](https://huggingface.co/datasets/lcrucial1f/MPS-CLIP_Data/tree/main)

**Model Pipeline:** [pipeline.pdf](assets/pipeline.pdf)

</div>

---

## Overview

### Pipeline

> Note: Some markdown renderers (e.g., GitHub) may not preview PDF inline. Please open it via the link above.

<p align="center">
  <a href="assets/pipeline.pdf">
    <b>Click to view: assets/pipeline.pdf</b>
  </a>
</p>

---

## Visualization Results

<p align="center">
  <img src="assets/visualiazation.png" alt="Visualization Summary" width="92%" />
</p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="assets/visual_2.png" alt="visual_2" style="width: 100%;">
      <br>
      visual_2
    </td>
    <td align="center" width="50%">
      <img src="assets/visual_3.png" alt="visual_3" style="width: 100%;">
      <br>
      visual_3
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/visual_4.png" alt="visual_4" style="width: 100%;">
      <br>
      visual_4
    </td>
    <td align="center" width="50%">
      <img src="assets/visualiazation.png" alt="visualiazation" style="width: 100%;">
      <br>
      visualiazation
    </td>
  </tr>
</table>

---

## Installation

### 1. Install Dependencies

Set up the environment by running:

```bash
pip install -r requirements.txt
```

---

## Prepare Data

All experiments are based on the **RSITMD** and **RSICD** datasets. Please refer to:

- Paper: https://arxiv.org/abs/2601.18190
- Demo page: https://lcrucial1f.github.io/
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

The annotation files for the datasets are located in the `data/finetune` directory.

---

## Training

### Step 1: Download Pretrained Weights (GeoRSCLIP)

Download the **GeoRSCLIP** pre-trained model from:

- https://huggingface.co/Zilun/GeoRSCLIP/blob/main/ckpt/RS5M_ViT-B-32_RET-2.pt

Place the checkpoint in:

```text
models/pretrain/
```

### Step 2: (Optional) Multi-GPU / Distributed Setup

If you encounter distributed environment issues, you can modify the `get_dist_launch` function in `run.py`. For example, for a 2-GPU setup:

```python
elif args.dist == 'f2':
        return "CUDA_VISIBLE_DEVICES=8,9 WORLD_SIZE=2 YOUR_OWN_PYTHON_PATH -W ignore -m torch.distributed.launch --master_port 25903 --nproc_per_node=2 " \
               "--nnodes=1 "
```

> Note: Remember to replace `CUDA_VISIBLE_DEVICES=8,9` with your own GPU IDs, and `YOUR_OWN_PYTHON_PATH` with your actual python executable path (e.g., `/root/miniconda3/bin/python`).

### Step 3: Start Training

Start training with:

```bash
python run.py --task 'itr_rsitmd_vit' --dist "f2" --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/full_rsitmd_vit'

python run.py --task 'itr_rsicd_vit' --dist "f2" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/full_rsicd_vit'
```

---

## Testing

To evaluate the model, set `if_evaluation` to `True` in the corresponding `configs/yaml` file, then run:

```bash
python run.py --task 'itr_rsitmd_vit' --dist "f2" --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/test' --checkpoint './checkpoints/MPS-CLIP/full_rsitmd_vit/checkpoint_best.pth' --evaluate

python run.py --task 'itr_rsicd_vit' --dist "f2" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/test' --checkpoint './checkpoints/MPS-CLIP/full_rsicd_vit/checkpoint_best.pth' --evaluate
```