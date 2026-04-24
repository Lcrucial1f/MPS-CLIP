<div align="center">

# [ICME2026] MPS-CLIP
**Multi-Perspective Subimage CLIP with Keyword Guidance for Remote Sensing Image-Text Retrieval**

[![arXiv](https://img.shields.io/badge/arXiv-2601.18190-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.18190)
[![Project Page](https://img.shields.io/badge/Project%20Page-MPS--CLIP-blue?style=flat-square)](https://lcrucial1f.github.io/)
[![HuggingFace Datasets](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-orange?style=flat-square)](https://huggingface.co/datasets/lcrucial1f/MPS-CLIP_Data/tree/main)

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#%EF%B8%8F-visualization">Results</a> •
  <a href="#%EF%B8%8F-installation">Installation</a> •
  <a href="#-data-preparation">Data</a> •
  <a href="#-training">Training</a> •
  <a href="#-testing--evaluation">Testing</a>
</p>

</div>

---

## 📖 Overview

**MPS-CLIP** introduces a novel approach for Remote Sensing Image-Text Retrieval (RSITR) by leveraging multi-perspective subimages and keyword guidance to enhance feature alignment.

### The Pipeline
<div align="center">
  <img src="assets/pipeline.png" alt="MPS-CLIP Pipeline" width="100%">
</div>

---

## 🖼️ Visualization

Qualitative retrieval results on RSITMD and RSICD datasets.

<div align="center">
  
| **Retrieval Example 2** | **Retrieval Example 2** |
|:---:|:---:|
| <img src="assets/visualization.png" width="100%"> | <img src="assets/visual_2.png" width="100%"> |

| **Retrieval Example 3** | **Retrieval Example 4** |
|:---:|:---:|
| <img src="assets/visual_3.png" width="100%"> | <img src="assets/visual_4.png" width="100%"> |

</div>

---

## 🛠️ Installation

Install dependencies:

```bash
pip install -r requirements.txt

```

---

## 📂 Data Preparation

We use RSITMD and RSICD datasets for all experiments.

**Download Data:**
Access the datasets via HuggingFace 🤗.
Annotation files are located in `data/finetune`.

**Configure Paths:**
Modify the `image_root` in the corresponding YAML config files (`configs/*.yaml`):

```yaml
# For RSICD
image_root: '/path/to/your/rsicd/images'
# For RSITMD
image_root: '/path/to/your/rsitmd/images'

```

---

## 🚀 Training

**1. Download Pretrained Weights**
We utilize GeoRSCLIP as the backbone.

* Download: `RS5M_ViT-B-32_RET-2.pt`
* Place in: `models/pretrain/`

**2. Distributed Training Setup (Optional)**
If you need to run on specific GPUs (e.g., 2 GPUs), modify the `get_dist_launch` function in `run.py`.

Example Configuration (2 GPUs):
Change `YOUR_OWN_PYTHON_PATH` to your python executable (e.g., `/root/miniconda3/bin/python`).

```python
elif args.dist == 'f2':
    return "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 YOUR_OWN_PYTHON_PATH -W ignore -m torch.distributed.launch --master_port 25903 --nproc_per_node=2 --nnodes=1 "

```

**3. Start Training**
Run the following commands to start training on the respective datasets:

RSITMD Dataset:

```bash
python run.py \
  --task 'itr_rsitmd_vit' \
  --dist "f2" \
  --config 'configs/Retrieval_rsitmd_vit.yaml' \
  --output_dir './checkpoints/MPS-CLIP/full_rsitmd_vit'

```

RSICD Dataset:

```bash
python run.py \
  --task 'itr_rsicd_vit' \
  --dist "f2" \
  --config 'configs/Retrieval_rsicd_vit.yaml' \
  --output_dir './checkpoints/MPS-CLIP/full_rsicd_vit'

```

---

## ⚡ Testing / Evaluation

To evaluate the model, ensure `if_evaluation: True` is set in the config file, or simply pass the `--evaluate` flag.

Evaluate on RSITMD:

```bash
python run.py \
  --task 'itr_rsitmd_vit' \
  --dist "f2" \
  --config 'configs/Retrieval_rsitmd_vit.yaml' \
  --output_dir './checkpoints/MPS-CLIP/test' \
  --checkpoint './checkpoints/MPS-CLIP/full_rsitmd_vit/checkpoint_best.pth' \
  --evaluate

```

Evaluate on RSICD:

```bash
python run.py \
  --task 'itr_rsicd_vit' \
  --dist "f2" \
  --config 'configs/Retrieval_rsicd_vit.yaml' \
  --output_dir './checkpoints/MPS-CLIP/test' \
  --checkpoint './checkpoints/MPS-CLIP/full_rsicd_vit/checkpoint_best.pth' \
  --evaluate

```

---

## 🖊️ Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex


```

```

```
