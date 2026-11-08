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

The repository includes the MPS-CLIP retrieval model and its customized
`open_clip` implementation. No source files need to be copied from another
project.

---

## 📂 Data Preparation

We use RSITMD and RSICD datasets for all experiments.

**Download Data:**
Download the images and the keyword-guided subimages from
[HuggingFace 🤗](https://huggingface.co/datasets/lcrucial1f/MPS-CLIP_Data) and
extract them into `data/`:

```bash
hf download lcrucial1f/MPS-CLIP_Data --repo-type dataset --local-dir data/
cd data
unzip rsicd.zip                       # -> rsicd/{train,val,test}
unzip rsitmd.zip                      # -> rsitmd/{train,val,test}
unzip rsicd_subimage.zip -d rsicd     # -> rsicd/{train_sub1,test_sub1}
unzip rsitmd_subimage.zip -d rsitmd   # -> rsitmd/{train_sub1,test_sub1}
```

The annotations containing extracted nouns use the `*_train1.json` and
`*_test1.json` names in `data/finetune` and are selected by the default configs.

**Configure Paths:**
The default configs use the layout above. If you place the data elsewhere,
update the four root fields in the corresponding YAML file:

```yaml
image_root: 'data/rsicd'
train_sub_root: 'data/rsicd/train_sub1'
val_sub_root: 'data/rsicd/test_sub1'
test_sub_root: 'data/rsicd/test_sub1'
```

Each subimage directory must follow this layout:

```text
train_sub1/<image_stem>/<noun>_blackbg.png
```

**Generate Subimages (optional):**
The released subimages were produced by the scripts in `images_segmentation/`.
You only need them to rebuild the subimages or to process a new dataset.

1. Extract nouns from the captions with DeepSeek:

   ```bash
   export DEEPSEEK_API_KEY=<your key>
   python images_segmentation/extracting_keywords.py \
       --input data/finetune/rsicd_train.json \
       --output data/finetune/rsicd_train1.json
   ```

   The LLM output is not deterministic, so a re-run gives slightly different
   nouns than the released `*1.json` files.

2. Segment one black-background subimage per (image, noun) with LangSAM
   (GroundingDINO + SAM ViT-H):

   ```bash
   python images_segmentation/launcher.py --gpus 0,1,2,3 \
       --json_path data/finetune/rsicd_train1.json \
       --image_root data/rsicd \
       --out_root data/rsicd/train_sub1
   ```

   `launcher.py` starts one `segmentation.py` process per GPU. Use
   `segmentation.py` with the same arguments to run on a single GPU.

Segmentation needs a separate environment with
[segment-geospatial](https://github.com/opengeos/segment-geospatial)
(`pip install segment-geospatial groundingdino-py`); it was tested with
segment-geospatial 1.0.1, torch 2.9.1, transformers 4.57.3 and
huggingface-hub 0.36.0 (transformers 4.x refuses to import with huggingface-hub 1.x).
The SAM and GroundingDINO weights (about 3.7 GB) are downloaded on the first run.

---

## 🚀 Training

**1. Download Pretrained Weights (GeoRSCLIP tasks only)**

* Download: `RS5M_ViT-B-32_RET-2.pt`
* Place in: `models/pretrain/`

The `*_vit` tasks use the OpenAI CLIP initialization. The `*_geo` tasks load
the GeoRSCLIP checkpoint configured by `pretrained_path`.

**2. Start Training**

`--dist fN` launches N processes on the GPUs visible through
`CUDA_VISIBLE_DEVICES`.

RSITMD Dataset:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --task 'itr_rsitmd_vit' \
  --dist "f2" \
  --output_dir './checkpoints/MPS-CLIP/full_rsitmd_vit'
```

RSICD Dataset:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --task 'itr_rsicd_vit' \
  --dist "f2" \
  --output_dir './checkpoints/MPS-CLIP/full_rsicd_vit'
```

---

## ⚡ Testing / Evaluation

Pass `--evaluate`; the launcher sets evaluation mode in the loaded config.

Evaluate on RSITMD:

```bash
python run.py \
  --task 'itr_rsitmd_vit' \
  --dist "gpu0" \
  --output_dir './checkpoints/MPS-CLIP/test' \
  --checkpoint './checkpoints/MPS-CLIP/full_rsitmd_vit/checkpoint_best.pth' \
  --evaluate
```

Evaluate on RSICD:

```bash
python run.py \
  --task 'itr_rsicd_vit' \
  --dist "gpu0" \
  --output_dir './checkpoints/MPS-CLIP/test' \
  --checkpoint './checkpoints/MPS-CLIP/full_rsicd_vit/checkpoint_best.pth' \
  --evaluate
```

---

## 🖊️ Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{li2026mpsclip,
  title={Multi-Perspective Subimage CLIP with Keyword Guidance for Remote Sensing Image-Text Retrieval},
  author={Li, Yifan and Wang, Shiying and Huang, Jianqiang},
  journal={arXiv preprint arXiv:2601.18190},
  year={2026}
}
```
