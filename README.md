```markdown
# MPS-CLIP







### Environments 

Set up the environment by running:

```bash
pip install -r requirements.txt
```

### Datasets 

All experiments are based on the **RSITMD** and **RSICD** datasets:

- [RSITMD](https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD)
- [RSICD](https://github.com/201528014227051/RSICD_optimal)



Then modify the corresponding `configs/yaml` file:

```yaml
image_root: './images/datasets_name/'
```

The annotation files for the datasets are located in the `data/finetune` directory.

### Training 

Download the **GeoRSCLIP** pre-trained model from:

- https://huggingface.co/Zilun/GeoRSCLIP/blob/main/ckpt/RS5M_ViT-B-32_RET-2.pt

Place the checkpoint in:

```text
models/pretrain/
```

If you encounter distributed environment issues, you can modify the `get_dist_launch` function in `run.py`. For example, for a 2-GPU setup:

```python
elif args.dist == 'f2':
        return "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 /root/miniconda3/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 " \
               "--nnodes=1 "
```

Start training with:

```bash
python run.py --task 'itr_rsitmd_vit' --dist "f2" --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/full_rsitmd_vit'

python run.py --task 'itr_rsicd_vit' --dist "f2" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/full_rsicd_vit'

```

### Testing 

To evaluate the model, set `if_evaluation` to `True` in the corresponding `configs/yaml` file, then run:

```bash
python run.py --task 'itr_rsitmd_vit' --dist "f2" --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/test' --checkpoint './checkpoints/MPS-CLIP/full_rsitmd_vit/checkpoint_best.pth' --evaluate

python run.py --task 'itr_rsicd_vit' --dist "f2" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/MPS-CLIP/test' --checkpoint './checkpoints/MPS-CLIP/full_rsicd_vit/checkpoint_best.pth' --evaluate

