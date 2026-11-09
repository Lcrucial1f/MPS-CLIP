import os
import json
from samgeo.text_sam import LangSAM
from PIL import Image
import numpy as np
import argparse


def merge_by_image(data_list):
    """One item per image with the union of its nouns, so each (image, noun) is segmented once."""
    merged = {}
    for item in data_list:
        nouns = merged.setdefault(item["image"], {})
        for noun in item["nouns"]:
            text_prompt = noun.strip()
            if text_prompt:
                nouns[text_prompt] = None
    return [{"image": image, "nouns": list(nouns)} for image, nouns in merged.items()]


def process_one_item(item, idx, total, rank, sam, image_root, out_root):
    rel_image_path = item["image"]
    image_path = os.path.join(image_root, rel_image_path)
    image_name = os.path.basename(rel_image_path)
    base_name = os.path.splitext(image_name)[0]
    out_dir = os.path.join(out_root, base_name)
    os.makedirs(out_dir, exist_ok=True)

    nouns = item["nouns"]
    prefix = f"[R{rank}] "

    print(f"\n{prefix}[{idx+1}/{total}] image: {image_path}, nouns: {nouns}", flush=True)

    for noun in nouns:
        text_prompt = noun.strip()
        if not text_prompt:
            continue
        print(f"{prefix}  -> segment noun: {text_prompt}", flush=True)

        try:
            sam.predict(
                image=image_path,
                text_prompt=text_prompt,
                box_threshold=0.1,
                text_threshold=0.24,
            )
        except Exception as e:
            print(f"{prefix}     failure, pass; The reason is: {e}", flush=True)
            continue

        orig = np.array(sam.image)
        mask = sam.prediction
        mask_bool = mask > 0

        h, w, _ = orig.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[mask_bool] = orig[mask_bool]

        safe_noun = text_prompt.replace(" ", "_")
        out_name = f"{safe_noun}_blackbg.png"
        out_path = os.path.join(out_dir, out_name)

        Image.fromarray(rgb).save(out_path)
        print(f"{prefix}     saved to: {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Annotation JSON with a 'nouns' field, e.g. data/finetune/rsicd_train1.json")
    parser.add_argument("--image_root", required=True, help="Root that the 'image' paths are relative to, e.g. data/rsicd")
    parser.add_argument("--out_root", required=True, help="Output subimage root, e.g. data/rsicd/train_sub1")
    parser.add_argument("--rank", type=int, default=0, help="Process rank (index of the current process/GPU)")
    parser.add_argument("--world_size", type=int, default=1, help="World size (total number of processes/GPUs).")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = merge_by_image(json.load(f))

    N = len(data_list)
    rank = args.rank
    world_size = args.world_size

    indices = list(range(N))
    per = (N + world_size - 1) // world_size
    start = rank * per
    end = min(start + per, N)
    my_indices = indices[start:end]

    shard = [data_list[i] for i in my_indices]
    total = len(shard)

    print(f"[R{rank}] images={N}, start={start}, end={end}, shard_len={total}", flush=True)
    print(f"[R{rank}] first 3 indices: {my_indices[:3]}", flush=True)
    print(f"[R{rank}] first 3 images: {[item['image'] for item in shard[:3]]}", flush=True)
    sam = LangSAM()

    for idx, item in enumerate(shard):
        process_one_item(item, idx, total, rank, sam, args.image_root, args.out_root)
