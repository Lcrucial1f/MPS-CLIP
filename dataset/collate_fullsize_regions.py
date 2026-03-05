import torch

def fullsize_regions_collate_pack(batch):
    
    ids = [b["id"] for b in batch]
    images = torch.stack([b["image"] for b in batch], dim=0)

    flat_regions = []
    segments = [0]
    for b in batch:
        rlist = b["regions"]
        flat_regions.extend(rlist)
        segments.append(segments[-1] + len(rlist))

    if len(flat_regions) > 0:
        regions = torch.stack(flat_regions, dim=0)
    else:
       
        regions = torch.empty(0)

    bboxes = [b["bboxes"] for b in batch]

    return {
        "ids": ids,
        "images": images,      # [B, C, H, W]
        "regions": regions,    # [Nsum, C, H, W]
        "segments": segments,  
        "bboxes": bboxes
    }
