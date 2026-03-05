import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageWithFullsizeRegionsDataset(Dataset):
    """
    Supports two region sources:

    - regions_are_fullsize=True:
    The index field "regions" contains full-resolution images (same size as the
    original image). Each region image keeps the foreground on a black background,
    so it can be loaded and used directly.

    - regions_are_fullsize=False:
    The index provides "crops" + "bboxes". In __getitem__, each crop is pasted
    back onto a full-resolution canvas according to its bounding box.

    Example index item:
    {
    "id": "xxx",
    "image": "/abs/path/img.jpg",
    "regions": ["/abs/path/region_0.png", ...],  # optional
    "crops":   ["/abs/path/crop_0.png", ...],    # optional
    "bboxes":  [[x0, y0, x1, y1], ...]           # aligned with crops
    }
    """
    def __init__(self, index_json, transform_image=None, transform_region=None,
                 regions_are_fullsize=True, fill_value=0):
        self.items = json.load(open(index_json, 'r', encoding='utf-8'))
        self.t_img = transform_image
        self.t_reg = transform_region
        self.regions_are_fullsize = regions_are_fullsize
        self.fill_value = fill_value  

    def __len__(self):
        return len(self.items)

    def _paste_crop_to_canvas(self, crop_img, bbox, canvas_size):
        # crop_img: PIL.Image (RGB); bbox: [x0,y0,x1,y1]; canvas_size: (W, H)
        from PIL import Image as PILImage
        W, H = canvas_size
        canvas = PILImage.new("RGB", (W, H), (self.fill_value, self.fill_value, self.fill_value))
        x0, y0, x1, y1 = [int(v) for v in bbox]
        
        cw, ch = crop_img.size
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        if (cw, ch) != (bw, bh):
            crop_img = crop_img.resize((bw, bh))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        canvas.paste(crop_img, (x0, y0))
        return canvas

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        W, H = img.size

        region_imgs = []
        if self.regions_are_fullsize:
            for p in it.get("regions", []):
                ri = Image.open(p).convert("RGB")
                region_imgs.append(ri)
        else:
            crops = it.get("crops", [])
            bboxes = it.get("bboxes", [])
            assert len(crops) == len(bboxes), "The number of crops and bboxes does not match."
            for p, box in zip(crops, bboxes):
                cimg = Image.open(p).convert("RGB")
                ri = self._paste_crop_to_canvas(cimg, box, (W, H))
                region_imgs.append(ri)

        img_t = self.t_img(img) if self.t_img else img
        regions_t = [self.t_reg(r) if self.t_reg else r for r in region_imgs]

        return {
            "id": it.get("id", str(i)),
            "image": img_t,         # Tensor(C,H',W') after CLIP-like transform
            "regions": regions_t,   # list[Tensor(C,H',W')]
            "bboxes": it.get("bboxes", [])
        }
 