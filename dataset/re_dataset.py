import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    """
    Training set:
    ann_file: a list of JSON file paths. Each JSON item contains fields such as:
        - "image": "train/port_167.jpg"
        - "image_id": 6735
        - "caption": ...
        - "label": ...
        - "nouns": ["port", "buildings", "trees"]
    image_root: root directory of the original images
    sub_root:   root directory of the region (sub-image) files, organized as:
        <sub_root>/port_167/<noun>_blackbg.png

    Returns:
    image:      Tensor [C, H, W]
    regions:    list[Tensor [C, H, W]]  # corresponds to nouns; one region per noun
    caption:    str
    img_id_idx: int
    label:      LongTensor(1)
    """
    def __init__(self, ann_file, transform, image_root, sub_root,
                 max_words=30, region_transform=None):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.region_transform = region_transform if region_transform is not None else transform
        self.image_root = image_root
        self.sub_root = sub_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids:
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def _get_base_name(self, image_path: str) -> str:
        """
        Extract "port_167" from ann["image"] = "train/port_167.jpg".
        This is used as the subdirectory name for region images: <sub_root>/port_167/
        """
        filename = os.path.basename(image_path)   # "port_167.jpg"
        base, _ = os.path.splitext(filename)      # "port_167"
        return base

    def _noun_to_filename(self, noun: str) -> str:
        """
        Map each noun to its region image filename: <noun>_blackbg.png
        """
        return f"{noun}_blackbg.png"

    def __getitem__(self, index):

        ann = self.ann[index]

        
        image_rel = ann['image']                          # "train/port_167.jpg"
        image_path = os.path.join(self.image_root, image_rel)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

       
        caption = pre_caption(ann['caption'], self.max_words)
        label = torch.tensor(ann['label'])

        
        nouns = ann.get('nouns', [])                      # 例如 ["port","buildings","trees"]
        regions = []

        base_name = self._get_base_name(image_rel)        # "port_167"
        sub_dir = os.path.join(self.sub_root, base_name)  # "<sub_root>/port_167/"

        if os.path.isdir(sub_dir):
            for noun in nouns:
                fname = self._noun_to_filename(noun)      # "port_blackbg.png" 等
                sub_path = os.path.join(sub_dir, fname)
                if os.path.exists(sub_path):
                    rimg = Image.open(sub_path).convert('RGB')
                    rimg = self.region_transform(rimg)
                    regions.append(rimg)
                else:
                    
                    
                    pass
        else:
           
            # print(f"[WARN] sub dir not found: {sub_dir}")
            pass

        return image, regions, caption, self.img_ids[ann['image_id']], label
import json
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_eval_dataset(Dataset):
    """
    Evaluation set (multi-caption version):
    ann_file: a JSON file where each entry looks like:
        {
        "image": "val/xxx.jpg",
        "caption": [...],        # multiple captions
        "nouns":   [...]         # optional
        }
    image_root: root directory of the original images
    sub_root:   root directory of the region (sub-image) files:
        <sub_root>/<base_name>/<noun>_blackbg.png

    Attributes provided for evaluation:
    - self.text    : a flattened list of all captions
    - self.img2txt : {img_id: [txt_id1, txt_id2, ...]}
    - self.txt2img : {txt_id: img_id}

    __getitem__ currently returns: (image, index)
    This matches the evaluation loop:
        for image, img_id in data_loader:
    exactly. Region images (regions) are loaded but not returned for now; if you
    need them during evaluation, modify the evaluation code accordingly.
    """

    def __init__(self, ann_file, transform, image_root, sub_root,
                 max_words=30, region_transform=None):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.region_transform = region_transform if region_transform is not None else transform
        self.image_root = image_root
        self.sub_root = sub_root
        self.max_words = max_words

        
        self.text = []     
        self.image = []    
        self.txt2img = {}  
        self.img2txt = {}  

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []

            
            for caption in ann['caption']:
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        
        return len(self.image)

    
    def _get_base_name(self, image_path: str) -> str:
        filename = os.path.basename(image_path)
        base, _ = os.path.splitext(filename)
        return base

    def _noun_to_filename(self, noun: str) -> str:
        return f"{noun}_blackbg.png"

    def __getitem__(self, index):
        """
        Currently, it only returns (image, index) to stay compatible with the evaluation loop:
        for image, img_id in data_loader:
        If later you want to fuse regions during evaluation, update the evaluation code accordingly.
        """
        ann = self.ann[index]

        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        
        nouns = ann.get('nouns', [])
        regions = []
        base_name = self._get_base_name(ann['image'])
        sub_dir = os.path.join(self.sub_root, base_name)
        if os.path.isdir(sub_dir):
            for noun in nouns:
                fname = self._noun_to_filename(noun)
                sub_path = os.path.join(sub_dir, fname)
                if os.path.exists(sub_path):
                    rimg = Image.open(sub_path).convert('RGB')
                    rimg = self.region_transform(rimg)
                    regions.append(rimg)
        
        return image, regions, index 
