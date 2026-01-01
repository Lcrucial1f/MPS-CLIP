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
    训练集：
      ann_file: 列表，每个是 json 文件路径，其中字段包括
        - "image": "train/port_167.jpg"
        - "image_id": 6735
        - "caption": ...
        - "label": ...
        - "nouns": ["port", "buildings", "trees"]
      image_root: 原图根目录
      sub_root:   子图根目录，结构为:
          <sub_root>/port_167/<noun>_blackbg.png

    返回:
      image:      Tensor [C,H,W]
      regions:    list[Tensor [C,H,W]]  # 对应 nouns，每个 noun 一个子图
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
        从 ann['image'] = "train/port_167.jpg" 得到 "port_167"
        用于子图目录名 <sub_root>/port_167/
        """
        filename = os.path.basename(image_path)   # "port_167.jpg"
        base, _ = os.path.splitext(filename)      # "port_167"
        return base

    def _noun_to_filename(self, noun: str) -> str:
        """
        noun 映射到子图文件名: <noun>_blackbg.png
        """
        return f"{noun}_blackbg.png"

    def __getitem__(self, index):

        ann = self.ann[index]

        # 1. 读原图
        image_rel = ann['image']                          # "train/port_167.jpg"
        image_path = os.path.join(self.image_root, image_rel)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # 2. 文本和标签
        caption = pre_caption(ann['caption'], self.max_words)
        label = torch.tensor(ann['label'])

        # 3. 读子图
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
                    # 子图缺失就跳过
                    # print(f"[WARN] sub image not found: {sub_path}")
                    pass
        else:
            # 没有该原图对应的子图目录
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
    评估集（多 caption 版本）：
      ann_file: json，每条形如:
        {
          "image": "val/xxx.jpg",
          "caption": [...],        # 多条描述
          "nouns":   [...]         # 可选
        }
      image_root: 原图根目录
      sub_root:   子图根目录: <sub_root>/<base_name>/<noun>_blackbg.png

    提供给 evaluation 使用的属性：
      - self.text    : 所有 caption 扁平后的列表
      - self.img2txt : {img_id: [txt_id1, txt_id2, ...]}
      - self.txt2img : {txt_id: img_id}

    __getitem__ 当前返回: (image, index)
      与 evaluation 中的:
        for image, img_id in data_loader:
      完全对齐。子图 regions 先读不返回，如需在 eval 用再改 evaluation。
    """

    def __init__(self, ann_file, transform, image_root, sub_root,
                 max_words=30, region_transform=None):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.region_transform = region_transform if region_transform is not None else transform
        self.image_root = image_root
        self.sub_root = sub_root
        self.max_words = max_words

        # ========= 关键：兼容旧 evaluation 的结构 =========
        self.text = []     # 所有 caption
        self.image = []    # 每个图像的相对路径
        self.txt2img = {}  # 文本索引 -> 图像索引
        self.img2txt = {}  # 图像索引 -> [文本索引...]

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []

            # ann['caption'] 是一个 list
            for caption in ann['caption']:
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        # 和旧版保持一致：按图像数量算长度
        return len(self.image)

    # ======== 子图相关的工具函数（你原来有的） ========
    def _get_base_name(self, image_path: str) -> str:
        filename = os.path.basename(image_path)
        base, _ = os.path.splitext(filename)
        return base

    def _noun_to_filename(self, noun: str) -> str:
        return f"{noun}_blackbg.png"

    def __getitem__(self, index):
        """
        当前只返回 (image, index)，以兼容 evaluation 的:
          for image, img_id in data_loader:
        如果后续要在 eval 中融合 regions，再一起改 evaluation。
        """
        ann = self.ann[index]

        # 原图
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # 如果你现在暂时不在 eval 中用子图，这里不需要返回 regions。
        # 子图读取逻辑如果要保留，可以写在这里（先不返回）：
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
        
        return image, regions, index  # 如果 evaluation 改成能接 regions 再用这行