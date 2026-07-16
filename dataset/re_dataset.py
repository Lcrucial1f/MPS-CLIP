import json
import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from dataset.utils import pre_caption


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


class RegionImageMixin:
    def _load_image(self, relative_path):
        image_path = os.path.join(self.image_root, relative_path)
        with Image.open(image_path) as image:
            return self.transform(image.convert('RGB'))

    def _load_regions(self, relative_path, nouns):
        base_name = os.path.splitext(os.path.basename(relative_path))[0]
        sub_dir = os.path.join(self.sub_root, base_name)
        if not os.path.isdir(sub_dir):
            return []

        regions = []
        for noun in dict.fromkeys(nouns):
            sub_path = os.path.join(sub_dir, f'{noun}_blackbg.png')
            if not os.path.isfile(sub_path):
                continue
            with Image.open(sub_path) as image:
                regions.append(self.region_transform(image.convert('RGB')))
        return regions


class re_train_dataset(RegionImageMixin, Dataset):
    """RSITR training pairs with keyword-guided subimages."""

    def __init__(self, ann_file, transform, image_root, sub_root,
                 max_words=30, region_transform=None):
        self.ann = []
        for path in ann_file:
            self.ann.extend(_read_json(path))

        self.transform = transform
        self.region_transform = region_transform or transform
        self.image_root = image_root
        self.sub_root = sub_root
        self.max_words = max_words

        self.img_ids = {}
        for annotation in self.ann:
            image_id = annotation['image_id']
            if image_id not in self.img_ids:
                self.img_ids[image_id] = len(self.img_ids)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        annotation = self.ann[index]
        image = self._load_image(annotation['image'])
        regions = self._load_regions(annotation['image'], annotation.get('nouns', []))
        caption = pre_caption(annotation['caption'], self.max_words)
        label = torch.tensor(annotation['label'], dtype=torch.long)
        image_index = self.img_ids[annotation['image_id']]
        return image, regions, caption, image_index, label


class re_eval_dataset(RegionImageMixin, Dataset):
    """RSITR evaluation images, captions, and keyword-guided subimages."""

    def __init__(self, ann_file, transform, image_root, sub_root,
                 max_words=30, region_transform=None):
        self.ann = _read_json(ann_file)
        self.transform = transform
        self.region_transform = region_transform or transform
        self.image_root = image_root
        self.sub_root = sub_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        text_index = 0
        for image_index, annotation in enumerate(self.ann):
            self.image.append(annotation['image'])
            self.img2txt[image_index] = []
            for caption in annotation['caption']:
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[image_index].append(text_index)
                self.txt2img[text_index] = image_index
                text_index += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        annotation = self.ann[index]
        image = self._load_image(annotation['image'])
        regions = self._load_regions(annotation['image'], annotation.get('nouns', []))
        return image, regions, index
