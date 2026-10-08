import os
import json

from torch.utils.data import Dataset
from PIL import Image

from data.utils import pre_caption


def build_flickr8k_gt_json(ann_root, split):
    """Build COCO-format GT JSON from Flickr8k token file (cached after first run)."""
    split_files = {'val': 'Flickr_8k.devImages.txt', 'test': 'Flickr_8k.testImages.txt'}
    out_file = os.path.join(ann_root, f'flickr8k_{split}_gt.json')

    if os.path.exists(out_file):
        return out_file

    with open(os.path.join(ann_root, split_files[split])) as f:
        split_images = sorted(set(f.read().strip().split('\n')))

    img_id_map = {name: i for i, name in enumerate(split_images)}
    images = [{'id': i} for i in range(len(split_images))]
    annotations = []
    ann_id = 0

    with open(os.path.join(ann_root, 'Flickr8k.token.txt')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            img_cap, caption = parts
            img_name = img_cap.split('#')[0]
            if img_name not in img_id_map:
                continue
            annotations.append({
                'image_id': img_id_map[img_name],
                'id': ann_id,
                'caption': caption,
            })
            ann_id += 1

    with open(out_file, 'w') as f:
        json.dump({'images': images, 'annotations': annotations}, f)

    print(f'Flickr8k GT JSON saved to {out_file}')
    return out_file


class flickr8k_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):
        with open(os.path.join(ann_root, 'Flickr_8k.trainImages.txt')) as f:
            train_images = set(f.read().strip().split('\n'))

        self.annotation = []
        with open(os.path.join(ann_root, 'Flickr8k.token.txt')) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                img_cap, caption = parts
                img_name = img_cap.split('#')[0]
                if img_name in train_images:
                    self.annotation.append({'image': img_name, 'caption': caption})

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
        image = self.transform(image)
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        return image, caption, index


class flickr8k_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):
        split_files = {'val': 'Flickr_8k.devImages.txt', 'test': 'Flickr_8k.testImages.txt'}

        with open(os.path.join(ann_root, split_files[split])) as f:
            self.images = sorted(set(f.read().strip().split('\n')))

        self.img_id_map = {name: i for i, name in enumerate(self.images)}
        self.image_root = image_root
        self.transform = transform

        build_flickr8k_gt_json(ann_root, split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        image = Image.open(os.path.join(self.image_root, img_name)).convert('RGB')
        return self.transform(image), self.img_id_map[img_name]
