import numpy as np
import pandas as pd
import os
import json
import cv2

from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .util_dict import coco_label_inverse, int2classes

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

def get_coco(coco_root, dataType):
    annotationFile = os.path.join(coco_root, f'annotations/instances_{dataType}.json')
    return COCO(annotationFile)

def get_sampled_file(sample_file):
    return pd.read_csv(sample_file)


def onehot_encoding(num_classes, class_id):
    onehot = [0 for _ in range(num_classes)]
    onehot[class_id] = 1
    return onehot

def get_train_transforms(resize=512):
    return A.Compose(
        [
            #A.RandomSizedCrop(min_max_height=(400, 400), height=resize, width=resize, p=0.5),
            A.ShiftScaleRotate(scale_limit=0.8, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
            ],p=0.9),
            #A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=resize, width=resize, p=1),
            #A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms(resize=512):
    return A.Compose(
        [
            A.Resize(height=resize, width=resize, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

class COCODataSet(Dataset):
    def __init__(self, coco_root, dataType, percentage_file=None, num_classes=80, percentage_sampled=True, transforms=None):
        self.coco_root = coco_root
        self.dataType = dataType
        self.num_classes = num_classes
        self.percentage_sampled = percentage_sampled
        self.coco = get_coco(coco_root, dataType)
        if percentage_sampled:
            self.image_ids = get_sampled_file(percentage_file)
        else:
            self.image_ids = self.coco.getImgIds()

        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        img, box, classes, width, height = self.load_images_and_boxes(index)
        target = {}
        target['boxes'] = box
        target['labels'] = torch.tensor(classes)

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': img,
                    'bboxes': target['boxes'],
                    'labels': target['labels']
                })
                if len(sample['bboxes']) > 0:
                       image = sample['image']
                       target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                       break
        return image, target, width, height

    def load_images_and_boxes(self, index):
        image_file = self.coco.loadImgs(self.image_ids[index])[0]
        image_name = image_file['file_name']
        image_width = image_file['width']
        image_height = image_file['height']
        image_path = os.path.join(self.coco_root, self.dataType, image_name)
        im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)
        im = im / 255.0

        ann_ids = self.coco.getAnnIds(self.image_ids[index])
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        classes = []
        if len(anns) != 0:
            for ann in anns:
                bboxes.append(ann['bbox'])
                classes.append(onehot_encoding(self.num_classes, coco_label_inverse[ann['category_id']]))
            bboxes = np.array(bboxes)
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
            classes = np.array(classes)
        return im, bboxes, classes, image_width, image_height

train_dataset = COCODataSet(coco_root=config['coco_root'],
                            dataType=config['train_data'],
                            percentage_file=config['percentage_file'],
                            num_classes=config['num_classes'],
                            percentage_sampled=True,
                            transforms=get_train_transforms())

valid_dataset = COCODataSet(coco_root=config['coco_root'],
                            dataType=config['valid_data'],
                            num_classes=config['num_classes'],
                            percentage_file=False,
                            transforms=get_valid_transforms())




