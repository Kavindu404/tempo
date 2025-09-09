import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pycocotools.mask as mask_util
from typing import Dict, List, Tuple

class InstanceSegmentationDataset(Dataset):
    def __init__(self, json_path: str, image_dir: str, transforms=None, is_train: bool = True):
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.image_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_anns:
                self.image_id_to_anns[image_id] = []
            self.image_id_to_anns[image_id].append(ann)
        
        # Filter images that have annotations
        self.images = [img for img in self.coco_data['images'] 
                      if img['id'] in self.image_id_to_anns]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        image_info = self.images[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations for this image
        annotations = self.image_id_to_anns.get(image_id, [])
        
        # Prepare masks and labels
        masks = []
        labels = []
        bboxes = []
        areas = []
        
        for ann in annotations:
            # Convert segmentation to mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                from pycocotools import mask as mask_utils
                rle = mask_utils.frPyObjects(ann['segmentation'], 
                                           image_info['height'], 
                                           image_info['width'])
                mask = mask_utils.decode(rle)
                if len(mask.shape) == 3:
                    mask = mask.sum(axis=2) > 0
            else:
                # RLE format
                mask = mask_util.decode(ann['segmentation'])
            
            masks.append(mask.astype(np.uint8))
            labels.append(1)  # Single class, always 1
            bboxes.append(ann['bbox'])  # [x, y, w, h]
            areas.append(ann['area'])
        
        # Convert to proper format
        if len(masks) == 0:
            # Handle images with no annotations
            masks = np.zeros((0, image_info['height'], image_info['width']), dtype=np.uint8)
            labels = []
            bboxes = []
            areas = []
        else:
            masks = np.stack(masks, axis=0)
        
        # Apply transformations
        if self.transforms:
            # Prepare masks for albumentations (expects list of 2D arrays)
            mask_list = [masks[i] for i in range(masks.shape[0])] if masks.shape[0] > 0 else []
            
            transformed = self.transforms(
                image=image,
                masks=mask_list,
                bboxes=[(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in bboxes],
                category_ids=labels
            )
            
            image = transformed['image']
            mask_list = transformed['masks']
            transformed_bboxes = transformed['bboxes']
            labels = transformed['category_ids']
            
            # Convert back to numpy array
            if len(mask_list) > 0:
                masks = np.stack(mask_list, axis=0)
                # Convert bboxes back to [x, y, w, h] format
                bboxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] 
                         for box in transformed_bboxes]
            else:
                masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)
                bboxes = []
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Prepare target dictionary
        target = {
            'masks': torch.from_numpy(masks).float() if masks.shape[0] > 0 else torch.zeros((0, masks.shape[1], masks.shape[2])),
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(bboxes, dtype=torch.float32) if len(bboxes) > 0 else torch.zeros((0, 4)),
            'image_id': torch.tensor(image_id),
            'area': torch.tensor(areas, dtype=torch.float32) if len(areas) > 0 else torch.zeros((0,)),
            'iscrowd': torch.zeros(len(labels), dtype=torch.uint8),
            'orig_size': torch.tensor([image_info['height'], image_info['width']]),
            'size': torch.tensor([image.shape[1], image.shape[2]])
        }
        
        return image, target

def get_transforms(config, is_train=True):
    if is_train:
        transform = A.Compose([
            A.Resize(config.image_size[0], config.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    else:
        transform = A.Compose([
            A.Resize(config.image_size[0], config.image_size[1]),
            A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    
    return transform

def collate_fn(batch):
    """Custom collate function to handle variable number of instances"""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
