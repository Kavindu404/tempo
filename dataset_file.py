import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as mask_utils


class MaskBoxSyncTransform:
    """Wrapper to ensure masks and boxes stay synchronized during transforms"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, **kwargs):
        # Apply the transform
        result = self.transform(**kwargs)
        
        # Ensure masks and bboxes have the same count
        if 'bboxes' in result and 'masks' in result:
            num_boxes = len(result['bboxes'])
            num_masks = len(result['masks'])
            
            if num_boxes < num_masks:
                # More masks than boxes - keep only masks for existing boxes
                result['masks'] = result['masks'][:num_boxes]
            elif num_boxes > num_masks and num_masks > 0:
                # More boxes than masks - duplicate last mask
                while len(result['masks']) < num_boxes:
                    result['masks'].append(result['masks'][-1])
        
        return result


class InstanceSegmentationDataset(Dataset):
    def __init__(self, json_path, image_dir, transforms=None, is_train=True):
        self.image_dir = image_dir
        self.is_train = is_train
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        masks = []
        labels = []
        
        for ann in anns:
            # Only process annotations that have both bbox and segmentation
            if 'bbox' in ann and 'segmentation' in ann:
                x, y, w, h = ann['bbox']
                # Skip invalid boxes
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    
                    if isinstance(ann['segmentation'], dict):
                        rle = ann['segmentation']
                        mask = mask_utils.decode(rle)
                    elif isinstance(ann['segmentation'], list):
                        h, w = image.shape[:2]
                        rle = mask_utils.frPyObjects(ann['segmentation'], h, w)
                        mask = mask_utils.decode(mask_utils.merge(rle))
                    masks.append(mask)
                    labels.append(0)  # Single class
        
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            masks = np.array(masks, dtype=np.uint8)
            labels = np.array(labels, dtype=np.int64)
        
        if self.transforms:
            # Create a custom label for each mask to track them through augmentation
            instance_labels = list(range(len(labels)))
            
            transformed = self.transforms(
                image=image,
                masks=list(masks) if len(masks) > 0 else [],
                bboxes=list(boxes) if len(boxes) > 0 else [],
                labels=instance_labels if len(labels) > 0 else []
            )
            
            image = transformed['image']
            transformed_masks = transformed['masks']
            transformed_boxes = transformed['bboxes']
            transformed_instance_labels = transformed['labels']
            
            # Rebuild arrays using only kept instances
            if len(transformed_instance_labels) > 0:
                # Normalize to flat integer list
                kept_indices = [int(i) if not isinstance(i, (list, tuple)) else int(i[0]) 
                                for i in transformed_instance_labels]
                
                masks = np.array(transformed_masks)
                boxes = np.array(transformed_boxes)
                labels = labels[kept_indices] if len(labels) > 0 else np.zeros((0,), dtype=np.int64)
            else:
                masks = np.zeros((0, image.shape[-2], image.shape[-1]), dtype=np.uint8)
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'masks': torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in masks]) 
                      if len(masks) > 0 else torch.zeros((0, image.shape[-2], image.shape[-1]), dtype=torch.uint8),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.as_tensor([int(img_info['height']), int(img_info['width'])]),
            'size': torch.as_tensor([image.shape[-2], image.shape[-1]])
        }
        
        return image, target


def get_transforms(config, is_train=True):
    if is_train:
        transform = A.Compose([
            A.RandomResizedCrop(config.img_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
           additional_targets={'masks': 'masks'})
    else:
        transform = A.Compose([
            A.Resize(config.img_size[0], config.img_size[1]),
            A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
           additional_targets={'masks': 'masks'})
    
    return transform


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0], dim=0)
    return batch
