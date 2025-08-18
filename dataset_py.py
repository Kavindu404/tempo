import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pycocotools.mask as mask_utils
from typing import Dict, List, Tuple, Optional


class COCOSegmentationDataset(Dataset):
    """
    COCO-format dataset for segmentation tasks.
    Supports both instance and semantic segmentation.
    """
    
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transform: Optional[transforms.Compose] = None,
        target_size: int = 512,
        max_objects: int = 100,
        use_satellite_norm: bool = True
    ):
        """
        Args:
            img_dir: Path to image directory
            ann_file: Path to COCO annotation file
            transform: Optional transforms to apply
            target_size: Target image size for resizing
            max_objects: Maximum number of objects per image (for padding)
            use_satellite_norm: Use satellite imagery normalization
        """
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.target_size = target_size
        self.max_objects = max_objects
        self.use_satellite_norm = use_satellite_norm
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        self.num_classes = len(self.categories)
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Filter images that have annotations
        self.img_ids = [img_id for img_id in self.images.keys() 
                       if img_id in self.img_to_anns]
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default transforms based on satellite or natural imagery"""
        if self.use_satellite_norm:
            # Satellite imagery normalization (SAT-493M)
            normalize = transforms.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143)
            )
        else:
            # ImageNet normalization (LVD-1689M)
            normalize = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.target_size, self.target_size), 
                antialias=True
            ),
            normalize
        ])
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        anns = self.img_to_anns[img_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Process annotations
        boxes = []
        masks = []
        labels = []
        areas = []
        
        for ann in anns:
            # Skip if no segmentation
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            # Get bounding box (in COCO format: [x, y, w, h])
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to normalized coordinates relative to target size
            norm_x1 = x / orig_w
            norm_y1 = y / orig_h
            norm_x2 = (x + w) / orig_w
            norm_y2 = (y + h) / orig_h
            
            boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
            
            # Process segmentation mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                mask = self._polygon_to_mask(
                    ann['segmentation'], orig_h, orig_w
                )
            else:
                # RLE format
                mask = mask_utils.decode(ann['segmentation'])
            
            # Resize mask to target size
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_NEAREST
            )
            masks.append(mask_resized)
            
            labels.append(ann['category_id'])
            areas.append(ann['area'])
        
        # Pad to max_objects
        num_objects = len(boxes)
        
        if num_objects > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            masks = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in masks])
            labels = torch.tensor(labels, dtype=torch.long)
            areas = torch.tensor(areas, dtype=torch.float32)
        else:
            # Create dummy annotations if no objects
            boxes = torch.zeros((1, 4), dtype=torch.float32)
            masks = torch.zeros((1, self.target_size, self.target_size), dtype=torch.uint8)
            labels = torch.zeros(1, dtype=torch.long)
            areas = torch.zeros(1, dtype=torch.float32)
            num_objects = 1
        
        # Pad tensors
        if num_objects < self.max_objects:
            pad_size = self.max_objects - num_objects
            
            boxes = torch.cat([
                boxes,
                torch.zeros((pad_size, 4), dtype=torch.float32)
            ])
            
            masks = torch.cat([
                masks,
                torch.zeros((pad_size, self.target_size, self.target_size), dtype=torch.uint8)
            ])
            
            labels = torch.cat([
                labels,
                torch.zeros(pad_size, dtype=torch.long)
            ])
            
            areas = torch.cat([
                areas,
                torch.zeros(pad_size, dtype=torch.float32)
            ])
        
        # Truncate if too many objects
        elif num_objects > self.max_objects:
            boxes = boxes[:self.max_objects]
            masks = masks[:self.max_objects]
            labels = labels[:self.max_objects]
            areas = areas[:self.max_objects]
            num_objects = self.max_objects
        
        # Create validity mask (1 for real objects, 0 for padding)
        valid = torch.zeros(self.max_objects, dtype=torch.bool)
        valid[:num_objects] = True
        
        return {
            'image': image_tensor,
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'areas': areas,
            'valid': valid,
            'image_id': img_id,
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([self.target_size, self.target_size])
        }
    
    def _polygon_to_mask(self, polygons: List, height: int, width: int) -> np.ndarray:
        """Convert polygon segmentation to binary mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for polygon in polygons:
            polygon = np.array(polygon).reshape(-1, 2)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        
        return mask
    
    def get_category_names(self) -> List[str]:
        """Get list of category names"""
        return [self.categories[i]['name'] for i in sorted(self.categories.keys())]


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batch processing"""
    # Stack image tensors
    images = torch.stack([item['image'] for item in batch])
    
    # Stack other tensors
    batch_dict = {
        'images': images,
        'boxes': torch.stack([item['boxes'] for item in batch]),
        'masks': torch.stack([item['masks'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'areas': torch.stack([item['areas'] for item in batch]),
        'valid': torch.stack([item['valid'] for item in batch]),
        'image_ids': [item['image_id'] for item in batch],
        'orig_sizes': torch.stack([item['orig_size'] for item in batch]),
        'sizes': torch.stack([item['size'] for item in batch])
    }
    
    return batch_dict


def create_dataloaders(
    train_img_dir: str = "data/train/images",
    train_ann_file: str = "data/train/annotations.json",
    val_img_dir: str = "data/val/images", 
    val_ann_file: str = "data/val/annotations.json",
    batch_size: int = 8,
    num_workers: int = 4,
    target_size: int = 512,
    max_objects: int = 100,
    use_satellite_norm: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders"""
    
    train_dataset = COCOSegmentationDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        target_size=target_size,
        max_objects=max_objects,
        use_satellite_norm=use_satellite_norm
    )
    
    val_dataset = COCOSegmentationDataset(
        img_dir=val_img_dir,
        ann_file=val_ann_file,
        target_size=target_size,
        max_objects=max_objects,
        use_satellite_norm=use_satellite_norm
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
