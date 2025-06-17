# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import json
import os
from pathlib import Path
from typing import Callable, Optional, Union
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
import torch
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


class CustomInstanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_folder_path: str,
        annotations_json_path: str,
        transforms: Optional[Callable] = None,
        class_mapping: Optional[dict] = None,
    ):
        self.img_folder_path = Path(img_folder_path)
        self.transforms = transforms
        self.class_mapping = class_mapping or {}
        
        # Load annotations
        with open(annotations_json_path, 'r') as f:
            self.annotation_data = json.load(f)
        
        # Create mappings
        self.image_id_to_file_name = {
            image["id"]: image["file_name"] for image in self.annotation_data["images"]
        }
        
        # Group annotations by image_id
        self.annotations_by_image_id = {}
        for annotation in self.annotation_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(annotation)
        
        # Filter out images without annotations
        self.image_ids = [
            img_id for img_id in self.image_id_to_file_name.keys()
            if img_id in self.annotations_by_image_id
        ]
        
        print(f"Loaded {len(self.image_ids)} images with annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        file_name = self.image_id_to_file_name[image_id]
        
        # Load image
        img_path = self.img_folder_path / file_name
        with open(img_path, 'rb') as f:
            img = tv_tensors.Image(Image.open(f).convert("RGB"))
        
        # Get annotations for this image
        annotations = self.annotations_by_image_id[image_id]
        
        # Parse annotations
        masks, labels, is_crowd = self._parse_annotations(annotations, img.shape[-2:])
        
        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)) if masks else tv_tensors.Mask(torch.empty((0, *img.shape[-2:]), dtype=torch.bool)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def _parse_annotations(self, annotations, img_shape):
        """Parse annotations to extract masks, labels, and crowd flags"""
        masks, labels, is_crowd = [], [], []
        height, width = img_shape
        
        for annotation in annotations:
            # Skip if category not in mapping (if mapping provided)
            category_id = annotation["category_id"]
            if self.class_mapping and category_id not in self.class_mapping:
                continue
            
            # Map category ID
            if self.class_mapping:
                mapped_label = self.class_mapping[category_id]
            else:
                # If no mapping provided, use category_id directly (assuming 0-indexed)
                mapped_label = category_id
            
            # Handle segmentation
            if "segmentation" in annotation:
                segmentation = annotation["segmentation"]
                
                # Handle RLE format
                if isinstance(segmentation, dict):
                    # Already in RLE format
                    rle = segmentation
                else:
                    # Convert polygon to RLE
                    rles = coco_mask.frPyObjects(segmentation, height, width)
                    rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
                
                # Decode mask
                mask = coco_mask.decode(rle)
                mask = torch.from_numpy(mask).bool()
                
                masks.append(mask)
                labels.append(mapped_label)
                is_crowd.append(bool(annotation.get("iscrowd", 0)))
        
        return masks, labels, is_crowd


class CustomInstance(LightningDataModule):
    def __init__(
        self,
        img_folder_path: str,
        train_annotations_json: str,
        val_annotations_json: str,
        class_mapping: Optional[dict] = None,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 80,
        color_jitter_enabled: bool = False,
        scale_range: tuple[float, float] = (0.1, 2.0),
        check_empty_targets: bool = True,
    ) -> None:
        super().__init__(
            path="",  # Not used for custom dataset
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        
        self.img_folder_path = img_folder_path
        self.train_annotations_json = train_annotations_json
        self.val_annotations_json = val_annotations_json
        self.class_mapping = class_mapping
        
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
    
    def setup(self, stage: Union[str, None] = None) -> "CustomInstance":
        self.train_dataset = CustomInstanceDataset(
            img_folder_path=self.img_folder_path,
            annotations_json_path=self.train_annotations_json,
            transforms=self.transforms,
            class_mapping=self.class_mapping,
        )
        
        self.val_dataset = CustomInstanceDataset(
            img_folder_path=self.img_folder_path,
            annotations_json_path=self.val_annotations_json,
            transforms=None,  # No augmentations for validation
            class_mapping=self.class_mapping,
        )
        
        return self
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )