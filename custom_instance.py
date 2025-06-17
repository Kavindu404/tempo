# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Optional, Dict, List
from torch.utils.data import DataLoader
from torchvision import tv_tensors
import torch
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.custom_dataset import CustomDataset


class CustomInstance(LightningDataModule):
    def __init__(
        self,
        train_images_dir: Union[str, Path],
        train_annotations_file: Union[str, Path],
        val_images_dir: Union[str, Path],
        val_annotations_file: Union[str, Path],
        class_mapping: Optional[Dict[int, int]] = None,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 80,
        color_jitter_enabled: bool = False,
        scale_range: tuple[float, float] = (0.1, 2.0),
        check_empty_targets: bool = True,
        img_suffix: str = ".jpg",
    ) -> None:
        """
        Custom instance segmentation data module.
        
        Args:
            train_images_dir: Path to training images directory
            train_annotations_file: Path to training annotations JSON file (COCO format)
            val_images_dir: Path to validation images directory  
            val_annotations_file: Path to validation annotations JSON file (COCO format)
            class_mapping: Optional mapping from original class IDs to new class IDs
            num_workers: Number of data loading workers
            batch_size: Batch size
            img_size: Target image size (height, width)
            num_classes: Number of classes
            color_jitter_enabled: Whether to apply color jittering
            scale_range: Range for random scaling
            check_empty_targets: Whether to skip images with empty targets
            img_suffix: Image file extension
        """
        super().__init__(
            path=None,  # Not used for custom datasets
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        
        self.train_images_dir = Path(train_images_dir)
        self.train_annotations_file = Path(train_annotations_file)
        self.val_images_dir = Path(val_images_dir)
        self.val_annotations_file = Path(val_annotations_file)
        self.class_mapping = class_mapping or {}
        self.img_suffix = img_suffix
        
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(
        mask=None,
        annotations: List[Dict] = None,
        width: int = None,
        height: int = None,
        image_path: Path = None,
        class_mapping: Dict[int, int] = None,
        **kwargs
    ):
        """
        Parse annotations into masks, labels, and crowd flags.
        
        Args:
            mask: Not used for instance segmentation
            annotations: List of COCO-style annotations
            width: Image width
            height: Image height
            image_path: Path to image file
            class_mapping: Mapping from original to new class IDs
            
        Returns:
            tuple: (masks, labels, is_crowd)
        """
        if not annotations:
            return [], [], []
            
        masks, labels, is_crowd = [], [], []
        class_mapping = class_mapping or {}

        for annotation in annotations:
            cls_id = annotation.get('category_id')
            
            # Skip if class not in mapping (if mapping is provided)
            if class_mapping and cls_id not in class_mapping:
                continue
                
            # Map class ID if mapping is provided
            mapped_cls_id = class_mapping.get(cls_id, cls_id) if class_mapping else cls_id
            
            # Handle different segmentation formats
            segmentation = annotation.get('segmentation', [])
            
            if isinstance(segmentation, list) and len(segmentation) > 0:
                # Polygon format
                if isinstance(segmentation[0], list):
                    # Multiple polygons
                    rles = coco_mask.frPyObjects(segmentation, height, width)
                    rle = coco_mask.merge(rles)
                else:
                    # Single polygon
                    rles = coco_mask.frPyObjects([segmentation], height, width)
                    rle = rles[0]
                    
                mask_array = coco_mask.decode(rle)
                
            elif isinstance(segmentation, dict):
                # RLE format
                if 'counts' in segmentation and 'size' in segmentation:
                    mask_array = coco_mask.decode(segmentation)
                else:
                    continue
            else:
                # Try to use bbox if segmentation is not available
                bbox = annotation.get('bbox')
                if bbox:
                    x, y, w, h = bbox
                    mask_array = np.zeros((height, width), dtype=np.uint8)
                    mask_array[int(y):int(y+h), int(x):int(x+w)] = 1
                else:
                    continue
            
            # Convert to tensor
            mask_tensor = tv_tensors.Mask(mask_array, dtype=torch.bool)
            
            # Skip empty masks
            if mask_tensor.sum() == 0:
                continue
                
            masks.append(mask_tensor)
            labels.append(mapped_cls_id)
            is_crowd.append(bool(annotation.get('iscrowd', 0)))

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        """Setup train and validation datasets"""
        
        # Create target parser with class mapping
        def train_target_parser(**kwargs):
            return self.target_parser(class_mapping=self.class_mapping, **kwargs)
        
        def val_target_parser(**kwargs):
            return self.target_parser(class_mapping=self.class_mapping, **kwargs)
        
        # Training dataset
        self.train_dataset = CustomDataset(
            images_dir=self.train_images_dir,
            annotations_file=self.train_annotations_file,
            target_parser=train_target_parser,
            transforms=self.transforms,
            img_suffix=self.img_suffix,
            check_empty_targets=self.check_empty_targets,
        )
        
        # Validation dataset
        self.val_dataset = CustomDataset(
            images_dir=self.val_images_dir,
            annotations_file=self.val_annotations_file,
            target_parser=val_target_parser,
            transforms=None,  # No transforms for validation
            img_suffix=self.img_suffix,
            check_empty_targets=self.check_empty_targets,
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
    
    @classmethod
    def from_coco_dataset(
        cls,
        train_images_dir: Union[str, Path],
        train_annotations_file: Union[str, Path],
        val_images_dir: Union[str, Path],
        val_annotations_file: Union[str, Path],
        **kwargs
    ):
        """
        Create CustomInstance with COCO class mapping.
        
        This uses the same class mapping as the original COCOInstance.
        """
        coco_class_mapping = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
            11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
            20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
            31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
            39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
            48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
            56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
            64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
            76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
            85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
        }
        
        return cls(
            train_images_dir=train_images_dir,
            train_annotations_file=train_annotations_file,
            val_images_dir=val_images_dir,
            val_annotations_file=val_annotations_file,
            class_mapping=coco_class_mapping,
            **kwargs
        )


# Example usage and utility functions
def create_class_mapping_from_annotations(annotations_file: Union[str, Path]) -> Dict[int, int]:
    """
    Create a class mapping from annotations file.
    Maps original class IDs to sequential IDs starting from 0.
    
    Args:
        annotations_file: Path to COCO-style annotations file
        
    Returns:
        Dictionary mapping original class IDs to new sequential IDs
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Get unique category IDs from annotations
    category_ids = set()
    for annotation in data.get('annotations', []):
        category_ids.add(annotation['category_id'])
    
    # Create sequential mapping
    sorted_ids = sorted(category_ids)
    class_mapping = {original_id: new_id for new_id, original_id in enumerate(sorted_ids)}
    
    return class_mapping


def validate_dataset_structure(
    images_dir: Union[str, Path],
    annotations_file: Union[str, Path],
    img_suffix: str = ".jpg"
) -> Dict[str, any]:
    """
    Validate dataset structure and return statistics.
    
    Args:
        images_dir: Path to images directory
        annotations_file: Path to annotations file
        img_suffix: Image file extension
        
    Returns:
        Dictionary with validation results and statistics
    """
    images_dir = Path(images_dir)
    annotations_file = Path(annotations_file)
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Check if directories exist
    if not images_dir.exists():
        results["valid"] = False
        results["errors"].append(f"Images directory does not exist: {images_dir}")
        return results
    
    if not annotations_file.exists():
        results["valid"] = False
        results["errors"].append(f"Annotations file does not exist: {annotations_file}")
        return results
    
    # Load annotations
    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        results["valid"] = False
        results["errors"].append(f"Invalid JSON in annotations file: {e}")
        return results
    
    # Count images and annotations
    image_files = list(images_dir.glob(f"*{img_suffix}"))
    results["stats"]["total_images_in_dir"] = len(image_files)
    results["stats"]["total_annotations"] = len(data.get("annotations", []))
    results["stats"]["total_images_in_json"] = len(data.get("images", []))
    
    # Check for missing images
    if "images" in data:
        json_filenames = {img["file_name"] for img in data["images"]}
        dir_filenames = {img.name for img in image_files}
        
        missing_in_dir = json_filenames - dir_filenames
        missing_in_json = dir_filenames - json_filenames
        
        if missing_in_dir:
            results["warnings"].append(f"Images in JSON but not in directory: {len(missing_in_dir)}")
        if missing_in_json:
            results["warnings"].append(f"Images in directory but not in JSON: {len(missing_in_json)}")
    
    return results