#!/usr/bin/env python3
"""
Data preparation script for SAM 2.1 fine-tuning.
This script converts your dataset format to the format expected by SAM 2.1.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as mask_utils
import argparse
from tqdm import tqdm
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset for SAM 2.1 fine-tuning')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--annotation_file', type=str, default='annotations/640.json', help='Path to annotation file relative to dataset_dir')
    return parser.parse_args()

def polygon_to_mask(segmentation, height, width):
    """Convert polygon segmentation to binary mask."""
    if isinstance(segmentation, list) and len(segmentation) > 0:
        if isinstance(segmentation[0], list):  # Multiple polygons
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        else:  # Single polygon
            rle = mask_utils.frPyObjects([segmentation], height, width)[0]
        mask = mask_utils.decode(rle)
        return mask
    else:
        # Return an empty mask if segmentation is empty
        return np.zeros((height, width), dtype=np.uint8)

def prepare_sam_dataset(dataset_dir, output_dir, val_ratio, annotation_file):
    """Prepare dataset for SAM 2.1 fine-tuning."""
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    # Load annotations
    with open(os.path.join(dataset_dir, annotation_file), 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['images'])} images and {len(data['annotations'])} annotations")
    
    # Create image ID to file name mapping
    image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    image_id_to_size = {img['id']: (img['height'], img['width']) for img in data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Create category mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Determine train/val split
    all_image_ids = list(image_id_to_file.keys())
    np.random.shuffle(all_image_ids)
    val_size = int(len(all_image_ids) * val_ratio)
    train_ids = set(all_image_ids[val_size:])
    val_ids = set(all_image_ids[:val_size])
    
    # Process train and val sets
    for split, image_ids in [('train', train_ids), ('val', val_ids)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'masks'), exist_ok=True)
        
        # Create a metadata file for this split
        metadata = []
        
        for image_id in tqdm(image_ids, desc=f"Processing {split} set"):
            if image_id not in annotations_by_image:
                # Skip images without annotations
                continue
                
            # Get image file name and source path
            file_name = image_id_to_file[image_id]
            height, width = image_id_to_size[image_id]
            
            # Determine source image path based on whether it's in train or val in original dataset
            if os.path.exists(os.path.join(dataset_dir, 'images/train', file_name)):
                src_path = os.path.join(dataset_dir, 'images/train', file_name)
            elif os.path.exists(os.path.join(dataset_dir, 'images/val', file_name)):
                src_path = os.path.join(dataset_dir, 'images/val', file_name)
            else:
                print(f"Warning: Could not find image {file_name} for image_id {image_id}")
                continue
            
            # Copy the image
            dst_image_path = os.path.join(split_dir, 'images', file_name)
            shutil.copy(src_path, dst_image_path)
            
            # Process annotations for this image
            image_anns = annotations_by_image[image_id]
            
            for ann_idx, ann in enumerate(image_anns):
                category_id = ann['category_id']
                category_name = category_id_to_name[category_id]
                
                # Convert segmentation to mask
                seg_mask = polygon_to_mask(ann['segmentation'], height, width)
                
                # Save the mask
                mask_file = f"{os.path.splitext(file_name)[0]}_{ann_idx}.png"
                mask_path = os.path.join(split_dir, 'masks', mask_file)
                cv2.imwrite(mask_path, seg_mask * 255)
                
                # Add metadata entry
                metadata.append({
                    'image_path': os.path.join('images', file_name),
                    'mask_path': os.path.join('masks', mask_file),
                    'category_id': category_id,
                    'category_name': category_name,
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd']
                })
        
        # Save metadata
        with open(os.path.join(split_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Processed {len(metadata)} annotations in {split} set")

if __name__ == "__main__":
    args = parse_args()
    prepare_sam_dataset(args.dataset_dir, args.output_dir, args.val_ratio, args.annotation_file)
    print("Dataset preparation completed!")
