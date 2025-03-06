import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import shutil

def resize_image(img, target_size=(224, 224)):
    """Resize an image to target size while maintaining aspect ratio with padding"""
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize the image
    resized_img = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of target size
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Calculate position to paste the resized image (center it)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    
    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return canvas, (x_offset, y_offset, scale)

def update_annotation(ann, transform_params):
    """Update annotation coordinates based on resizing transform parameters"""
    x_offset, y_offset, scale = transform_params
    
    # Update segmentation coordinates
    new_segmentation = []
    for segm in ann['segmentation']:
        new_seg = []
        for i in range(0, len(segm), 2):
            if i+1 < len(segm):
                x = segm[i] * scale + x_offset
                y = segm[i+1] * scale + y_offset
                new_seg.extend([x, y])
        new_segmentation.append(new_seg)
    
    # Update bbox
    x, y, w, h = ann['bbox']
    new_bbox = [
        x * scale + x_offset,
        y * scale + y_offset,
        w * scale,
        h * scale
    ]
    
    # Create updated annotation
    updated_ann = ann.copy()
    updated_ann['segmentation'] = new_segmentation
    updated_ann['bbox'] = new_bbox
    
    return updated_ann

def main():
    parser = argparse.ArgumentParser(description='Preprocess custom dataset to 224x224')
    parser.add_argument('--input_dir', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dataset directory')
    parser.add_argument('--annotation_file', type=str, required=True, help='COCO annotation file path')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split to process')
    args = parser.parse_args()
    
    # Setup paths
    input_image_dir = os.path.join(args.input_dir, 'images', args.split)
    output_dir = args.output_dir
    output_image_dir = os.path.join(output_dir, 'images', args.split)
    
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    # Load COCO annotations
    coco = COCO(args.annotation_file)
    img_ids = coco.getImgIds()
    
    # Create new annotation file
    new_annotations = {
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories']
    }
    
    print(f"Processing {len(img_ids)} images from {args.split} split...")
    
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img_path = os.path.join(input_image_dir, img_info['file_name'])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        
        # Resize image
        resized_img, transform_params = resize_image(img)
        
        # Save resized image
        output_img_path = os.path.join(output_image_dir, img_info['file_name'])
        cv2.imwrite(output_img_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        
        # Update image info
        new_img_info = img_info.copy()
        new_img_info['width'] = 224
        new_img_info['height'] = 224
        new_annotations['images'].append(new_img_info)
        
        # Update annotations
        for ann in anns:
            updated_ann = update_annotation(ann, transform_params)
            new_annotations['annotations'].append(updated_ann)
    
    # Save new annotation file
    output_annotation_path = os.path.join(output_dir, 'annotations', os.path.basename(args.annotation_file))
    with open(output_annotation_path, 'w') as f:
        json.dump(new_annotations, f)
    
    print(f"Preprocessing complete. Resized images and updated annotations saved to {output_dir}")

if __name__ == "__main__":
    main()
