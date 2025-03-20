"""
Modified from RT-DETR inference script to run inference on COCO dataset
and save results in COCO format
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T
import pycocotools.coco as coco
import time
from tqdm import tqdm
import json
import numpy as np 
from PIL import Image, ImageDraw
import datetime
import multiprocessing
from functools import partial
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig


class CocoInferenceModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        
    def forward(self, images, orig_target_sizes):
        input_sizes = torch.tensor([[images.shape[-1], images.shape[-2]]], device=images.device)
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes, input_sizes)
        return outputs


def preprocess_image(img_path, default_height, default_width, device):
    """Preprocess image for inference"""
    im_pil = Image.open(img_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    scaleX = default_width / w
    scaleY = default_height / h

    scale = scaleX if scaleX < scaleY else scaleY

    new_H = int(scale*h)
    new_W = int(scale*w)

    val_h = (default_height - new_H)//2
    val_w = (default_width - new_W)//2

    transforms = T.Compose([
        T.Resize((new_H, new_W)),
        T.Pad(padding=(val_w, val_h, val_w, val_h)),
        T.ToTensor(),
    ])

    im_data = transforms(im_pil)[None].to(device)
    return im_data, orig_size, im_pil


def convert_to_coco_annotation(labels, boxes, coords, scores, image_id, threshold=0.5):
    """Convert model output to COCO annotation format"""
    annotations = []
    scr = scores[0]
    mask = scr > threshold
    
    lab = labels[0][mask].cpu().numpy()
    box = boxes[0][mask].cpu().numpy()
    scrs = scores[0][mask].cpu().numpy()
    coord = coords[0][mask].cpu().numpy()
    
    for i, (label, bbox, score, polygon) in enumerate(zip(lab, box, scrs, coord)):
        x1, y1, x2, y2 = bbox
        
        # Convert bbox format from [x1, y1, x2, y2] to [x, y, width, height]
        width = x2 - x1
        height = y2 - y1
        
        # Convert polygon coordinates to COCO segmentation format
        segmentation = polygon.reshape(-1).tolist()
        
        annotation = {
            'id': len(annotations) + 1,  # We'll update this later
            'image_id': image_id,
            'category_id': int(label),
            'bbox': [float(x1), float(y1), float(width), float(height)],
            'segmentation': [segmentation],
            'area': float(width * height),
            'iscrowd': 0,
            'score': float(score)
        }
        
        annotations.append(annotation)
    
    return annotations


def process_batch(model, img_batch, img_ids, default_height, default_width, device, threshold=0.5):
    """Process a batch of images"""
    batch_results = []
    
    for img_path, img_id in zip(img_batch, img_ids):
        img_data, orig_size, _ = preprocess_image(img_path, default_height, default_width, device)
        outputs = model(img_data, orig_size)
        labels, boxes, coords, scores = outputs
        
        annotations = convert_to_coco_annotation(labels, boxes, coords, scores, img_id, threshold)
        batch_results.extend(annotations)
    
    return batch_results


def main(args):
    """Main function to run inference on COCO dataset and save results"""
    # Load configuration
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load model
    cfg.model.load_state_dict(state)
    default_height, default_width = cfg.model.encoder.eval_spatial_size

    # Create model
    model = CocoInferenceModel(
        cfg.model.deploy(),
        cfg.postprocessor.deploy()
    ).to(args.device)
    model.eval()

    # Load COCO dataset
    coco_api = coco.COCO(args.input_json)
    image_ids = coco_api.getImgIds()
    
    # Get image paths
    images = coco_api.loadImgs(image_ids)
    img_paths = [os.path.join(args.img_dir, img['file_name']) for img in images]
    
    # Create output COCO format
    coco_output = {
        'images': images,
        'annotations': [],
        'categories': coco_api.loadCats(coco_api.getCatIds()),
        'info': {
            'description': 'Inference results',
            'version': '1.0',
            'year': datetime.datetime.now().year,
            'contributor': 'RT-DETR inference script',
            'date_created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    
    # Determine batch size based on available resources
    batch_size = args.batch_size if args.batch_size > 0 else 4  # Default batch size
    
    # Process images with batching and tqdm progress bar
    all_annotations = []
    annotation_id = 1
    
    print(f"Running inference on {len(img_paths)} images with batch size {batch_size}...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Processing images"):
            batch_img_paths = img_paths[i:i+batch_size]
            batch_img_ids = image_ids[i:i+batch_size]
            
            batch_results = []
            for j, (img_path, img_id) in enumerate(zip(batch_img_paths, batch_img_ids)):
                img_data, orig_size, _ = preprocess_image(img_path, default_height, default_width, args.device)
                outputs = model(img_data, orig_size)
                labels, boxes, coords, scores = outputs
                
                annotations = convert_to_coco_annotation(labels, boxes, coords, scores, img_id, args.threshold)
                
                # Update annotation IDs
                for ann in annotations:
                    ann['id'] = annotation_id
                    annotation_id += 1
                
                batch_results.extend(annotations)
            
            all_annotations.extend(batch_results)
            
            # Optionally save intermediate results
            if args.save_interval > 0 and (i + batch_size) % args.save_interval == 0:
                temp_output = coco_output.copy()
                temp_output['annotations'] = all_annotations
                with open(f"{args.output_json}.temp", 'w') as f:
                    json.dump(temp_output, f)
    
    # Add all annotations to the output
    coco_output['annotations'] = all_annotations
    
    # Save final results
    with open(args.output_json, 'w') as f:
        json.dump(coco_output, f)
    
    print(f"Inference complete. Results saved to {args.output_json}")
    print(f"Total annotations: {len(all_annotations)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run inference on COCO dataset and save results in COCO format')
    parser.add_argument('-c', '--config', type=str, help='Model configuration file')
    parser.add_argument('-r', '--resume', type=str, help='Checkpoint file to resume from')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--input_json', type=str, required=True, help='Input COCO JSON file')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_json', type=str, required=True, help='Output COCO JSON file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--save_interval', type=int, default=100, help='Save intermediate results every N batches (0 to disable)')
    
    args = parser.parse_args()
    main(args)
