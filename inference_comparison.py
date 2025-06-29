#!/usr/bin/env python3
"""
Model Inference Comparison Script
Compares EoMT and YOLOv8 instance segmentation models on a folder of images.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import yaml
from ultralytics import YOLO
from lightning.pytorch import LightningModule
from jsonargparse import ArgumentParser
import torchvision.transforms.v2.functional as F

# Import your model classes
from training.mask_classification_instance import MaskClassificationInstance
from models.eomt import EoMT
from models.vit import ViT


class InferenceComparator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_output_dirs()
        
        # Initialize models
        self.eomt_model = None
        self.yolo_model = None
        
        # Statistics tracking
        self.stats = {
            'eomt': {'loading_time': 0, 'total_inference_time': 0, 'num_images': 0, 'detections': []},
            'yolo': {'loading_time': 0, 'total_inference_time': 0, 'num_images': 0, 'detections': []}
        }
        
        # COCO format results
        self.coco_results = {
            'eomt': {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'object'}]},
            'yolo': {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'object'}]}
        }
        
    def setup_output_dirs(self):
        """Create output directory structure"""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.eomt_dir = self.output_dir / 'eomt'
        self.yolo_dir = self.output_dir / 'yolo'
        
        self.eomt_dir.mkdir(exist_ok=True)
        self.yolo_dir.mkdir(exist_ok=True)
        
    def load_eomt_model(self):
        """Load EoMT model from config and checkpoint"""
        print("Loading EoMT model...")
        start_time = time.time()
        
        # Parse config file
        parser = ArgumentParser()
        parser.add_class_arguments(MaskClassificationInstance, "model")
        
        with open(self.args.eomt_config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create model from config
        model_config = config_dict['model']
        model_class = model_config['class_path']
        model_args = model_config['init_args']
        
        # Update checkpoint path
        model_args['ckpt_path'] = self.args.eomt_checkpoint
        
        # Initialize model
        self.eomt_model = MaskClassificationInstance(**model_args)
        self.eomt_model.to(self.device)
        self.eomt_model.eval()
        
        loading_time = time.time() - start_time
        self.stats['eomt']['loading_time'] = loading_time
        print(f"EoMT model loaded in {loading_time:.2f} seconds")
        
    def load_yolo_model(self):
        """Load YOLOv8 model"""
        print("Loading YOLOv8 model...")
        start_time = time.time()
        
        self.yolo_model = YOLO(self.args.yolo_checkpoint)
        
        loading_time = time.time() - start_time
        self.stats['yolo']['loading_time'] = loading_time
        print(f"YOLOv8 model loaded in {loading_time:.2f} seconds")
        
    def preprocess_image_eomt(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for EoMT model"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image then to tensor
        pil_image = Image.fromarray(image)
        
        # Resize and pad to model input size
        img_size = self.eomt_model.img_size
        
        # Calculate scaling factor
        scale_factor = min(img_size[0] / pil_image.height, img_size[1] / pil_image.width)
        new_height = int(pil_image.height * scale_factor)
        new_width = int(pil_image.width * scale_factor)
        
        # Resize image
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to tensor
        tensor_image = F.pil_to_tensor(resized_image).float()
        
        # Pad to target size
        pad_height = img_size[0] - new_height
        pad_width = img_size[1] - new_width
        
        tensor_image = F.pad(tensor_image, [0, 0, pad_width, pad_height])
        
        # Add batch dimension and move to device
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        return tensor_image, (scale_factor, new_height, new_width, pil_image.size)
        
    def postprocess_eomt_results(self, mask_logits, class_logits, image_info, original_size):
        """Postprocess EoMT model results"""
        scale_factor, new_height, new_width, original_img_size = image_info
        
        # Get predictions
        mask_logits = mask_logits[-1]  # Use final layer predictions
        class_logits = class_logits[-1]
        
        # Process predictions
        scores = class_logits.softmax(dim=-1)[:, :-1]  # Remove background class
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=self.device)  # Single class
        
        # Filter by confidence threshold
        valid_mask = scores[:, 0] > self.args.confidence_threshold
        if not valid_mask.any():
            return []
        
        scores = scores[valid_mask]
        mask_logits = mask_logits[valid_mask]
        
        # Convert masks to binary
        masks = (mask_logits.sigmoid() > 0.5)
        
        # Resize masks back to original image size
        masks = F.resize(masks.float(), (new_height, new_width), interpolation=F.InterpolationMode.NEAREST)
        
        # Remove padding and resize to original
        masks = masks[:, :new_height, :new_width]
        masks = F.resize(masks, original_size[::-1], interpolation=F.InterpolationMode.NEAREST)
        
        detections = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_np = mask.cpu().numpy().astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Convert contour to segmentation format
                segmentation = largest_contour.flatten().tolist()
                
                detection = {
                    'bbox': [x, y, w, h],
                    'score': float(score[0].cpu()),
                    'category_id': 0,
                    'segmentation': [segmentation],
                    'area': float(cv2.contourArea(largest_contour)),
                    'mask': mask_np
                }
                detections.append(detection)
                
        return detections
        
    def inference_eomt(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """Run inference with EoMT model"""
        start_time = time.time()
        
        # Preprocess
        tensor_image, image_info = self.preprocess_image_eomt(image)
        
        # Inference
        with torch.no_grad():
            mask_logits, class_logits = self.eomt_model(tensor_image)
        
        # Postprocess
        detections = self.postprocess_eomt_results(mask_logits, class_logits, image_info, image.shape[:2])
        
        inference_time = time.time() - start_time
        return detections, inference_time
        
    def inference_yolo(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """Run inference with YOLOv8 model"""
        start_time = time.time()
        
        # Run inference
        results = self.yolo_model(image, conf=self.args.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.masks is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                masks = result.masks.data.cpu().numpy()
                
                for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # Resize mask to original image size
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        segmentation = largest_contour.flatten().tolist()
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(w), float(h)],
                            'score': float(score),
                            'category_id': 0,
                            'segmentation': [segmentation],
                            'area': float(cv2.contourArea(largest_contour)),
                            'mask': mask_binary
                        }
                        detections.append(detection)
        
        inference_time = time.time() - start_time
        return detections, inference_time
        
    def draw_annotations(self, image: np.ndarray, detections: List[Dict], model_name: str) -> np.ndarray:
        """Draw annotations on image"""
        annotated_image = image.copy()
        
        for detection in detections:
            # Draw mask
            mask = detection['mask']
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = [0, 255, 0]  # Green color
            annotated_image = cv2.addWeighted(annotated_image, 0.7, colored_mask, 0.3, 0)
            
            # Draw bounding box
            x, y, w, h = detection['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw score
            score_text = f"{detection['score']:.2f}"
            cv2.putText(annotated_image, score_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add model name
        cv2.putText(annotated_image, f"Model: {model_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return annotated_image
        
    def process_single_image(self, image_path: Path, image_id: int):
        """Process a single image with both models"""
        print(f"Processing {image_path.name}...")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load {image_path}")
            return
            
        # Add image info to COCO format
        height, width = image.shape[:2]
        image_info = {
            'id': image_id,
            'file_name': image_path.name,
            'height': height,
            'width': width
        }
        
        self.coco_results['eomt']['images'].append(image_info.copy())
        self.coco_results['yolo']['images'].append(image_info.copy())
        
        # EoMT inference
        eomt_detections, eomt_time = self.inference_eomt(image)
        self.stats['eomt']['total_inference_time'] += eomt_time
        self.stats['eomt']['detections'].append(len(eomt_detections))
        
        # YOLOv8 inference
        yolo_detections, yolo_time = self.inference_yolo(image)
        self.stats['yolo']['total_inference_time'] += yolo_time
        self.stats['yolo']['detections'].append(len(yolo_detections))
        
        # Draw annotations and save
        eomt_annotated = self.draw_annotations(image, eomt_detections, "EoMT")
        yolo_annotated = self.draw_annotations(image, yolo_detections, "YOLOv8")
        
        cv2.imwrite(str(self.eomt_dir / image_path.name), eomt_annotated)
        cv2.imwrite(str(self.yolo_dir / image_path.name), yolo_annotated)
        
        # Add annotations to COCO format
        annotation_id = len(self.coco_results['eomt']['annotations'])
        
        for detection in eomt_detections:
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': detection['category_id'],
                'bbox': detection['bbox'],
                'area': detection['area'],
                'segmentation': detection['segmentation'],
                'iscrowd': 0,
                'score': detection['score']
            }
            self.coco_results['eomt']['annotations'].append(annotation)
            annotation_id += 1
            
        for detection in yolo_detections:
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': detection['category_id'],
                'bbox': detection['bbox'],
                'area': detection['area'],
                'segmentation': detection['segmentation'],
                'iscrowd': 0,
                'score': detection['score']
            }
            self.coco_results['yolo']['annotations'].append(annotation)
            annotation_id += 1
            
        # Log individual image stats
        with open(self.output_dir / 'detailed_log.txt', 'a') as f:
            f.write(f"\n{image_path.name}:\n")
            f.write(f"  EoMT: {len(eomt_detections)} detections, {eomt_time:.3f}s\n")
            f.write(f"  YOLOv8: {len(yolo_detections)} detections, {yolo_time:.3f}s\n")
            
    def process_all_images(self):
        """Process all images in the input folder"""
        image_folder = Path(self.args.image_folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        image_paths = [
            p for p in image_folder.iterdir() 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {image_folder}")
            return
            
        print(f"Found {len(image_paths)} images")
        
        # Initialize detailed log
        with open(self.output_dir / 'detailed_log.txt', 'w') as f:
            f.write(f"Model Inference Comparison\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image folder: {self.args.image_folder}\n")
            f.write(f"EoMT config: {self.args.eomt_config}\n")
            f.write(f"EoMT checkpoint: {self.args.eomt_checkpoint}\n")
            f.write(f"YOLOv8 checkpoint: {self.args.yolo_checkpoint}\n")
            f.write(f"Confidence threshold: {self.args.confidence_threshold}\n")
            f.write(f"Total images: {len(image_paths)}\n\n")
            
        # Process images
        for i, image_path in enumerate(image_paths):
            self.process_single_image(image_path, i)
            
        self.stats['eomt']['num_images'] = len(image_paths)
        self.stats['yolo']['num_images'] = len(image_paths)
        
    def save_results(self):
        """Save final results and statistics"""
        # Save COCO format results
        with open(self.output_dir / 'eomt_results.json', 'w') as f:
            json.dump(self.coco_results['eomt'], f, indent=2)
            
        with open(self.output_dir / 'yolo_results.json', 'w') as f:
            json.dump(self.coco_results['yolo'], f, indent=2)
            
        # Calculate and save statistics
        with open(self.output_dir / 'summary_statistics.txt', 'w') as f:
            f.write("Model Comparison Summary\n")
            f.write("="*50 + "\n\n")
            
            for model_name, stats in self.stats.items():
                f.write(f"{model_name.upper()} Model:\n")
                f.write(f"  Loading time: {stats['loading_time']:.2f} seconds\n")
                f.write(f"  Total inference time: {stats['total_inference_time']:.2f} seconds\n")
                f.write(f"  Average inference time per image: {stats['total_inference_time']/stats['num_images']:.3f} seconds\n")
                f.write(f"  Total detections: {sum(stats['detections'])}\n")
                f.write(f"  Average detections per image: {np.mean(stats['detections']):.2f}\n")
                f.write(f"  Min detections per image: {min(stats['detections']) if stats['detections'] else 0}\n")
                f.write(f"  Max detections per image: {max(stats['detections']) if stats['detections'] else 0}\n")
                f.write("\n")
                
        print("\nResults saved:")
        print(f"  - Annotated images: {self.eomt_dir} and {self.yolo_dir}")
        print(f"  - COCO results: {self.output_dir}/eomt_results.json and {self.output_dir}/yolo_results.json")
        print(f"  - Summary statistics: {self.output_dir}/summary_statistics.txt")
        print(f"  - Detailed log: {self.output_dir}/detailed_log.txt")
        
    def run(self):
        """Run the complete comparison"""
        print("Starting model comparison...")
        
        # Load models
        self.load_eomt_model()
        self.load_yolo_model()
        
        # Process all images
        self.process_all_images()
        
        # Save results
        self.save_results()
        
        print("Comparison completed!")


def main():
    parser = argparse.ArgumentParser(description='Compare EoMT and YOLOv8 instance segmentation models')
    parser.add_argument('--image_folder', type=str, required=True,
                       help='Path to folder containing images for inference')
    parser.add_argument('--eomt_config', type=str, required=True,
                       help='Path to EoMT model config file')
    parser.add_argument('--eomt_checkpoint', type=str, required=True,
                       help='Path to EoMT model checkpoint')
    parser.add_argument('--yolo_checkpoint', type=str, required=True,
                       help='Path to YOLOv8 model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image_folder):
        print(f"Error: Image folder {args.image_folder} does not exist")
        return
        
    if not os.path.exists(args.eomt_config):
        print(f"Error: EoMT config file {args.eomt_config} does not exist")
        return
        
    if not os.path.exists(args.eomt_checkpoint):
        print(f"Error: EoMT checkpoint {args.eomt_checkpoint} does not exist")
        return
        
    if not os.path.exists(args.yolo_checkpoint):
        print(f"Error: YOLOv8 checkpoint {args.yolo_checkpoint} does not exist")
        return
    
    # Run comparison
    comparator = InferenceComparator(args)
    comparator.run()


if __name__ == "__main__":
    main()
