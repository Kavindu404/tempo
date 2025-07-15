#!/usr/bin/env python3
"""
YOLO Segmentation + DINOv2 Multi-Label Classification Pipeline

This script:
1. Loads COCO JSON and images
2. Runs YOLOv8 segmentation to get polygons and bboxes
3. Crops and masks polygon regions
4. Classifies using 3 DINOv2 models (roof-type, roof-condition, roof-material)
5. Saves results in COCO format with attention maps for low-confidence predictions

Usage:
    python yolo_dinov2_pipeline.py --input_json coco.json --image_dir images/ 
                                   --yolo_model yolo_seg.pt --output_dir results/
                                   --roof_type_model roof_type.pth --roof_condition_model roof_condition.pth
                                   --roof_material_model roof_material.pth
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from tqdm import tqdm

# Import your modules
from model import ProductionDINOv2MultiLabelClassifier, AttentionVisualizer
from dataset import create_transforms

class YOLODINOv2Pipeline:
    def __init__(self, yolo_model_path, roof_type_model, roof_condition_model, 
                 roof_material_model, device='cuda'):
        """
        Initialize the pipeline
        
        Args:
            yolo_model_path: Path to YOLOv8 segmentation model
            roof_type_model: Path to roof type classifier
            roof_condition_model: Path to roof condition classifier
            roof_material_model: Path to roof material classifier
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        print("Loading YOLO segmentation model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load DINOv2 classifiers
        print("Loading DINOv2 classifiers...")
        self.classifiers = {
            'roof-type': self._load_dinov2_model(roof_type_model),
            'roof-condition': self._load_dinov2_model(roof_condition_model),
            'roof-material': self._load_dinov2_model(roof_material_model)
        }
        
        # Create transform for DINOv2
        self.transform = create_transforms(is_training=False)
        
        # Initialize attention visualizer
        self.attention_visualizer = AttentionVisualizer()
        
        print(f"✅ Pipeline initialized on {self.device}")
    
    def _load_dinov2_model(self, model_path):
        """Load a DINOv2 classifier with proper parameter inference"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model_state = checkpoint['model_state_dict']
            
            # Get parameters from checkpoint or infer them
            if 'num_classes' in checkpoint and checkpoint['num_classes'] is not None:
                num_classes = checkpoint['num_classes']
            else:
                # Fallback: infer from classifier weight
                classifier_weight = self._find_classifier_weight(model_state)
                num_classes = classifier_weight.shape[0]
            
            # Get LoRA parameters
            if 'lora_r' in checkpoint and 'lora_alpha' in checkpoint:
                lora_r = checkpoint['lora_r']
                lora_alpha = checkpoint['lora_alpha']
            else:
                # Fallback: infer from state dict
                lora_r, lora_alpha = self._infer_lora_params(model_state)
            
            # Create model
            from model import DINOv2MultiLabelClassifier
            model = DINOv2MultiLabelClassifier(
                num_classes=num_classes,
                r=lora_r,
                alpha=lora_alpha
            )
            
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()
            
            # Get class names if available
            class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
            
            return {
                'model': model,
                'class_names': class_names,
                'num_classes': num_classes
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")
    
    def _find_classifier_weight(self, state_dict):
        """Find classifier weight in state dict"""
        possible_keys = [
            'classifier.weight', 'classifier.2.weight', 'classifier.1.weight', 
            'classifier.3.weight', 'classifier.4.weight'
        ]
        
        for key in possible_keys:
            if key in state_dict:
                return state_dict[key]
        
        # Search for any linear layer in classifier module
        classifier_weights = []
        for key, tensor in state_dict.items():
            if (key.startswith('classifier.') and 
                key.endswith('.weight') and 
                len(tensor.shape) == 2):
                classifier_weights.append((key, tensor))
        
        if classifier_weights:
            classifier_weights.sort(key=lambda x: x[0])
            return classifier_weights[-1][1]
        
        raise ValueError("Could not find classifier weight")
    
    def _infer_lora_params(self, state_dict):
        """Infer LoRA parameters from state dict"""
        lora_r = None
        for key, tensor in state_dict.items():
            if '.lora_A.weight' in key:
                lora_r = tensor.shape[0]
                break
        
        if lora_r is None:
            lora_r = 16  # Default
        
        lora_alpha = lora_r * 2  # Common convention
        return lora_r, lora_alpha
    
    def _safe_filename(self, name):
        """Convert class name to safe filename by replacing problematic characters"""
        return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    
    def _polygon_to_mask(self, polygon, img_shape):
        """Convert polygon coordinates to binary mask"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        if len(polygon) >= 6:  # At least 3 points
            points = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        
        return mask.astype(bool)
    
    def _create_masked_crop(self, image, bbox, polygon):
        """Create masked crop from image using polygon"""
        # Convert bbox to integers
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure bbox is within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return None
            return None
        
        # Create mask for the polygon
        mask = self._polygon_to_mask(polygon, image.shape)
        
        # Crop the region
        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        
        # Create masked image (polygon area kept, rest filled with zeros)
        masked_crop = np.zeros_like(image_crop)
        masked_crop[mask_crop] = image_crop[mask_crop]
        
        return masked_crop
    
    def _classify_batch(self, image_arrays, model_info, threshold=0.4):
        """Classify batch of images using DINOv2 model"""
        if not image_arrays:
            return []
        
        # Convert all images to tensors
        tensor_batch = []
        pil_images = []
        
        for image_array in image_arrays:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            tensor_image = self.transform(pil_image)
            tensor_batch.append(tensor_image)
            pil_images.append(pil_image)
        
        # Stack into batch
        batch_tensor = torch.stack(tensor_batch).to(self.device)
        
        # Get predictions and attention for batch
        with torch.no_grad():
            logits, attention_weights = model_info['model'](batch_tensor, return_attention=True)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        
        # Process results for each image in batch
        results = []
        for i, (probs, pil_image) in enumerate(zip(probabilities, pil_images)):
            top_class_idx = np.argmax(probs)
            top_class_name = model_info['class_names'][top_class_idx]
            top_confidence = probs[top_class_idx]
            has_confident_pred = np.any(probs > threshold)
            
            # Get attention for this specific image
            img_attention = attention_weights[i] if attention_weights is not None else None
            
            results.append({
                'class_idx': top_class_idx,
                'class_name': top_class_name,
                'confidence': float(top_confidence),
                'all_probabilities': probs,
                'has_confident_pred': has_confident_pred,
                'attention_weights': img_attention,
                'original_image': pil_image
            })
        
        return results
    
    def _save_attention_visualization(self, result, model_name, output_dir, image_id, detection_id):
        """Save attention visualization for low-confidence predictions"""
        if result['has_confident_pred']:
            return  # Only save for low-confidence predictions
        
        # Create model-specific attention directory
        safe_model_name = self._safe_filename(model_name)
        attention_dir = Path(output_dir) / f"attention_maps_{safe_model_name}"
        attention_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        safe_class_name = self._safe_filename(result['class_name'])
        filename = f"img_{image_id}_det_{detection_id}_{safe_class_name}_conf_{result['confidence']:.3f}.png"
        save_path = attention_dir / filename
        
        # Create visualization
        try:
            self.attention_visualizer.visualize_attention(
                image=result['original_image'],
                attention_weights=result['attention_weights'],
                predictions=torch.tensor(result['all_probabilities']),
                class_names=[self._safe_filename(name) for name in self.classifiers[model_name]['class_names']],
                threshold=0.4,
                save_path=str(save_path)
            )
        except Exception as e:
            print(f"Warning: Failed to save attention visualization: {e}")
    
    def _create_coco_annotation(self, detection_id, image_id, bbox, polygon, class_idx, confidence, area):
        """Create COCO format annotation"""
        # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
        x1, y1, x2, y2 = bbox
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]
        
        annotation = {
            "id": detection_id,
            "image_id": image_id,
            "category_id": int(class_idx),
            "bbox": coco_bbox,
            "segmentation": [polygon.tolist()] if isinstance(polygon, np.ndarray) else [polygon],
            "area": float(area),
            "iscrowd": 0,
            "score": float(confidence)
        }
        
        return annotation
    
    def process_dataset(self, input_json, image_dir, output_dir, confidence_threshold=0.4, batch_size=16):
        """Process the entire dataset with batch processing"""
        print("Starting dataset processing...")
        
        # Load input COCO JSON
        with open(input_json, 'r') as f:
            input_data = json.load(f)
        
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results for each model
        results = {}
        for model_name in self.classifiers.keys():
            safe_model_name = self._safe_filename(model_name)
            
            results[model_name] = {
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Create categories from class names
            for idx, class_name in enumerate(self.classifiers[model_name]['class_names']):
                results[model_name]["categories"].append({
                    "id": idx,
                    "name": class_name,
                    "supercategory": model_name
                })
        
        # Copy images info to all results
        for model_name in results.keys():
            results[model_name]["images"] = input_data["images"].copy()
        
        # Process each image
        detection_counters = {model_name: 0 for model_name in self.classifiers.keys()}
        
        # Collect all crops for batch processing
        all_crops = []
        crop_metadata = []
        
        print("Extracting crops from YOLO detections...")
        for image_info in tqdm(input_data["images"], desc="Processing images"):
            image_id = image_info["id"]
            image_filename = image_info["file_name"]
            image_path = os.path.join(image_dir, image_filename)
            
            # Load image
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            # Run YOLO segmentation
            try:
                yolo_results = self.yolo_model(image_path, verbose=False)
            except Exception as e:
                print(f"YOLO inference failed for {image_path}: {e}")
                continue
            
            # Process each detection
            for result in yolo_results:
                if result.masks is None:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                masks = result.masks.xy  # List of polygons
                confidences = result.boxes.conf.cpu().numpy()
                
                for i, (bbox, polygon, yolo_conf) in enumerate(zip(boxes, masks, confidences)):
                    # Create masked crop
                    masked_crop = self._create_masked_crop(image, bbox, polygon.flatten())
                    
                    if masked_crop is None:
                        continue
                    
                    # Calculate area
                    area = np.sum(self._polygon_to_mask(polygon.flatten(), image.shape))
                    
                    # Store crop and metadata
                    all_crops.append(masked_crop)
                    crop_metadata.append({
                        'image_id': image_id,
                        'bbox': bbox,
                        'polygon': polygon.flatten(),
                        'area': area,
                        'yolo_conf': yolo_conf
                    })
        
        print(f"Extracted {len(all_crops)} crops, processing in batches of {batch_size}...")
        
        # Process crops in batches for each model
        for model_name, model_info in self.classifiers.items():
            print(f"Processing {model_name} classifier...")
            
            for start_idx in tqdm(range(0, len(all_crops), batch_size), 
                                desc=f"Classifying with {model_name}"):
                end_idx = min(start_idx + batch_size, len(all_crops))
                
                # Get batch of crops
                batch_crops = all_crops[start_idx:end_idx]
                batch_metadata = crop_metadata[start_idx:end_idx]
                
                # Classify batch
                batch_results = self._classify_batch(batch_crops, model_info, confidence_threshold)
                
                # Process results
                for crop_meta, classification_result in zip(batch_metadata, batch_results):
                    detection_counters[model_name] += 1
                    detection_id = detection_counters[model_name]
                    
                    # Save attention visualization if low confidence
                    self._save_attention_visualization(
                        classification_result, model_name, output_dir, 
                        crop_meta['image_id'], detection_id
                    )
                    
                    # Create COCO annotation
                    annotation = self._create_coco_annotation(
                        detection_id, crop_meta['image_id'], crop_meta['bbox'], 
                        crop_meta['polygon'], classification_result['class_idx'], 
                        classification_result['confidence'], crop_meta['area']
                    )
                    
                    results[model_name]["annotations"].append(annotation)
        
        # Save results for each model
        for model_name, model_results in results.items():
            safe_model_name = self._safe_filename(model_name)
            output_file = output_path / f"{safe_model_name}_results.json"
            
            with open(output_file, 'w') as f:
                json.dump(model_results, f, indent=2)
            
            print(f"✅ Saved {len(model_results['annotations'])} annotations for {model_name} to {output_file}")
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_json": str(input_json),
            "output_dir": str(output_dir),
            "batch_size": batch_size,
            "total_images": len(input_data["images"]),
            "total_crops": len(all_crops),
            "results_summary": {
                model_name: {
                    "total_detections": len(results[model_name]["annotations"]),
                    "num_classes": len(results[model_name]["categories"]),
                    "class_names": [cat["name"] for cat in results[model_name]["categories"]]
                }
                for model_name in results.keys()
            }
        }
        
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Processing completed. Summary saved to {summary_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='YOLO + DINOv2 Classification Pipeline')
    
    # Input arguments
    parser.add_argument('--input_json', type=str, required=True,
                       help='Input COCO JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--yolo_model', type=str, required=True,
                       help='Path to YOLOv8 segmentation model')
    
    # Model arguments
    parser.add_argument('--roof_type_model', type=str, required=True,
                       help='Path to roof type classifier')
    parser.add_argument('--roof_condition_model', type=str, required=True,
                       help='Path to roof condition classifier')
    parser.add_argument('--roof_material_model', type=str, required=True,
                       help='Path to roof material classifier')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for DINOv2 inference (default: 16)')
    parser.add_argument('--confidence_threshold', type=float, default=0.4,
                       help='Confidence threshold for saving attention maps')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.input_json, args.yolo_model, args.roof_type_model, 
                      args.roof_condition_model, args.roof_material_model]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    
    # Initialize pipeline
    pipeline = YOLODINOv2Pipeline(
        yolo_model_path=args.yolo_model,
        roof_type_model=args.roof_type_model,
        roof_condition_model=args.roof_condition_model,
        roof_material_model=args.roof_material_model,
        device=args.device
    )
    
    # Process dataset
    results = pipeline.process_dataset(
        input_json=args.input_json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    for model_name in results.keys():
        safe_model_name = pipeline._safe_filename(model_name)
        print(f"  - {safe_model_name}_results.json")
        print(f"  - attention_maps_{safe_model_name}/ (if low confidence predictions)")
    print("  - processing_summary.json")

if __name__ == "__main__":
    main()
