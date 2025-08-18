import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import glob
from tqdm import tqdm

from model import build_model
from utils import denormalize_image, setup_logging
from dataset import COCOSegmentationDataset
import pycocotools.mask as mask_utils


class SegmentationInference:
    """Inference class for segmentation model"""
    
    def __init__(
        self,
        model_path: str,
        repo_dir: str = '.',
        backbone_name: str = 'dinov3_vitl16',
        weights: Optional[str] = None,
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        use_satellite_norm: bool = True,
        target_size: int = 512
    ):
        self.model_path = model_path
        self.repo_dir = repo_dir
        self.backbone_name = backbone_name
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.use_satellite_norm = use_satellite_norm
        self.target_size = target_size
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self._load_model()
        
        # Setup transforms
        self._setup_transforms()
    
    def _load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration from checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            num_classes = config.get('num_classes', 80)
            num_queries = config.get('num_queries', 100)
            hidden_dim = config.get('hidden_dim', 1024)
        else:
            # Default values - you might want to make these configurable
            num_classes = 80
            num_queries = 100
            hidden_dim = 1024
        
        # Build model
        self.model = build_model(
            backbone_name=self.backbone_name,
            repo_dir=self.repo_dir,
            weights=self.weights,
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DDP wrapper
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _setup_transforms(self):
        """Setup image transforms"""
        from torchvision import transforms
        
        if self.use_satellite_norm:
            normalize = transforms.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143)
            )
        else:
            normalize = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.target_size, self.target_size),
                antialias=True
            ),
            normalize
        ])
    
    def predict_single(
        self,
        image_path: str,
        save_visualization: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Predict on single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Post-process predictions
        predictions = self._post_process_predictions(
            outputs, [(orig_h, orig_w)]
        )[0]
        
        # Save visualization if requested
        if save_visualization and output_dir:
            self._save_visualization(
                image, predictions, image_path, output_dir
            )
        
        return predictions
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8,
        save_visualizations: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """Predict on batch of images"""
        all_predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_orig_sizes = []
            
            # Load and preprocess batch
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                orig_w, orig_h = image.size
                batch_orig_sizes.append((orig_h, orig_w))
                
                image_tensor = self.transform(image)
                batch_images.append(image_tensor)
            
            # Stack batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # Post-process predictions
            batch_predictions = self._post_process_predictions(
                outputs, batch_orig_sizes
            )
            
            # Save visualizations if requested
            if save_visualizations and output_dir:
                for j, (path, pred) in enumerate(zip(batch_paths, batch_predictions)):
                    image = Image.open(path).convert('RGB')
                    self._save_visualization(image, pred, path, output_dir)
            
            all_predictions.extend(batch_predictions)
        
        return all_predictions
    
    def predict_directory(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: int = 8,
        save_visualizations: bool = True,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ) -> Dict[str, Dict]:
        """Predict on all images in directory"""
        # Find all image files
        image_paths = []
        for ext in image_extensions:
            pattern = os.path.join(input_dir, f"**/*{ext}")
            image_paths.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(input_dir, f"**/*{ext.upper()}")
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        if not image_paths:
            print(f"No images found in {input_dir}")
            return {}
        
        print(f"Found {len(image_paths)} images")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Predict on all images
        predictions = self.predict_batch(
            image_paths=image_paths,
            batch_size=batch_size,
            save_visualizations=save_visualizations,
            output_dir=output_dir
        )
        
        # Create results dictionary
        results = {}
        for path, pred in zip(image_paths, predictions):
            rel_path = os.path.relpath(path, input_dir)
            results[rel_path] = pred
        
        # Save results to JSON
        results_file = os.path.join(output_dir, 'predictions.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for path, pred in results.items():
                serializable_results[path] = {
                    'boxes': pred['boxes'].tolist() if isinstance(pred['boxes'], np.ndarray) else pred['boxes'],
                    'scores': pred['scores'].tolist() if isinstance(pred['scores'], np.ndarray) else pred['scores'],
                    'labels': pred['labels'].tolist() if isinstance(pred['labels'], np.ndarray) else pred['labels'],
                    'masks_rle': pred['masks_rle']  # Already in serializable format
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        return results
    
    def _post_process_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        orig_sizes: List[Tuple[int, int]]
    ) -> List[Dict]:
        """Post-process model outputs"""
        pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [batch, num_queries, 4]
        pred_masks = outputs['pred_masks']    # [batch, num_queries, H, W]
        
        batch_size = pred_logits.shape[0]
        batch_predictions = []
        
        for i in range(batch_size):
            orig_h, orig_w = orig_sizes[i]
            
            # Get predictions for this image
            logits = pred_logits[i]  # [num_queries, num_classes+1]
            boxes = pred_boxes[i]    # [num_queries, 4]
            masks = pred_masks[i]    # [num_queries, H, W]
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            scores, labels = probs[:, :-1].max(dim=-1)  # Exclude no-object class
            
            # Filter by confidence
            keep = scores > self.confidence_threshold
            if not keep.any():
                # No detections
                batch_predictions.append({
                    'boxes': np.empty((0, 4)),
                    'scores': np.empty(0),
                    'labels': np.empty(0),
                    'masks': np.empty((0, orig_h, orig_w)),
                    'masks_rle': []
                })
                continue
            
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            masks = masks[keep]
            
            # Convert normalized boxes to pixel coordinates
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
            
            # Convert masks to original size
            masks_resized = F.interpolate(
                masks.unsqueeze(1),
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Apply sigmoid and threshold
            masks_binary = (masks_resized.sigmoid() > 0.5).cpu().numpy()
            
            # Convert masks to RLE format
            masks_rle = []
            for mask in masks_binary:
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('utf-8')
                masks_rle.append(rle)
            
            predictions = {
                'boxes': boxes.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'labels': labels.cpu().numpy(),
                'masks': masks_binary,
                'masks_rle': masks_rle
            }
            
            batch_predictions.append(predictions)
        
        return batch_predictions
    
    def _save_visualization(
        self,
        image: Image.Image,
        predictions: Dict,
        image_path: str,
        output_dir: str
    ):
        """Save visualization of predictions"""
        # Convert PIL image to numpy
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(img_array)
        
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']
        masks = predictions['masks']
        
        # Draw predictions
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)
            
            # Draw mask
            mask_colored = np.zeros((h, w, 4))
            mask_colored[:, :, 0] = mask  # Red channel
            mask_colored[:, :, 3] = mask * 0.3  # Alpha
            axes[1].imshow(mask_colored)
            
            # Draw label
            axes[1].text(
                x1, y1 - 5, f'Class {label}: {score:.2f}',
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        axes[1].set_title(f'Predictions ({len(boxes)} detections)')
        axes[1].axis('off')
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_prediction.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DinoV3 Segmentation Inference')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--repo_dir', type=str, default='.',
                       help='DinoV3 repository directory')
    parser.add_argument('--backbone_name', type=str, default='dinov3_vitl16',
                       help='DinoV3 backbone name')
    parser.add_argument('--weights', type=str, default=None,
                       help='Pretrained backbone weights')
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input image/directory path')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'directory'],
                       default='auto', help='Inference mode')
    
    # Inference arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for batch inference')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--target_size', type=int, default=512,
                       help='Target image size')
    parser.add_argument('--use_satellite_norm', action='store_true', default=True,
                       help='Use satellite imagery normalization')
    
    # Visualization arguments
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                       help='Save prediction visualizations')
    parser.add_argument('--image_extensions', nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help='Image file extensions to process')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Setup logging
    setup_logging('inference.log')
    
    # Determine inference mode
    if args.mode == 'auto':
        if os.path.isfile(args.input):
            mode = 'single'
        elif os.path.isdir(args.input):
            mode = 'directory'
        else:
            raise ValueError(f"Input path does not exist: {args.input}")
    else:
        mode = args.mode
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    print("Initializing inference...")
    inference = SegmentationInference(
        model_path=args.model_path,
        repo_dir=args.repo_dir,
        backbone_name=args.backbone_name,
        weights=args.weights,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        use_satellite_norm=args.use_satellite_norm,
        target_size=args.target_size
    )
    
    # Run inference
    start_time = time.time()
    
    if mode == 'single':
        print(f"Running inference on single image: {args.input}")
        predictions = inference.predict_single(
            image_path=args.input,
            save_visualization=args.save_visualizations,
            output_dir=args.output_dir
        )
        
        # Save predictions
        output_file = os.path.join(args.output_dir, 'predictions.json')
        with open(output_file, 'w') as f:
            # Convert numpy arrays to lists
            serializable_pred = {
                'boxes': predictions['boxes'].tolist(),
                'scores': predictions['scores'].tolist(),
                'labels': predictions['labels'].tolist(),
                'masks_rle': predictions['masks_rle']
            }
            json.dump(serializable_pred, f, indent=2)
        
        print(f"Found {len(predictions['boxes'])} objects")
        print(f"Results saved to {output_file}")
    
    elif mode == 'directory':
        print(f"Running inference on directory: {args.input}")
        results = inference.predict_directory(
            input_dir=args.input,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            save_visualizations=args.save_visualizations,
            image_extensions=tuple(args.image_extensions)
        )
        
        total_detections = sum(len(pred['boxes']) for pred in results.values())
        print(f"Processed {len(results)} images")
        print(f"Total detections: {total_detections}")
    
    else:
        # Batch mode - expect list of image paths
        if os.path.isfile(args.input):
            with open(args.input, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
        else:
            raise ValueError("For batch mode, input should be a text file with image paths")
        
        print(f"Running batch inference on {len(image_paths)} images")
        predictions = inference.predict_batch(
            image_paths=image_paths,
            batch_size=args.batch_size,
            save_visualizations=args.save_visualizations,
            output_dir=args.output_dir
        )
        
        # Save results
        results_file = os.path.join(args.output_dir, 'batch_predictions.json')
        with open(results_file, 'w') as f:
            serializable_results = []
            for i, (path, pred) in enumerate(zip(image_paths, predictions)):
                serializable_results.append({
                    'image_path': path,
                    'boxes': pred['boxes'].tolist(),
                    'scores': pred['scores'].tolist(),
                    'labels': pred['labels'].tolist(),
                    'masks_rle': pred['masks_rle']
                })
            json.dump(serializable_results, f, indent=2)
        
        total_detections = sum(len(pred['boxes']) for pred in predictions)
        print(f"Total detections: {total_detections}")
        print(f"Results saved to {results_file}")
    
    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()
