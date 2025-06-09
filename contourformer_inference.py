import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Optional
import pycocotools.mask as mask_utils

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from src.core import YAMLConfig


class ContourFormerInference:
    """
    ContourFormer model inference class for instance segmentation.
    """
    
    def __init__(self, config_path: str, model_checkpoint: str, device: str = 'cpu'):
        """
        Initialize the ContourFormer inference model.
        
        Args:
            config_path: Path to the model configuration YAML file
            model_checkpoint: Path to the trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.config_path = config_path
        self.model_checkpoint = model_checkpoint
        
        # Load model configuration and initialize model
        self._load_model()
        
        # Set up visualization parameters
        self.colors = [
            "#1F77B4", "#FF7F0E", "#2EA02C", "#D62827", "#9467BD", 
            "#8C564B", "#E377C2", "#7E7E7E", "#BCBD20", "#1ABECF",
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
        ]
        
    def _load_model(self):
        """Load the ContourFormer model and configuration."""
        print(f"Loading model configuration from: {self.config_path}")
        
        # Load configuration
        cfg = YAMLConfig(self.config_path)
        
        # Disable pretrained weights loading since we're loading from checkpoint
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            
        # Initialize model
        self.model = cfg.model.to(self.device)
        self.postprocessor = cfg.postprocessor.to(self.device)
        
        # Load checkpoint
        print(f"Loading model checkpoint from: {self.model_checkpoint}")
        checkpoint = torch.load(self.model_checkpoint, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
            print("Loading from EMA weights")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Loading from model weights")
        else:
            state_dict = checkpoint
            print("Loading directly from checkpoint")
        
        # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                cleaned_state_dict[key[7:]] = value
            else:
                cleaned_state_dict[key] = value
        
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        
        # Deploy mode for inference
        self.model = self.model.deploy()
        self.postprocessor = self.postprocessor.deploy()
        
        print("ContourFormer model loaded successfully!")
        
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess the image for model inference.
        
        Args:
            image: PIL Image to preprocess
            target_size: Target size for the model input
            
        Returns:
            Tuple of (preprocessed tensor, preprocessing info dict)
        """
        # Get original size
        orig_w, orig_h = image.size
        
        # Calculate scaling to maintain aspect ratio
        scale_w = target_size[0] / orig_w
        scale_h = target_size[1] / orig_h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image
        resized_image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Create padded image
        padded_image = Image.new('RGB', target_size, (0, 0, 0))
        
        # Calculate padding offsets
        offset_x = (target_size[0] - new_w) // 2
        offset_y = (target_size[1] - new_h) // 2
        
        # Paste resized image onto padded canvas
        padded_image.paste(resized_image, (offset_x, offset_y))
        
        # Convert to tensor
        image_tensor = torch.tensor(np.array(padded_image), dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Store preprocessing info for post-processing
        preprocess_info = {
            'original_size': (orig_w, orig_h),
            'target_size': target_size,
            'scale': scale,
            'new_size': (new_w, new_h),
            'offset': (offset_x, offset_y)
        }
        
        return image_tensor.to(self.device), preprocess_info
    
    @torch.no_grad()
    def run_inference(self, image_tensor: torch.Tensor) -> Dict:
        """
        Run ContourFormer inference on the preprocessed image.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary containing model predictions
        """
        print("Running ContourFormer inference...")
        
        # Set input size for postprocessing
        input_size = torch.tensor([[image_tensor.shape[-1], image_tensor.shape[-2]]], 
                                 device=self.device)
        
        # Run model inference
        outputs = self.model(image_tensor)
        
        # Get original target size (same as input for satellite imagery)
        orig_target_size = torch.tensor([[image_tensor.shape[-1], image_tensor.shape[-2]]], 
                                      device=self.device)
        
        # Run postprocessing
        results = self.postprocessor(outputs, orig_target_size, input_size)
        
        print(f"Inference completed. Found {len(results[0]['labels'])} instances.")
        return results[0]  # Return first (and only) batch result
    
    def postprocess_results(self, results: Dict, preprocess_info: Dict) -> Dict:
        """
        Convert results back to original image coordinates.
        
        Args:
            results: Model prediction results
            preprocess_info: Information from preprocessing step
            
        Returns:
            Results adjusted to original image coordinates
        """
        # Extract preprocessing info
        orig_w, orig_h = preprocess_info['original_size']
        scale = preprocess_info['scale']
        offset_x, offset_y = preprocess_info['offset']
        
        # Adjust boxes and masks to original coordinates
        adjusted_results = {}
        
        # Copy tensors to CPU and convert to numpy for easier manipulation
        labels = results['labels'].cpu().numpy()
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        masks = results['masks']  # Already numpy array
        
        # Adjust bounding boxes
        adjusted_boxes = boxes.copy()
        adjusted_boxes[:, [0, 2]] = (adjusted_boxes[:, [0, 2]] - offset_x) / scale  # x coordinates
        adjusted_boxes[:, [1, 3]] = (adjusted_boxes[:, [1, 3]] - offset_y) / scale  # y coordinates
        
        # Clip to original image bounds
        adjusted_boxes[:, [0, 2]] = np.clip(adjusted_boxes[:, [0, 2]], 0, orig_w)
        adjusted_boxes[:, [1, 3]] = np.clip(adjusted_boxes[:, [1, 3]], 0, orig_h)
        
        # Adjust masks (resize and crop to original coordinates)
        adjusted_masks = []
        for mask in masks:
            if mask is not None and mask.size > 0:
                # Remove padding
                mask_cropped = mask[0, offset_y:offset_y + int(orig_h * scale), 
                                  offset_x:offset_x + int(orig_w * scale)]
                
                # Resize back to original size
                mask_resized = cv2.resize(mask_cropped.astype(np.float32), 
                                        (orig_w, orig_h), 
                                        interpolation=cv2.INTER_LINEAR)
                
                # Convert back to binary mask
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                adjusted_masks.append(mask_binary[None])  # Add batch dimension back
            else:
                adjusted_masks.append(np.zeros((1, orig_h, orig_w), dtype=np.uint8))
        
        adjusted_masks = np.array(adjusted_masks)
        
        adjusted_results = {
            'labels': labels,
            'boxes': adjusted_boxes,
            'scores': scores,
            'masks': adjusted_masks
        }
        
        return adjusted_results
    
    def predict(self, image: Image.Image, score_threshold: float = 0.5) -> Dict:
        """
        Complete prediction pipeline for a single image.
        
        Args:
            image: PIL Image to process
            score_threshold: Minimum confidence score for detections
            
        Returns:
            Dictionary containing filtered predictions in original image coordinates
        """
        # Preprocess image
        image_tensor, preprocess_info = self.preprocess_image(image)
        
        # Run inference
        raw_results = self.run_inference(image_tensor)
        
        # Postprocess results to original coordinates
        adjusted_results = self.postprocess_results(raw_results, preprocess_info)
        
        # Filter by score threshold
        scores = adjusted_results['scores']
        valid_indices = scores >= score_threshold
        
        filtered_results = {
            'labels': adjusted_results['labels'][valid_indices],
            'boxes': adjusted_results['boxes'][valid_indices],
            'scores': adjusted_results['scores'][valid_indices],
            'masks': adjusted_results['masks'][valid_indices]
        }
        
        print(f"Filtered to {len(filtered_results['labels'])} instances above threshold {score_threshold}")
        
        return filtered_results
    
    def visualize_results(self, image: Image.Image, results: Dict, 
                         output_path: Optional[str] = None,
                         show_boxes: bool = True,
                         show_masks: bool = True,
                         show_labels: bool = True,
                         line_width: int = 2) -> Image.Image:
        """
        Visualize the ContourFormer results on the original image.
        
        Args:
            image: Original PIL Image
            results: Model prediction results (already filtered and in original coordinates)
            output_path: Optional path to save the annotated image
            show_boxes: Whether to draw bounding boxes
            show_masks: Whether to draw contour masks
            show_labels: Whether to draw labels and scores
            line_width: Width of drawn lines
            
        Returns:
            Annotated PIL Image
        """
        print(f"Visualizing {len(results['labels'])} detected instances")
        
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Extract predictions
        labels = results['labels']
        boxes = results['boxes']
        scores = results['scores']
        masks = results['masks']
        
        # Draw each instance
        for i, (label, box, score, mask) in enumerate(zip(labels, boxes, scores, masks)):
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            if show_boxes:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # Draw contour from mask
            if show_masks and mask is not None and mask.size > 0:
                # Find contours in the mask
                mask_uint8 = (mask[0] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                for contour in contours:
                    if len(contour) > 2:
                        # Convert contour to list of points
                        points = []
                        for point in contour:
                            x, y = point[0]
                            points.extend([int(x), int(y)])
                        
                        if len(points) >= 6:  # At least 3 points (3*2 coordinates)
                            draw.polygon(points, outline=color, width=line_width)
            
            # Draw label and score
            if show_labels:
                x1, y1, x2, y2 = box
                text = f"Class {label}: {score:.3f}"
                # Get text bounding box for background
                bbox = draw.textbbox((0, 0), text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw text background
                text_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
                draw.rectangle(text_bg, fill=color)
                draw.text((x1 + 2, y1 - text_height - 2), text, fill="white")
        
        # Save if output path is provided
        if output_path:
            annotated_image.save(output_path, quality=95)
            print(f"Annotated image saved to: {output_path}")
        
        return annotated_image
