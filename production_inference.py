#!/usr/bin/env python3
"""
Production-level inference script for instance segmentation.

Usage:
    from inference import InstanceSegmentationInference
    
    # Initialize
    model = InstanceSegmentationInference(
        checkpoint_path="checkpoints/best_model.ckpt",
        device="cuda"
    )
    
    # Inference
    results = model.predict(pil_image, confidence_threshold=0.5)
    
    # Visualize (optional)
    model.visualize_results(pil_image, results, save_path="output.png")
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

from training.mask_classification_instance import MaskClassificationInstance


class InstanceSegmentationInference:
    """Production-ready inference class for instance segmentation."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize the inference model.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (.ckpt)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            img_size: Input image size for the model
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # Load model
        self.model = MaskClassificationInstance.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            strict=False
        )
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
        print(f"📐 Input size: {self.img_size}")
    
    def _preprocess_image(self, pil_image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess PIL image for model input.
        
        Args:
            pil_image: Input PIL image
            
        Returns:
            Tuple of (preprocessed_tensor, original_size)
        """
        # Store original size
        original_size = pil_image.size  # (width, height)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to tensor and normalize to [0, 1]
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)
        
        # Resize and pad to model input size
        img_tensor = self._resize_and_pad(img_tensor)
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, original_size
    
    def _resize_and_pad(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Resize and pad image to target size while maintaining aspect ratio."""
        # Calculate scale factor to fit within target size
        scale_factor = min(
            self.img_size[0] / img_tensor.shape[-2],  # height
            self.img_size[1] / img_tensor.shape[-1],  # width
        )
        
        # Calculate new size
        new_h = int(img_tensor.shape[-2] * scale_factor)
        new_w = int(img_tensor.shape[-1] * scale_factor)
        
        # Resize
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )[0]
        
        # Pad to target size
        pad_h = max(0, self.img_size[0] - new_h)
        pad_w = max(0, self.img_size[1] - new_w)
        padding = [0, 0, pad_w, pad_h]  # [pad_left, pad_top, pad_right, pad_bottom]
        
        img_tensor = F.pad(img_tensor, padding)
        
        return img_tensor
    
    def _postprocess_predictions(
        self,
        mask_logits: List[torch.Tensor],
        class_logits: List[torch.Tensor],
        original_size: Tuple[int, int],
        confidence_threshold: float = 0.5,
        max_instances: int = 100
    ) -> Dict:
        """
        Convert model outputs to final predictions.
        
        Args:
            mask_logits: List of mask predictions per layer
            class_logits: List of class predictions per layer
            original_size: Original image size (width, height)
            confidence_threshold: Minimum confidence for predictions
            max_instances: Maximum number of instances to return
            
        Returns:
            Dictionary with masks, labels, scores, and bboxes
        """
        # Use final layer predictions
        final_mask_logits = mask_logits[-1]  # (1, num_queries, H, W)
        final_class_logits = class_logits[-1]  # (1, num_queries, num_classes+1)
        
        # Remove batch dimension
        final_mask_logits = final_mask_logits[0]  # (num_queries, H, W)
        final_class_logits = final_class_logits[0]  # (num_queries, num_classes+1)
        
        # Resize masks to original image size
        final_mask_logits = F.interpolate(
            final_mask_logits.unsqueeze(0),
            size=(original_size[1], original_size[0]),  # PIL size is (w, h), tensor is (h, w)
            mode='bilinear',
            align_corners=False
        )[0]
        
        # Get class probabilities (exclude background class)
        class_probs = final_class_logits.softmax(dim=-1)[:, :-1]  # (num_queries, num_classes)
        
        # Get top predictions across all queries and classes
        scores_flat = class_probs.flatten()
        topk_scores, topk_indices = scores_flat.topk(
            min(max_instances, len(scores_flat)), sorted=False
        )
        
        # Convert flat indices back to query and class indices
        num_classes = class_probs.shape[1]
        query_indices = topk_indices // num_classes
        class_indices = topk_indices % num_classes
        
        # Get corresponding masks
        pred_masks = final_mask_logits[query_indices] > 0  # Binary masks
        pred_labels = class_indices
        pred_scores = topk_scores
        
        # Calculate mask quality scores
        mask_probs = final_mask_logits[query_indices].sigmoid()
        mask_scores = (mask_probs * pred_masks.float()).sum(dim=[1, 2]) / (pred_masks.sum(dim=[1, 2]) + 1e-6)
        
        # Combine class and mask scores
        final_scores = pred_scores * mask_scores
        
        # Filter by confidence
        keep_mask = final_scores > confidence_threshold
        pred_masks = pred_masks[keep_mask]
        pred_labels = pred_labels[keep_mask]
        final_scores = final_scores[keep_mask]
        
        # Sort by score
        if len(final_scores) > 0:
            sorted_indices = final_scores.argsort(descending=True)
            pred_masks = pred_masks[sorted_indices]
            pred_labels = pred_labels[sorted_indices]
            final_scores = final_scores[sorted_indices]
        
        # Convert to numpy for easier handling
        pred_masks = pred_masks.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        final_scores = final_scores.cpu().numpy()
        
        # Calculate bounding boxes
        bboxes = []
        for mask in pred_masks:
            if mask.sum() > 0:
                coords = np.where(mask)
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                bboxes.append([x_min, y_min, x_max, y_max])
            else:
                bboxes.append([0, 0, 0, 0])
        
        return {
            'masks': pred_masks,
            'labels': pred_labels,
            'scores': final_scores,
            'bboxes': np.array(bboxes),
            'num_instances': len(pred_masks)
        }
    
    @torch.no_grad()
    def predict(
        self,
        pil_image: Image.Image,
        confidence_threshold: float = 0.5,
        max_instances: int = 100
    ) -> Dict:
        """
        Run inference on a PIL image.
        
        Args:
            pil_image: Input PIL image
            confidence_threshold: Minimum confidence for predictions
            max_instances: Maximum number of instances to return
            
        Returns:
            Dictionary containing:
                - masks: (N, H, W) boolean array of instance masks
                - labels: (N,) array of class labels
                - scores: (N,) array of confidence scores
                - bboxes: (N, 4) array of bounding boxes [x_min, y_min, x_max, y_max]
                - num_instances: Number of detected instances
        """
        # Preprocess
        img_tensor, original_size = self._preprocess_image(pil_image)
        
        # Inference
        mask_logits, class_logits = self.model(img_tensor)
        
        # Postprocess
        results = self._postprocess_predictions(
            mask_logits, class_logits, original_size, confidence_threshold, max_instances
        )
        
        return results
    
    def visualize_results(
        self,
        pil_image: Image.Image,
        results: Dict,
        save_path: Optional[str] = None,
        show_labels: bool = True,
        show_scores: bool = True,
        alpha: float = 0.5
    ) -> Optional[Image.Image]:
        """
        Visualize prediction results on the image.
        
        Args:
            pil_image: Original PIL image
            results: Prediction results from predict()
            save_path: Path to save the visualization (optional)
            show_labels: Whether to show class labels
            show_scores: Whether to show confidence scores
            alpha: Transparency of mask overlays
            
        Returns:
            PIL Image if save_path is None, otherwise None
        """
        # Convert PIL to numpy
        img_np = np.array(pil_image)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_np)
        
        # Get results
        masks = results['masks']
        labels = results['labels']
        scores = results['scores']
        bboxes = results['bboxes']
        
        # Generate colors
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(masks), 1)))
        
        # Draw each instance
        for i, (mask, label, score, bbox) in enumerate(zip(masks, labels, scores, bboxes)):
            if mask.sum() > 0:
                color = colors[i % len(colors)]
                
                # Draw mask
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask] = color
                colored_mask[:, :, 3] = alpha * mask
                ax.imshow(colored_mask)
                
                # Draw bounding box
                x_min, y_min, x_max, y_max = bbox
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add text label
                text_parts = []
                if show_labels:
                    text_parts.append(f'Class {label}')
                if show_scores:
                    text_parts.append(f'{score:.2f}')
                
                if text_parts:
                    text = ': '.join(text_parts)
                    ax.text(
                        x_min, y_min - 5, text,
                        color=color, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                    )
        
        ax.set_title(f'Instance Segmentation Results ({len(masks)} instances)')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"💾 Visualization saved: {save_path}")
            return None
        else:
            # Convert to PIL Image
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return Image.fromarray(img_array)


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = InstanceSegmentationInference(
        checkpoint_path="checkpoints/best_model.ckpt",
        device="cuda"
    )
    
    # Load image
    image = Image.open("path/to/your/image.jpg")
    
    # Run inference
    results = model.predict(
        image,
        confidence_threshold=0.5,
        max_instances=50
    )
    
    print(f"Detected {results['num_instances']} instances")
    print(f"Labels: {results['labels']}")
    print(f"Scores: {results['scores']}")
    
    # Visualize and save
    model.visualize_results(
        image,
        results,
        save_path="output_visualization.png"
    )
    
    # Or get PIL image for further processing
    viz_image = model.visualize_results(image, results)
    # viz_image.show()  # Display
