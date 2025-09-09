import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import cv2
from PIL import Image
import torch.nn.functional as F
from typing import List, Dict, Tuple

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.viz_dir = config.exp_viz_dir
        self.threshold = config.viz_threshold
        
        # Create visualization directory
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Denormalization parameters
        self.mean = np.array(config.normalize_mean)
        self.std = np.array(config.normalize_std)
    
    def denormalize_image(self, tensor_image):
        """Denormalize image tensor to [0, 1] range"""
        if isinstance(tensor_image, torch.Tensor):
            image = tensor_image.cpu().numpy()
        else:
            image = tensor_image
            
        # Handle different input formats
        if image.ndim == 4:  # Batch dimension
            image = image[0]
        if image.shape[0] == 3:  # CHW format
            image = image.transpose(1, 2, 0)
        
        # Denormalize
        image = image * self.std + self.mean
        image = np.clip(image, 0, 1)
        return image
    
    def masks_to_contours(self, masks, threshold=0.5):
        """Convert masks to contours for visualization"""
        contours = []
        
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        for mask in masks:
            if mask.max() > threshold:
                # Threshold the mask
                binary_mask = (mask > threshold).astype(np.uint8)
                
                # Find contours
                contours_cv, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Convert to matplotlib format
                for contour in contours_cv:
                    if len(contour) > 2:  # Need at least 3 points for a contour
                        contour_points = contour.squeeze()
                        if contour_points.ndim == 2:
                            contours.append(contour_points)
        
        return contours
    
    def draw_contours_on_image(self, image, contours, color='red', linewidth=2):
        """Draw contours on image"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.set_title("Segmentation Contours")
        ax.axis('off')
        
        # Draw contours
        for contour in contours:
            if len(contour) > 2:
                # Close the contour
                contour_closed = np.vstack([contour, contour[0:1]])
                ax.plot(contour_closed[:, 0], contour_closed[:, 1], 
                       color=color, linewidth=linewidth)
        
        return fig, ax
    
    def create_visualization(self, image, target, prediction, image_name, threshold=None):
        """Create a visualization with original image, ground truth, and predictions"""
        if threshold is None:
            threshold = self.threshold
        
        # Denormalize image
        orig_image = self.denormalize_image(image)
        
        # Prepare ground truth masks
        gt_masks = target['masks'].cpu().numpy() if isinstance(target['masks'], torch.Tensor) else target['masks']
        gt_contours = self.masks_to_contours(gt_masks, threshold=0.5)
        
        # Prepare predictions
        if 'pred_masks' in prediction:
            pred_masks = prediction['pred_masks']
            pred_logits = prediction['pred_logits']
            
            # Apply sigmoid to masks and softmax to logits
            if isinstance(pred_masks, torch.Tensor):
                pred_masks = torch.sigmoid(pred_masks).cpu().numpy()
            if isinstance(pred_logits, torch.Tensor):
                pred_scores = torch.softmax(pred_logits, dim=-1).cpu().numpy()
            else:
                pred_scores = pred_logits
            
            # Filter predictions by confidence (class 1 score > threshold)
            if pred_scores.shape[-1] > 1:  # Has no-object class
                class_scores = pred_scores[:, 0]  # Class 1 scores (index 0 since we have single class)
            else:
                class_scores = pred_scores.squeeze()
            
            high_conf_idx = class_scores > threshold
            filtered_masks = pred_masks[high_conf_idx]
            pred_contours = self.masks_to_contours(filtered_masks, threshold=threshold)
        else:
            pred_contours = []
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Visualization: {image_name}", fontsize=16)
        
        # Original image
        axes[0].imshow(orig_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(orig_image)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        for contour in gt_contours:
            if len(contour) > 2:
                contour_closed = np.vstack([contour, contour[0:1]])
                axes[1].plot(contour_closed[:, 0], contour_closed[:, 1], 
                           color='green', linewidth=2, label='GT')
        
        # Predictions  
        axes[2].imshow(orig_image)
        axes[2].set_title(f"Predictions (threshold={threshold:.2f})")
        axes[2].axis('off')
        for contour in pred_contours:
            if len(contour) > 2:
                contour_closed = np.vstack([contour, contour[0:1]])
                axes[2].plot(contour_closed[:, 0], contour_closed[:, 1], 
                           color='red', linewidth=2, label='Pred')
        
        plt.tight_layout()
        return fig
    
    def save_batch_visualizations(self, images, targets, predictions, image_names, epoch, num_samples=None):
        """Save visualizations for a batch of images"""
        if num_samples is None:
            num_samples = min(len(images), self.config.num_viz_samples)
        
        num_samples = min(num_samples, len(images))
        
        for i in range(num_samples):
            try:
                # Get image name without extension
                img_name = os.path.splitext(image_names[i])[0] if isinstance(image_names[i], str) else f"image_{i}"
                
                # Create visualization
                fig = self.create_visualization(
                    images[i], 
                    targets[i], 
                    {k: v[i] if isinstance(v, torch.Tensor) and len(v.shape) > 1 else v for k, v in predictions.items()},
                    img_name
                )
                
                # Save figure
                save_path = os.path.join(
                    self.viz_dir, 
                    f"{img_name}_epoch{epoch:03d}_{i:02d}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved visualization: {save_path}")
                
            except Exception as e:
                print(f"Error creating visualization for image {i}: {str(e)}")
                continue
    
    def save_predictions_only(self, image, prediction, image_name, epoch, sample_idx):
        """Save only prediction visualization"""
        try:
            orig_image = self.denormalize_image(image)
            
            if 'pred_masks' in prediction:
                pred_masks = prediction['pred_masks']
                pred_logits = prediction['pred_logits']
                
                # Apply sigmoid to masks and softmax to logits
                if isinstance(pred_masks, torch.Tensor):
                    pred_masks = torch.sigmoid(pred_masks).cpu().numpy()
                if isinstance(pred_logits, torch.Tensor):
                    pred_scores = torch.softmax(pred_logits, dim=-1).cpu().numpy()
                else:
                    pred_scores = pred_logits
                
                # Filter by confidence
                if pred_scores.shape[-1] > 1:
                    class_scores = pred_scores[:, 0]
                else:
                    class_scores = pred_scores.squeeze()
                
                high_conf_idx = class_scores > self.threshold
                filtered_masks = pred_masks[high_conf_idx]
                pred_contours = self.masks_to_contours(filtered_masks, threshold=self.threshold)
            else:
                pred_contours = []
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(orig_image)
            ax.set_title(f"Predictions: {image_name}")
            ax.axis('off')
            
            for contour in pred_contours:
                if len(contour) > 2:
                    contour_closed = np.vstack([contour, contour[0:1]])
                    ax.plot(contour_closed[:, 0], contour_closed[:, 1], 
                           color='red', linewidth=2)
            
            # Save
            save_path = os.path.join(
                self.viz_dir,
                f"{image_name}_epoch{epoch:03d}_{sample_idx:02d}_pred.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error saving prediction visualization: {str(e)}")

def get_image_names_from_targets(targets, image_dir):
    """Extract image names from targets"""
    image_names = []
    for target in targets:
        image_id = target['image_id'].item() if isinstance(target['image_id'], torch.Tensor) else target['image_id']
        # This is a placeholder - you might need to adjust based on your dataset structure
        image_names.append(f"image_{image_id}.jpg")
    return image_names
