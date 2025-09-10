import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    return image

def mask_to_contours(mask, threshold=0.5):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    binary_mask = (mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def save_predictions(images, targets, outputs, config, epoch, save_dir, img_names=None):
    os.makedirs(save_dir, exist_ok=True)
    
    images = images.cpu()
    
    for idx in range(min(len(images), config.num_viz)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize image
        img = denormalize_image(images[idx], config.normalize_mean, config.normalize_std)
        img = img.permute(1, 2, 0).numpy()
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(img)
        if targets[idx]['masks'].shape[0] > 0:
            for mask in targets[idx]['masks']:
                contours = mask_to_contours(mask)
                for contour in contours:
                    axes[1].plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predictions
        axes[2].imshow(img)
        pred_scores = outputs['pred_logits'][idx].softmax(-1)[:, :-1].max(-1)[0]
        pred_masks = outputs['pred_masks'][idx]
        
        keep = pred_scores > config.viz_threshold
        if keep.any():
            pred_masks = pred_masks[keep]
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.unsqueeze(0),
                size=img.shape[:2],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            for mask in pred_masks:
                mask = mask.sigmoid()
                contours = mask_to_contours(mask)
                for contour in contours:
                    axes[2].plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
        
        axes[2].set_title(f'Predictions (threshold={config.viz_threshold})')
        axes[2].axis('off')
        
        # Save figure
        if img_names and idx < len(img_names):
            filename = f"{img_names[idx]}_{epoch}_{idx}.png"
        else:
            filename = f"img_{epoch}_{idx}.png"
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
        plt.close()

def visualize_batch(images, targets, outputs, config):
    batch_size = images.shape[0]
    n_display = min(4, batch_size)
    
    fig, axes = plt.subplots(n_display, 3, figsize=(12, 4*n_display))
    if n_display == 1:
        axes = axes.reshape(1, -1)
    
    images = images.cpu()
    
    for idx in range(n_display):
        # Denormalize image
        img = denormalize_image(images[idx], config.normalize_mean, config.normalize_std)
        img = img.permute(1, 2, 0).numpy()
        
        # Original
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Input')
        axes[idx, 0].axis('off')
        
        # Ground truth
        axes[idx, 1].imshow(img)
        if targets[idx]['masks'].shape[0] > 0:
            for mask in targets[idx]['masks']:
                contours = mask_to_contours(mask)
                for contour in contours:
                    axes[idx, 1].plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2)
        axes[idx, 1].set_title('GT')
        axes[idx, 1].axis('off')
        
        # Predictions
        axes[idx, 2].imshow(img)
        pred_scores = outputs['pred_logits'][idx].softmax(-1)[:, :-1].max(-1)[0]
        pred_masks = outputs['pred_masks'][idx]
        
        keep = pred_scores > config.viz_threshold
        if keep.any():
            pred_masks = pred_masks[keep]
            for mask in pred_masks:
                mask = mask.sigmoid()
                contours = mask_to_contours(mask)
                for contour in contours:
                    axes[idx, 2].plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
        
        axes[idx, 2].set_title('Predictions')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    return fig