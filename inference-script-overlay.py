#!/usr/bin/env python3
"""
Inference script for fine-tuned SAM 2.1 model that overlays masks in different colors
"""

import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import random

# Import from SAM 2.1 repo
import sys
sys.path.append("./")  # Add SAM 2.1 repo to path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser(description='SAM 2.1 inference with colored mask overlay')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned SAM 2.1 checkpoint')
    parser.add_argument('--model_config', type=str, default='sam2_hiera_b.yaml', help='Path to model config file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--point_mode', action='store_true', help='Use point prompt mode')
    parser.add_argument('--grid_size', type=int, default=32, help='Grid size for automatic point sampling')
    parser.add_argument('--num_points', type=int, default=1, help='Number of points to try per grid cell')
    parser.add_argument('--transparency', type=float, default=0.5, help='Transparency of mask overlay (0-1)')
    parser.add_argument('--score_threshold', type=float, default=0.8, help='Score threshold for keeping masks')
    parser.add_argument('--nms_threshold', type=float, default=0.7, help='IoU threshold for NMS')
    return parser.parse_args()

def load_model(checkpoint_path, model_config, device):
    """Load SAM 2.1 model."""
    model = build_sam2(model_config, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()
    return predictor

def generate_colors(num_colors):
    """Generate distinct colors for visualization."""
    colors = []
    for i in range(num_colors):
        # Use HSV color space for more distinct colors
        hue = i / num_colors
        saturation = 0.9
        value = 0.9
        rgb_color = hsv_to_rgb((hue, saturation, value))
        # Convert to 0-255 range
        color = tuple(int(c * 255) for c in rgb_color)
        colors.append(color)
    return colors

def apply_non_max_suppression(masks, scores, iou_threshold=0.7):
    """Apply non-maximum suppression to remove overlapping masks."""
    if len(masks) == 0:
        return [], []
        
    # Convert masks to binary
    binary_masks = masks > 0.5
    
    # Compute areas
    areas = np.sum(binary_masks, axis=(1, 2))
    
    # Sort by score
    order = np.argsort(-scores)
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU with remaining masks
        mask_i = binary_masks[i]
        area_i = areas[i]
        
        remaining = order[1:]
        if remaining.size == 0:
            break
            
        ious = []
        for j in remaining:
            mask_j = binary_masks[j]
            area_j = areas[j]
            
            # Compute intersection
            intersection = np.logical_and(mask_i, mask_j).sum()
            
            # Compute union
            union = area_i + area_j - intersection
            
            # Compute IoU
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        # Keep masks with IoU below threshold
        inds = np.where(np.array(ious) <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [masks[i] for i in keep], [scores[i] for i in keep]

def overlay_masks(image, masks, scores, transparency=0.5):
    """Overlay masks on the original image with different colors."""
    # Generate colors for masks
    colors = generate_colors(len(masks))
    
    # Create a copy of the original image
    overlay = image.copy()
    
    # Create a blank image for the mask overlay
    mask_overlay = np.zeros_like(image, dtype=np.float32)
    
    # Overlay each mask with a different color
    for i, (mask, score) in enumerate(zip(masks, scores)):
        color = colors[i]
        
        # Create colored mask
        colored_mask = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            colored_mask[:,:,c] = mask * color[c]
        
        # Add to overlay
        mask_overlay = np.where(
            np.expand_dims(mask, axis=2) > 0.5,
            mask_overlay * (1 - transparency) + colored_mask * transparency,
            mask_overlay
        )
    
    # Combine original image with mask overlay
    result = np.clip(image + mask_overlay, 0, 255).astype(np.uint8)
    
    return result

def sample_points_on_grid(image, grid_size=32, points_per_cell=1):
    """Sample points on a grid in the image."""
    height, width = image.shape[:2]
    
    all_points = []
    
    # Calculate grid step
    h_step = height // grid_size
    w_step = width // grid_size
    
    # Ensure at least 1 pixel between points
    h_step = max(1, h_step)
    w_step = max(1, w_step)
    
    # Sample points
    for h in range(0, height, h_step):
        for w in range(0, width, w_step):
            for _ in range(points_per_cell):
                # Sample randomly within the grid cell
                x = w + random.randint(0, min(w_step - 1, width - w - 1))
                y = h + random.randint(0, min(h_step - 1, height - h - 1))
                
                # Add point
                all_points.append([x, y])
    
    # Convert to numpy array
    points = np.array(all_points)
    
    return points

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading SAM 2.1 model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor = load_model(args.checkpoint, args.model_config, device)
    
    # Read image
    print(f"Processing image: {args.image_path}")
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image)
    
    # Get all masks
    all_masks = []
    all_scores = []
    
    if args.point_mode:
        # Sample points on a grid
        print(f"Sampling points on {args.grid_size}x{args.grid_size} grid...")
        points = sample_points_on_grid(image, args.grid_size, args.num_points)
        
        # For each point, get a mask
        for i, point in enumerate(points):
            if i % 100 == 0:
                print(f"Processing point {i+1}/{len(points)}...")
                
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([point]),
                    point_labels=np.array([1]),  # Foreground
                    multimask_output=True
                )
                
                # Keep only masks with high scores
                for mask, score in zip(masks, scores):
                    if score > args.score_threshold:
                        all_masks.append(mask)
                        all_scores.append(score)
            except Exception as e:
                print(f"Error processing point {point}: {e}")
                continue
    else:
        # Get masks for the entire image
        print("Generating masks for the entire image...")
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True
        )
        
        # For simplicity, assign a score of 1.0 to all masks
        scores = np.ones(len(masks))
        
        all_masks = masks
        all_scores = scores
    
    # Apply non-maximum suppression to remove overlapping masks
    print("Applying non-maximum suppression...")
    filtered_masks, filtered_scores = apply_non_max_suppression(
        all_masks, all_scores, iou_threshold=args.nms_threshold
    )
    
    print(f"Found {len(filtered_masks)} masks after filtering")
    
    # Overlay masks on original image
    print("Creating mask overlay...")
    result = overlay_masks(image, filtered_masks, filtered_scores, args.transparency)
    
    # Save output image
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save as PNG directly
    cv2.imwrite(output_path.replace('.jpg', '.png').replace('.jpeg', '.png'), 
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    print(f"Output saved to {output_path}")
    
    # Save individual masks for potential further use
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    for i, (mask, score) in enumerate(zip(filtered_masks, filtered_scores)):
        mask_path = os.path.join(masks_dir, f"{os.path.basename(args.image_path).split('.')[0]}_mask_{i}_{score:.2f}.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

if __name__ == "__main__":
    main()
