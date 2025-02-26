#!/usr/bin/env python3
"""
Inference script for fine-tuned SAM 2.1 model
"""

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

# Import from SAM 2.1 repo
import sys
sys.path.append("./")  # Add SAM 2.1 repo to path
from modeling.build_sam2 import build_sam2

def parse_args():
    parser = argparse.ArgumentParser(description='SAM 2.1 inference on custom dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned SAM 2.1 checkpoint')
    parser.add_argument('--base_checkpoint', type=str, required=True, help='Path to original SAM 2.1 checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--prompt_type', type=str, default='box', choices=['box', 'point'], help='Prompt type')
    parser.add_argument('--prompt', type=str, help='Prompt coordinates (x1,y1,x2,y2 for box or x,y for point)')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Output directory')
    parser.add_argument('--compare', action='store_true', help='Compare fine-tuned model with base model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    return parser.parse_args()

def preprocess_image(image_path, target_size=1024):
    """Preprocess image for inference."""
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get original size
    h, w = image.shape[:2]
    
    # Resize image
    image_resized = cv2.resize(image, (target_size, target_size))
    
    # Normalize
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
    
    return image_tensor.unsqueeze(0), (h, w), image

def parse_prompt(prompt_str, prompt_type, image_size, orig_size):
    """Parse prompt string to tensor."""
    h_orig, w_orig = orig_size
    
    # Scale factors
    scale_x = image_size / w_orig
    scale_y = image_size / h_orig
    
    if prompt_type == 'box':
        # Parse x1,y1,x2,y2
        x1, y1, x2, y2 = map(float, prompt_str.split(','))
        
        # Scale to target size
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y
        
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
    else:
        # Parse x,y
        x, y = map(float, prompt_str.split(','))
        
        # Scale to target size
        x = x * scale_x
        y = y * scale_y
        
        return torch.tensor([[x, y]], dtype=torch.float32)

def load_model(checkpoint_path, base_checkpoint_path, prompt_type, device):
    """Load SAM 2.1 model."""
    # Determine prompt encoder based on prompt type
    prompt_encoder = 'box' if prompt_type == 'box' else 'point'
    
    # Load fine-tuned model
    model = build_sam2(
        checkpoint=base_checkpoint_path,  # First load the base model
        prompt_encoder=prompt_encoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375]
    )
    
    # Load fine-tuned weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def load_base_model(checkpoint_path, prompt_type, device):
    """Load base SAM 2.1 model for comparison."""
    prompt_encoder = 'box' if prompt_type == 'box' else 'point'
    
    model = build_sam2(
        checkpoint=checkpoint_path,
        prompt_encoder=prompt_encoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375]
    )
    
    model = model.to(device)
    model.eval()
    
    return model

def run_inference(model, image_tensor, prompt_tensor, prompt_type, device):
    """Run inference with SAM 2.1 model."""
    image_tensor = image_tensor.to(device)
    prompt_tensor = prompt_tensor.to(device)
    
    with torch.no_grad():
        if prompt_type == 'box':
            outputs = model(image_tensor, boxes=prompt_tensor)
        else:
            points = prompt_tensor.reshape(-1, 1, 2)  # [B, 1, 2]
            labels = torch.ones(points.shape[0], 1, device=device)  # All foreground
            outputs = model(image_tensor, points=points, labels=labels)
    
    # Apply sigmoid to get probability map
    pred_mask = torch.sigmoid(outputs) > 0.5
    
    return pred_mask.cpu().numpy()

def visualize_results(image, mask, prompt, prompt_type, output_path):
    """Visualize segmentation results."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image with prompt
    ax1.imshow(image)
    ax1.set_title("Input Image with Prompt")
    
    # Add prompt visualization
    if prompt_type == 'box':
        x1, y1, x2, y2 = prompt[0]
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    else:
        ax1.scatter(prompt[0][0], prompt[0][1], c='r', s=40)
    
    # Plot mask
    ax2.imshow(image)
    mask_vis = np.ma.masked_where(mask[0, 0] == 0, mask[0, 0])
    ax2.imshow(mask_vis, alpha=0.5, cmap='jet')
    ax2.set_title("Predicted Mask")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_models(fine_tuned_mask, base_mask, image, prompt, prompt_type, output_path):
    """Compare results from fine-tuned and base models."""
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image with prompt
    ax1.imshow(image)
    ax1.set_title("Input Image with Prompt")
    
    # Add prompt visualization
    if prompt_type == 'box':
        x1, y1, x2, y2 = prompt[0]
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    else:
        ax1.scatter(prompt[0][0], prompt[0][1], c='r', s=40)
    
    # Plot base model mask
    ax2.imshow(image)
    base_mask_vis = np.ma.masked_where(base_mask[0, 0] == 0, base_mask[0, 0])
    ax2.imshow(base_mask_vis, alpha=0.5, cmap='cool')
    ax2.set_title("Base Model Mask")
    
    # Plot fine-tuned model mask
    ax3.imshow(image)
    ft_mask_vis = np.ma.masked_where(fine_tuned_mask[0, 0] == 0, fine_tuned_mask[0, 0])
    ax3.imshow(ft_mask_vis, alpha=0.5, cmap='jet')
    ax3.set_title("Fine-tuned Model Mask")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Preprocess image
    image_tensor, orig_size, original_image = preprocess_image(args.image_path, args.image_size)
    
    # Parse prompt
    if args.prompt:
        prompt_tensor = parse_prompt(args.prompt, args.prompt_type, args.image_size, orig_size)
    else:
        # If no prompt provided, use default prompt (center box or point)
        h, w = orig_size
        if args.prompt_type == 'box':
            # Default box covering central 50% of the image
            cx, cy = w / 2, h / 2
            box_w, box_h = w / 4, h / 4
            prompt_tensor = parse_prompt(f"{cx-box_w},{cy-box_h},{cx+box_w},{cy+box_h}", 
                                         args.prompt_type, args.image_size, orig_size)
        else:
            # Default point at center of the image
            prompt_tensor = parse_prompt(f"{w/2},{h/2}", args.prompt_type, args.image_size, orig_size)
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    fine_tuned_model = load_model(args.checkpoint, args.base_checkpoint, args.prompt_type, device)
    
    # Run inference with fine-tuned model
    print("Running inference with fine-tuned model...")
    fine_tuned_mask = run_inference(fine_tuned_model, image_tensor, prompt_tensor, args.prompt_type, device)
    
    # Compare with base model if requested
    if args.compare:
        print("Loading base model for comparison...")
        base_model = load_base_model(args.base_checkpoint, args.prompt_type, device)
        
        print("Running inference with base model...")
        base_mask = run_inference(base_model, image_tensor, prompt_tensor, args.prompt_type, device)
        
        # Visualize comparison
        output_path = os.path.join(args.output_dir, "comparison_result.png")
        compare_models(fine_tuned_mask, base_mask, original_image, prompt_tensor.cpu().numpy(), 
                      args.prompt_type, output_path)
        print(f"Comparison result saved to {output_path}")
    else:
        # Visualize fine-tuned model result only
        output_path = os.path.join(args.output_dir, "inference_result.png")
        visualize_results(original_image, fine_tuned_mask, prompt_tensor.cpu().numpy(), 
                         args.prompt_type, output_path)
        print(f"Result saved to {output_path}")
    
    # Save mask as binary file
    mask_path = os.path.join(args.output_dir, "mask.png")
    cv2.imwrite(mask_path, (fine_tuned_mask[0, 0] * 255).astype(np.uint8))
    print(f"Binary mask saved to {mask_path}")
    
    # Save information about the inference
    info = {
        "image_path": args.image_path,
        "prompt_type": args.prompt_type,
        "prompt": args.prompt,
        "checkpoint": args.checkpoint,
        "image_size": args.image_size,
        "device": args.device
    }
    
    with open(os.path.join(args.output_dir, "inference_info.json"), 'w') as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":
    main()
