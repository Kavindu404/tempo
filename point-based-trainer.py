#!/usr/bin/env python3
"""
Fine-tuning script for SAM 2.1 B+ model on COCO-format dataset using point prompts
Adapted from the LabPics implementation
"""

import os
import json
import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import random
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# Import from SAM 2.1 repo
import sys
sys.path.append("./")  # Add SAM 2.1 repo to path

# Assuming the SAM 2.1 model is imported according to your repository structure
# This matches the imports in the provided snippet
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pycocotools import mask as mask_utils


def parse_args():
    parser = argparse.ArgumentParser(description='SAM 2.1 fine-tuning using point prompts')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--annotation_file', type=str, default='annotations/640.json', help='Path to annotation file relative to dataset_dir')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for checkpoints')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SAM 2.1 checkpoint')
    parser.add_argument('--model_config', type=str, default='sam2_hiera_b.yaml', help='Path to model config file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='Weight decay')
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint save interval (iterations)')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--num_points', type=int, default=1, help='Number of points per mask')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize', action='store_true', help='Visualize training samples and predictions')
    return parser.parse_args()


def polygon_to_mask(segmentation, height, width):
    """Convert polygon segmentation to binary mask."""
    if isinstance(segmentation, list) and len(segmentation) > 0:
        if isinstance(segmentation[0], list):  # Multiple polygons
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        else:  # Single polygon
            rle = mask_utils.frPyObjects([segmentation], height, width)[0]
        mask = mask_utils.decode(rle)
        return mask
    else:
        # Return an empty mask if segmentation is empty
        return np.zeros((height, width), dtype=np.uint8)


def prepare_data(dataset_dir, annotation_file):
    """Load and prepare dataset from COCO format."""
    data = []
    
    # Load annotations
    with open(os.path.join(dataset_dir, annotation_file), 'r') as f:
        coco_data = json.load(f)
    
    print(f"Loaded {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Create image ID to file name and size mapping
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    image_id_to_size = {img['id']: (img['height'], img['width']) for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Create category mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Prepare data entries
    for image_id, annotations in annotations_by_image.items():
        file_name = image_id_to_file[image_id]
        height, width = image_id_to_size[image_id]
        
        # Determine image path based on whether it's in train or val
        if os.path.exists(os.path.join(dataset_dir, 'images/train', file_name)):
            image_path = os.path.join(dataset_dir, 'images/train', file_name)
        elif os.path.exists(os.path.join(dataset_dir, 'images/val', file_name)):
            image_path = os.path.join(dataset_dir, 'images/val', file_name)
        else:
            print(f"Warning: Could not find image {file_name} for image_id {image_id}")
            continue
        
        # Add entry for each annotation (mask) in the image
        for ann in annotations:
            # Convert segmentation to mask
            mask = polygon_to_mask(ann['segmentation'], height, width)
            
            # Skip if mask is empty
            if mask.sum() == 0:
                continue
            
            data.append({
                'image_path': image_path,
                'segmentation': ann['segmentation'],
                'height': height,
                'width': width,
                'category_id': ann['category_id'],
                'category_name': category_id_to_name[ann['category_id']],
                'bbox': ann['bbox']
            })
    
    print(f"Prepared {len(data)} data entries")
    return data


def read_single(data_entry):
    """Read and prepare a single data entry."""
    # Read image
    image = cv2.imread(data_entry['image_path'])[..., ::-1]  # BGR to RGB
    
    # Generate mask from segmentation
    height, width = data_entry['height'], data_entry['width']
    mask = polygon_to_mask(data_entry['segmentation'], height, width)
    
    # Resize image and mask to target size (1024x1024 as in the provided code)
    r = min(1024 / width, 1024 / height)
    new_height, new_width = int(height * r), int(width * r)
    
    image = cv2.resize(image, (new_width, new_height))
    mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Pad to 1024x1024
    if new_height < 1024:
        image = np.concatenate([image, np.zeros([1024 - new_height, new_width, 3], dtype=np.uint8)], axis=0)
        mask = np.concatenate([mask, np.zeros([1024 - new_height, new_width], dtype=np.uint8)], axis=0)
    if new_width < 1024:
        image = np.concatenate([image, np.zeros([1024, 1024 - new_width, 3], dtype=np.uint8)], axis=1)
        mask = np.concatenate([mask, np.zeros([1024, 1024 - new_width], dtype=np.uint8)], axis=1)
    
    # Get random points inside the mask
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        # Skip empty masks
        return None, None, None
    
    # Sample random points inside the mask
    point_indices = np.random.choice(len(coords), min(args.num_points, len(coords)), replace=False)
    points = []
    for idx in point_indices:
        yx = coords[idx]
        points.append([yx[1], yx[0]])  # Convert to [x, y] format
    
    return image, mask, points


def read_batch(data, batch_size=4, num_retries=10):
    """Read and prepare a batch of data."""
    limage = []
    lmask = []
    lpoints = []
    
    # Randomly sample data entries
    random_indices = np.random.choice(len(data), size=batch_size * 2, replace=False)
    
    for idx in random_indices:
        if len(limage) >= batch_size:
            break
            
        image, mask, points = read_single(data[idx])
        
        if image is None or mask is None or points is None:
            continue
            
        limage.append(image)
        lmask.append(mask)
        lpoints.append(points)
    
    # If we couldn't get enough samples, try again with a smaller batch
    if len(limage) < batch_size and num_retries > 0:
        additional_images, additional_masks, additional_points, _ = read_batch(
            data, batch_size - len(limage), num_retries - 1
        )
        limage.extend(additional_images)
        lmask.extend(additional_masks)
        lpoints.extend(additional_points)
    
    # Create labels (all points are foreground)
    labels = np.ones([len(limage), args.num_points])
    
    return limage, np.array(lmask), np.array(lpoints), labels


def visualize_samples(image, mask, points, prediction=None, iou=None, iteration=0):
    """Visualize training samples and predictions."""
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3 if prediction is None else 4, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Image with points
    axes[1].imshow(image)
    for point in points:
        axes[1].scatter(point[0], point[1], c='red', s=40)
    axes[1].set_title("Image with Points")
    axes[1].axis('off')
    
    # Ground truth mask
    axes[2].imshow(image)
    mask_vis = np.ma.masked_where(mask == 0, mask)
    axes[2].imshow(mask_vis, alpha=0.5, cmap='jet')
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis('off')
    
    # Predicted mask (if provided)
    if prediction is not None:
        axes[3].imshow(image)
        pred_vis = np.ma.masked_where(prediction == 0, prediction)
        axes[3].imshow(pred_vis, alpha=0.5, cmap='cool')
        axes[3].set_title(f"Predicted Mask (IoU: {iou:.3f})")
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "visualizations", f"sample_{iteration}.png"))
    plt.close()


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing dataset...")
    data = prepare_data(args.dataset_dir, args.annotation_file)
    
    # Load model
    print("Loading SAM 2.1 model...")
    sam2_model = build_sam2(args.model_config, args.checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set training parameters
    predictor.model.sam_mask_decoder.train(True)  # Enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True)  # Enable training of prompt encoder
    predictor.model.image_encoder.train(True)  # Enable training of image encoder
    
    # Remove any no_grad context managers in the model to allow training the image encoder
    # This is a manual process that depends on the model implementation
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = GradScaler()  # Mixed precision
    
    # Training loop
    print("Starting training...")
    mean_iou = 0
    
    for itr in tqdm(range(args.iterations)):
        with autocast():  # Cast to mixed precision
            # Load data batch
            images, masks, input_points, input_labels = read_batch(data, batch_size=args.batch_size)
            
            if len(images) == 0:
                print("Warning: Empty batch, skipping iteration")
                continue
            
            # Set image batch for processing
            predictor.set_image_batch(images)
            
            # Prepare prompts
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
            )
            
            # Prompt encoding
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
                masks=None,
            )
            
            # Mask decoder
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"],
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            
            # Upscale masks to original resolution
            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )
            
            # Loss calculation
            gt_masks = torch.tensor(masks.astype(np.float32)).cuda()
            pred_masks = torch.sigmoid(prd_masks[:, 0])  # Take the first mask and apply sigmoid
            
            # Cross entropy loss
            seg_loss = (
                -gt_masks * torch.log(pred_masks + 1e-5)
                - (1 - gt_masks) * torch.log(1 - pred_masks + 1e-5)
            ).mean()
            
            # IOU score loss
            pred_binary = (pred_masks > 0.5).float()
            intersection = (gt_masks * pred_binary).sum((1, 2))
            union = gt_masks.sum((1, 2)) + pred_binary.sum((1, 2)) - intersection
            iou = intersection / (union + 1e-8)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            
            # Total loss
            loss = seg_loss + score_loss * 0.05
        
        # Apply back propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Save model checkpoint
        if (itr + 1) % args.save_interval == 0 or itr == args.iterations - 1:
            torch.save(
                predictor.model.state_dict(),
                os.path.join(args.output_dir, f"sam2_finetuned_iter{itr+1}.pth")
            )
            print(f"Saved checkpoint at iteration {itr+1}")
        
        # Calculate and log metrics
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        
        if (itr + 1) % args.log_interval == 0:
            print(f"Iteration {itr+1}/{args.iterations}, Loss: {loss.item():.4f}, Mean IoU: {mean_iou:.4f}")
        
        # Visualize samples
        if args.visualize and (itr + 1) % args.save_interval == 0:
            for i in range(min(2, len(images))):
                visualize_samples(
                    images[i], 
                    masks[i], 
                    input_points[i], 
                    prediction=(pred_binary[i] > 0.5).cpu().numpy(), 
                    iou=iou[i].item(),
                    iteration=itr+1
                )
    
    # Save final model
    torch.save(
        predictor.model.state_dict(),
        os.path.join(args.output_dir, "sam2_finetuned_final.pth")
    )
    print(f"Final model saved. Final Mean IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
