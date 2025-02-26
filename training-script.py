#!/usr/bin/env python3
"""
Fine-tuning script for SAM 2.1 B+ model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime

# Import from SAM 2.1 repo
import sys
sys.path.append("./")  # Add SAM 2.1 repo to path
from modeling.build_sam2 import build_sam2
from modeling.utils import MetricLogger, SmoothedValue

# Import our custom dataset
from dataloader import get_dataloader

def dice_loss(pred, target):
    """Dice loss for binary segmentation."""
    smooth = 1.0
    
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    
    return loss.mean()

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    
    pt = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    
    loss = -alpha_t * ((1 - pt) ** gamma) * torch.log(pt + 1e-8)
    
    return loss.mean()

def boundary_loss(pred, target, kernel_size=5):
    """
    Boundary loss to focus on edge regions
    """
    # Create a boundary map from the target mask
    kernel = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
    kernel.weight.data.fill_(1.0)
    kernel.weight.requires_grad = False
    if torch.cuda.is_available():
        kernel = kernel.cuda()
    
    boundary = torch.abs(target - kernel(target))
    boundary = torch.clamp(boundary, 0, 1)
    
    # Apply boundary weight to the standard BCE loss
    pred = torch.sigmoid(pred)
    bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
    
    # Increase loss weight near boundaries
    weighted_bce = bce * (1 + 4 * boundary)
    
    return weighted_bce.mean()

def combined_loss(pred, target, dice_weight=1.0, focal_weight=1.0, boundary_weight=0.5):
    """Combined loss function."""
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)
    b_loss = boundary_loss(pred, target)
    
    return dice_weight * d_loss + focal_weight * f_loss + boundary_weight * b_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune SAM 2.1 model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to prepared dataset directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for checkpoints')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SAM 2.1 checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--prompt_type', type=str, default='box', choices=['box', 'point'], help='Prompt type')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Checkpoint save interval (epochs)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    return parser.parse_args()

def init_distributed(args):
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Using DDP without distributed training')
        args.distributed = False
        return

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank})', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    args.distributed = True

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize distributed training if needed
    if args.local_rank != -1:
        init_distributed(args)
        print(f"Running distributed training on rank {args.local_rank}")
    else:
        args.distributed = False
    
    # Initialize wandb if specified
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.init(
            project="sam2-finetune",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "prompt_type": args.prompt_type,
            }
        )
    
    # Set up device
    device = torch.device(f'cuda:{args.local_rank}' if args.local_rank != -1 else 'cuda:0')
    
    # Create model
    print("Creating model...")
    if args.prompt_type == 'box':
        prompt_encoder = 'box'
    else:
        prompt_encoder = 'point'
        
    model = build_sam2(
        checkpoint=args.checkpoint,
        prompt_encoder=prompt_encoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375]
    )
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        prompt_type=args.prompt_type
    )
    
    # Set up distributed training
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - linear warmup then cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader)  # 1 epoch warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Set up mixed precision training
    scaler = GradScaler() if args.fp16 else None
    
    # Training loop
    best_val_loss = float('inf')
    print("Starting training...")
    
    for epoch in range(args.epochs):
        # Set to training mode
        model.train()
        
        # Setup metric logger
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', SmoothedValue(window_size=20, fmt='{value:.4f}'))
        
        header = f'Epoch: [{epoch+1}/{args.epochs}]'
        
        # Training epoch
        for i, batch in enumerate(metric_logger.log_every(train_loader, args.log_interval, header)):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
                            # Forward pass with mixed precision
            if args.fp16:
                with autocast():
                    # Forward pass providing image + prompt to get predicted mask
                    if args.prompt_type == 'box':
                        # Using bounding box prompts
                        outputs = model(images, boxes=prompts)
                    else:
                        # Using point prompts
                        points = prompts.unsqueeze(1)  # Add points dimension [B, 1, 2]
                        labels = torch.ones(points.shape[0], 1, device=device)  # All foreground points
                        outputs = model(images, points=points, labels=labels)
                    
                    # Compare predicted mask with ground truth mask using loss function
                    loss = combined_loss(outputs, masks)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                if args.prompt_type == 'box':
                    outputs = model(images, boxes=prompts)
                else:
                    points = prompts.unsqueeze(1)  # Add points dimension
                    labels = torch.ones(points.shape[0], 1, device=device)  # All foreground
                    outputs = model(images, points=points, labels=labels)
                
                loss = combined_loss(outputs, masks)
                
                # Standard backward pass
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update metrics
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            # Log to wandb
            if args.use_wandb and (not args.distributed or args.local_rank == 0) and i % args.log_interval == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + (i / len(train_loader))
                })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_steps = 0
        
        print("Running validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                prompts = batch['prompt'].to(device)
                
                if args.prompt_type == 'box':
                    outputs = model(images, boxes=prompts)
                else:
                    points = prompts.unsqueeze(1)
                    labels = torch.ones(points.shape[0], 1, device=device)
                    outputs = model(images, points=points, labels=labels)
                
                # Calculate loss
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
                
                # Calculate dice score
                pred_masks = torch.sigmoid(outputs) > 0.5
                intersection = (pred_masks * masks).sum((1, 2, 3))
                union = pred_masks.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                dice = (2.0 * intersection) / (union + 1e-8)
                val_dice += dice.mean().item()
                
                val_steps += 1
        
        val_loss /= val_steps
        val_dice /= val_steps
        
        # Log validation results
        print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if args.use_wandb and (not args.distributed or args.local_rank == 0):
            wandb.log({
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch": epoch + 1
            })
        
        # Save checkpoint
        if (not args.distributed or args.local_rank == 0) and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"sam2_finetuned_epoch{epoch+1}.pth")
            
            if args.distributed:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss and (not args.distributed or args.local_rank == 0):
            best_val_loss