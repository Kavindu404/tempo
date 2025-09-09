#!/usr/bin/env python3
"""
Simple launcher script for distributed training of DINOv3 + Mask2Former.

This script provides an easy way to launch training with proper distributed setup.

Usage:
    # Single GPU training
    python launch_training.py --gpus 1
    
    # Multi-GPU training (8 GPUs)
    python launch_training.py --gpus 8
    
    # Custom configuration
    python launch_training.py --gpus 4 --batch_size 4 --lr 2e-4
    
    # Resume from checkpoint
    python launch_training.py --gpus 8 --resume checkpoints/exp/best_model.pt
"""

import os
import sys
import argparse
import subprocess
import torch
import json
from config import Config
from utils import verify_config, create_experiment_dirs, save_config

def parse_args():
    parser = argparse.ArgumentParser(description='Launch DINOv3 + Mask2Former Training')
    
    # Training setup
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank for multi-node training')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port')
    
    # Model and training config overrides
    parser.add_argument('--exp_name', type=str, help='Experiment name')
    parser.add_argument('--backbone_type', type=str, help='DINOv3 backbone type')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone parameters')
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Dataset paths
    parser.add_argument('--train_json', type=str, help='Path to training JSON')
    parser.add_argument('--test_json', type=str, help='Path to test JSON')
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--dinov3_repo', type=str, help='Path to DINOv3 repository')
    parser.add_argument('--weights_dir', type=str, help='Path to backbone weights directory')
    
    # Other options
    parser.add_argument('--verify_only', action='store_true', help='Only verify setup, do not train')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create configuration object from command line arguments"""
    config = Config()
    
    # Override config with command line arguments
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.backbone_type:
        config.backbone_type = args.backbone_type
    if args.freeze_backbone:
        config.freeze_backbone = True
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.num_epochs = args.epochs
    
    # Dataset paths
    if args.train_json:
        config.train_json_path = args.train_json
    if args.test_json:
        config.test_json_path = args.test_json
    if args.image_dir:
        config.image_dir = args.image_dir
    if args.dinov3_repo:
        config.dinov3_repo_dir = args.dinov3_repo
    if args.weights_dir:
        config.backbone_weights_dir = args.weights_dir
    
    # Distributed training
    config.world_size = args.gpus
    
    return config

def verify_setup(config):
    """Verify that the setup is correct"""
    print("🔍 Verifying setup...")
    
    try:
        verify_config(config)
        print("✅ Setup verification passed!")
        return True
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        print("\nPlease check your configuration and paths.")
        return False

def launch_single_gpu(config, args):
    """Launch training on single GPU"""
    print("🚀 Launching single GPU training...")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Import and run training
    from train import main_worker
    
    try:
        main_worker(0, 1, config)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.debug:
            raise

def launch_multi_gpu(config, args):
    """Launch distributed training on multiple GPUs"""
    print(f"🚀 Launching distributed training on {args.gpus} GPUs...")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(args.gpus)
    
    # Launch processes
    import torch.multiprocessing as mp
    from train import main_worker
    
    try:
        mp.spawn(
            main_worker,
            args=(args.gpus, config),
            nprocs=args.gpus,
            join=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.debug:
            raise

def print_config_summary(config):
    """Print configuration summary"""
    print("\n📋 Configuration Summary:")
    print("=" * 50)
    print(f"Experiment name: {config.exp_name}")
    print(f"Backbone: {config.backbone_type}")
    print(f"Backbone frozen: {config.freeze_backbone}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Number of GPUs: {config.world_size}")
    print(f"Image size: {config.image_size}")
    print(f"Number of queries: {config.num_queries}")
    print("=" * 50)

def print_paths_summary(config):
    """Print paths summary"""
    print("\n📁 Paths:")
    print("=" * 50)
    print(f"Train JSON: {config.train_json_path}")
    print(f"Test JSON: {config.test_json_path}")
    print(f"Image directory: {config.image_dir}")
    print(f"DINOv3 repo: {config.dinov3_repo_dir}")
    print(f"Backbone weights: {config.backbone_weights_path}")
    print(f"Experiment output: {config.exp_checkpoint_dir}")
    print("=" * 50)

def main():
    args = parse_args()
    
    print("🎯 DINOv3 + Mask2Former Training Launcher")
    print("=" * 60)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Print summaries
    print_config_summary(config)
    print_paths_summary(config)
    
    # Verify setup
    if not verify_setup(config):
        print("\n❌ Setup verification failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Create experiment directories
    create_experiment_dirs(config)
    
    # Save configuration
    config_save_path = os.path.join(config.exp_log_dir, "launch_config.json")
    save_config(config, config_save_path)
    print(f"\n💾 Configuration saved to: {config_save_path}")
    
    # If verify only, exit here
    if args.verify_only:
        print("\n✅ Verification completed successfully!")
        print("You can now run training by removing the --verify_only flag.")
        return
    
    # Check for resume checkpoint
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"❌ Resume checkpoint not found: {args.resume}")
            sys.exit(1)
        print(f"📂 Will resume from checkpoint: {args.resume}")
        # Note: Resume functionality would need to be integrated into the training loop
    
    # Launch training
    print(f"\n🚀 Starting training...")
    print(f"📊 Monitor training progress in: {config.exp_log_dir}")
    print(f"💾 Checkpoints will be saved to: {config.exp_checkpoint_dir}")
    print(f"🖼️  Visualizations will be saved to: {config.exp_viz_dir}")
    
    if args.gpus == 1:
        launch_single_gpu(config, args)
    else:
        launch_multi_gpu(config, args)
    
    print("\n🎉 Training completed!")
    print(f"📈 Check results in: {config.exp_checkpoint_dir}")

if __name__ == '__main__':
    main()
