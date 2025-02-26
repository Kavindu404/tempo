#!/usr/bin/env python3
"""
Complete pipeline script to prepare data, fine-tune, and run inference with SAM 2.1
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='SAM 2.1 fine-tuning pipeline')
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--annotation_file', type=str, default='annotations/640.json', help='Path to annotation file relative to dataset_dir')
    
    # SAM 2.1 parameters
    parser.add_argument('--sam_repo', type=str, required=True, help='Path to SAM 2.1 repository')
    parser.add_argument('--sam_checkpoint', type=str, required=True, help='Path to SAM 2.1 checkpoint')
    parser.add_argument('--prompt_type', type=str, default='box', choices=['box', 'point'], help='Prompt type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use (comma-separated)')
    
    # Pipeline control
    parser.add_argument('--skip_prepare', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip_train', action='store_true', help='Skip training')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference')
    
    # Output directories
    parser.add_argument('--prepared_data_dir', type=str, default='prepared_dataset', help='Directory for prepared data')
    parser.add_argument('--output_dir', type=str, default='sam_finetune_output', help='Output directory for checkpoints')
    parser.add_argument('--inference_dir', type=str, default='inference_output', help='Output directory for inference results')
    
    # Inference parameters
    parser.add_argument('--inference_image', type=str, help='Image path for inference test after training')
    
    return parser.parse_args()

def prepare_data(args):
    """Prepare dataset for SAM 2.1 fine-tuning."""
    print("=== Preparing dataset ===")
    
    # Copy scripts to current directory if they don't exist
    if not os.path.exists('data_preparation.py'):
        shutil.copy('data-preparation.py', 'data_preparation.py')
    
    # Run data preparation script
    cmd = [
        'python', 'data_preparation.py',
        '--dataset_dir', args.dataset_dir,
        '--output_dir', args.prepared_data_dir,
        '--annotation_file', args.annotation_file
    ]
    
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("Data preparation failed")
    
    print(f"Dataset prepared in {args.prepared_data_dir}")

def train_model(args):
    """Fine-tune SAM 2.1 model."""
    print("=== Fine-tuning SAM 2.1 ===")
    
    # Copy scripts to current directory if they don't exist
    if not os.path.exists('dataloader.py'):
        shutil.copy('dataloader.py', 'dataloader.py')
    
    if not os.path.exists('training_script.py'):
        shutil.copy('training-script.py', 'training_script.py')
    
    # Add SAM 2.1 repo to PYTHONPATH
    os.environ['PYTHONPATH'] = f"{args.sam_repo}:{os.environ.get('PYTHONPATH', '')}"
    
    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # Run training script
    cmd = [
        'python', 'training_script.py',
        '--data_dir', args.prepared_data_dir,
        '--output_dir', args.output_dir,
        '--checkpoint', args.sam_checkpoint,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--image_size', str(args.image_size),
        '--prompt_type', args.prompt_type
    ]
    
    if args.fp16:
        cmd.append('--fp16')
    
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("Training failed")
    
    print(f"Fine-tuning completed, checkpoints saved in {args.output_dir}")
    
    # Find the latest checkpoint
    checkpoints = list(Path(args.output_dir).glob('sam2_finetuned_epoch*.pth'))
    if not checkpoints:
        raise RuntimeError("No checkpoints found after training")
    
    latest_checkpoint = str(sorted(checkpoints, key=lambda x: int(x.stem.split('epoch')[1]))[-1])
    print(f"Latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint

def run_inference(args, checkpoint_path):
    """Run inference with fine-tuned model."""
    print("=== Running inference ===")
    
    # Copy inference script to current directory if it doesn't exist
    if not os.path.exists('inference_script.py'):
        shutil.copy('inference-script.py', 'inference_script.py')
    
    # Choose an image for inference
    if args.inference_image:
        image_path = args.inference_image
    else:
        # Find an image from the validation set
        val_images = list(Path(args.prepared_data_dir).glob('val/images/*'))
        if not val_images:
            raise RuntimeError("No validation images found")
        image_path = str(val_images[0])
        print(f"Using validation image: {image_path}")
    
    # Add SAM 2.1 repo to PYTHONPATH
    os.environ['PYTHONPATH'] = f"{args.sam_repo}:{os.environ.get('PYTHONPATH', '')}"
    
    # Run inference script
    cmd = [
        'python', 'inference_script.py',
        '--checkpoint', checkpoint_path,
        '--base_checkpoint', args.sam_checkpoint,
        '--image_path', image_path,
        '--prompt_type', args.prompt_type,
        '--output_dir', args.inference_dir,
        '--compare'  # Compare with base model
    ]
    
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("Inference failed")
    
    print(f"Inference completed, results saved in {args.inference_dir}")

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.prepared_data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.inference_dir, exist_ok=True)
    
    # Run pipeline
    try:
        if not args.skip_prepare:
            prepare_data(args)
        else:
            print("Skipping data preparation")
        
        if not args.skip_train:
            checkpoint_path = train_model(args)
        else:
            print("Skipping training")
            # Find the latest checkpoint if skipping training
            checkpoints = list(Path(args.output_dir).glob('sam2_finetuned_epoch*.pth'))
            if not checkpoints:
                raise RuntimeError("No checkpoints found, cannot skip training")
            checkpoint_path = str(sorted(checkpoints, key=lambda x: int(x.stem.split('epoch')[1]))[-1])
        
        if not args.skip_inference:
            run_inference(args, checkpoint_path)
        else:
            print("Skipping inference")
        
        print("Pipeline completed successfully!")
    
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
