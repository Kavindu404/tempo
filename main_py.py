#!/usr/bin/env python3
"""
Main entry point for DinoV3 Segmentation Training and Inference

This script provides a unified interface to run training, evaluation, and inference
for the DinoV3-based segmentation model.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def setup_environment():
    """Setup environment and check dependencies"""
    # Add current directory to Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Check if we're in the dinov3 repository
    if not (current_dir / 'dinov3').exists():
        print("Warning: dinov3 directory not found. Make sure you're running from the cloned dinov3 repository.")
    
    # Set PYTHONPATH environment variable
    os.environ['PYTHONPATH'] = str(current_dir)


def run_training(args):
    """Run training script with distributed support"""
    cmd = []
    
    # Check if we should use distributed training
    if args.num_gpus > 1:
        # Use torchrun for distributed training
        cmd.extend([
            'torchrun',
            f'--nproc_per_node={args.num_gpus}',
            '--nnodes=1',
            '--node_rank=0',
            '--master_addr=localhost',
            '--master_port=12355'
        ])
    else:
        cmd.append('python')
    
    cmd.append('train.py')
    
    # Add training arguments
    training_args = [
        f'--train_img_dir={args.train_img_dir}',
        f'--train_ann_file={args.train_ann_file}',
        f'--val_img_dir={args.val_img_dir}',
        f'--val_ann_file={args.val_ann_file}',
        f'--backbone_name={args.backbone_name}',
        f'--repo_dir={args.repo_dir}',
        f'--model_name={args.model_name}',
        f'--epochs={args.epochs}',
        f'--batch_size={args.batch_size}',
        f'--lr={args.lr}',
        f'--target_size={args.target_size}',
        f'--save_n={args.save_n}',
        f'--eval_n={args.eval_n}',
        f'--n_viz_samples={args.n_viz_samples}',
        f'--num_workers={args.num_workers}',
        f'--seed={args.seed}'
    ]
    
    if args.weights:
        training_args.append(f'--weights={args.weights}')
    
    if args.use_satellite_norm:
        training_args.append('--use_satellite_norm')
    
    if args.use_amp:
        training_args.append('--use_amp')
    
    if args.resume_from:
        training_args.append(f'--resume_from={args.resume_from}')
    
    cmd.extend(training_args)
    
    print("Running training with command:")
    print(" ".join(cmd))
    print()
    
    # Run training
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def run_inference(args):
    """Run inference script"""
    cmd = ['python', 'inference.py']
    
    # Add inference arguments
    inference_args = [
        f'--model_path={args.model_path}',
        f'--input={args.input}',
        f'--output_dir={args.output_dir}',
        f'--repo_dir={args.repo_dir}',
        f'--backbone_name={args.backbone_name}',
        f'--device={args.device}',
        f'--batch_size={args.batch_size}',
        f'--confidence_threshold={args.confidence_threshold}',
        f'--target_size={args.target_size}'
    ]
    
    if args.weights:
        inference_args.append(f'--weights={args.weights}')
    
    if args.use_satellite_norm:
        inference_args.append('--use_satellite_norm')
    
    if args.save_visualizations:
        inference_args.append('--save_visualizations')
    
    if args.mode:
        inference_args.append(f'--mode={args.mode}')
    
    cmd.extend(inference_args)
    
    print("Running inference with command:")
    print(" ".join(cmd))
    print()
    
    # Run inference
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def run_evaluation(args):
    """Run standalone evaluation"""
    cmd = ['python', '-c', '''
import torch
from model import build_model
from evaluation import SegmentationEvaluator
from dataset import create_dataloaders
import json
import argparse
import sys

# Parse arguments passed from main
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--val_img_dir", required=True)
parser.add_argument("--val_ann_file", required=True)
parser.add_argument("--repo_dir", default=".")
parser.add_argument("--backbone_name", default="dinov3_vitl16")
parser.add_argument("--weights", default=None)
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--target_size", type=int, default=512)
parser.add_argument("--use_satellite_norm", action="store_true")
eval_args = parser.parse_args()

# Load model
print("Loading model...")
checkpoint = torch.load(eval_args.model_path, map_location=eval_args.device)
num_classes = checkpoint.get("num_classes", 80)

model = build_model(
    backbone_name=eval_args.backbone_name,
    repo_dir=eval_args.repo_dir,
    weights=eval_args.weights,
    num_classes=num_classes
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Handle DDP wrapper
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(eval_args.device)

# Create data loader
_, val_loader = create_dataloaders(
    val_img_dir=eval_args.val_img_dir,
    val_ann_file=eval_args.val_ann_file,
    batch_size=eval_args.batch_size,
    target_size=eval_args.target_size,
    use_satellite_norm=eval_args.use_satellite_norm
)

# Create evaluator
evaluator = SegmentationEvaluator(
    gt_file=eval_args.val_ann_file,
    class_names=val_loader.dataset.get_category_names()
)

# Run evaluation
print("Running evaluation...")
metrics = evaluator.evaluate(model, val_loader, eval_args.device)

# Print results
print("\\nEvaluation Results:")
print("=" * 40)
for metric, value in metrics.items():
    print(f"{metric:>12s}: {value:.4f}")

# Save results
output_file = "evaluation_results.json"
with open(output_file, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\\nResults saved to {output_file}")
''']
    
    # Add evaluation arguments
    eval_args = [
        f'--model_path={args.model_path}',
        f'--val_img_dir={args.val_img_dir}',
        f'--val_ann_file={args.val_ann_file}',
        f'--repo_dir={args.repo_dir}',
        f'--backbone_name={args.backbone_name}',
        f'--device={args.device}',
        f'--batch_size={args.batch_size}',
        f'--target_size={args.target_size}'
    ]
    
    if args.weights:
        eval_args.append(f'--weights={args.weights}')
    
    if args.use_satellite_norm:
        eval_args.append('--use_satellite_norm')
    
    cmd.extend(eval_args)
    
    print("Running evaluation...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='DinoV3 Segmentation - Training and Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Training with satellite model
  python main.py train --weights path/to/satellite/model.pth --use_satellite_norm
  
  # Training with multiple GPUs
  python main.py train --num_gpus 8 --batch_size 64
  
  # Inference on single image
  python main.py inference --model_path checkpoints/best_model.pt --input image.jpg --output_dir results/
  
  # Inference on directory
  python main.py inference --model_path checkpoints/best_model.pt --input images/ --output_dir results/
  
  # Evaluation only
  python main.py evaluate --model_path checkpoints/best_model.pt
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train segmentation model')
    
    # Data arguments
    train_parser.add_argument('--train_img_dir', type=str, default='data/train/images',
                             help='Training images directory')
    train_parser.add_argument('--train_ann_file', type=str, default='data/train/annotations.json',
                             help='Training annotations file')
    train_parser.add_argument('--val_img_dir', type=str, default='data/val/images',
                             help='Validation images directory')
    train_parser.add_argument('--val_ann_file', type=str, default='data/val/annotations.json',
                             help='Validation annotations file')
    
    # Model arguments
    train_parser.add_argument('--backbone_name', type=str, default='dinov3_vitl16',
                             help='DinoV3 backbone name')
    train_parser.add_argument('--repo_dir', type=str, default='.',
                             help='DinoV3 repository directory')
    train_parser.add_argument('--weights', type=str, default=None,
                             help='Pretrained weights path/URL (satellite model recommended)')
    train_parser.add_argument('--model_name', type=str, default='dinov3_segmentation',
                             help='Model name for saving')
    
    # Training arguments
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=16,
                             help='Total batch size across all GPUs')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_parser.add_argument('--target_size', type=int, default=512,
                             help='Target image size')
    train_parser.add_argument('--use_satellite_norm', action='store_true', default=True,
                             help='Use satellite imagery normalization')
    train_parser.add_argument('--use_amp', action='store_true',
                             help='Use automatic mixed precision')
    
    # System arguments
    train_parser.add_argument('--num_gpus', type=int, default=1,
                             help='Number of GPUs to use')
    train_parser.add_argument('--num_workers', type=int, default=4,
                             help='Number of data loading workers')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Logging and saving
    train_parser.add_argument('--save_n', type=int, default=2,
                             help='Save checkpoint every N epochs')
    train_parser.add_argument('--eval_n', type=int, default=2,
                             help='Evaluate every N epochs')
    train_parser.add_argument('--n_viz_samples', type=int, default=10,
                             help='Number of visualization samples')
    
    # Resume training
    train_parser.add_argument('--resume_from', type=str, default=None,
                             help='Resume training from checkpoint')
    
    # Inference parser
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    
    inference_parser.add_argument('--model_path', type=str, required=True,
                                 help='Path to trained model checkpoint')
    inference_parser.add_argument('--input', type=str, required=True,
                                 help='Input image/directory path')
    inference_parser.add_argument('--output_dir', type=str, required=True,
                                 help='Output directory')
    inference_parser.add_argument('--repo_dir', type=str, default='.',
                                 help='DinoV3 repository directory')
    inference_parser.add_argument('--backbone_name', type=str, default='dinov3_vitl16',
                                 help='DinoV3 backbone name')
    inference_parser.add_argument('--weights', type=str, default=None,
                                 help='Pretrained backbone weights')
    inference_parser.add_argument('--device', type=str, default='auto',
                                 choices=['auto', 'cpu', 'cuda'],
                                 help='Device to use for inference')
    inference_parser.add_argument('--batch_size', type=int, default=8,
                                 help='Batch size for inference')
    inference_parser.add_argument('--confidence_threshold', type=float, default=0.5,
                                 help='Confidence threshold')
    inference_parser.add_argument('--target_size', type=int, default=512,
                                 help='Target image size')
    inference_parser.add_argument('--use_satellite_norm', action='store_true', default=True,
                                 help='Use satellite imagery normalization')
    inference_parser.add_argument('--save_visualizations', action='store_true', default=True,
                                 help='Save prediction visualizations')
    inference_parser.add_argument('--mode', type=str, choices=['single', 'batch', 'directory'],
                                 help='Inference mode (auto-detected if not specified)')
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    
    eval_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to trained model checkpoint')
    eval_parser.add_argument('--val_img_dir', type=str, default='data/val/images',
                            help='Validation images directory')
    eval_parser.add_argument('--val_ann_file', type=str, default='data/val/annotations.json',
                            help='Validation annotations file')
    eval_parser.add_argument('--repo_dir', type=str, default='.',
                            help='DinoV3 repository directory')
    eval_parser.add_argument('--backbone_name', type=str, default='dinov3_vitl16',
                            help='DinoV3 backbone name')
    eval_parser.add_argument('--weights', type=str, default=None,
                            help='Pretrained backbone weights')
    eval_parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use for evaluation')
    eval_parser.add_argument('--batch_size', type=int, default=8,
                            help='Batch size for evaluation')
    eval_parser.add_argument('--target_size', type=int, default=512,
                            help='Target image size')
    eval_parser.add_argument('--use_satellite_norm', action='store_true', default=True,
                            help='Use satellite imagery normalization')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Setup environment
    setup_environment()
    
    # Parse arguments
    args = parse_args()
    
    if args.command is None:
        print("Error: Please specify a command (train, inference, or evaluate)")
        print("Use --help for more information")
        return 1
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('viz', exist_ok=True)
    
    # Run specified command
    if args.command == 'train':
        print("=" * 60)
        print("Starting DinoV3 Segmentation Training")
        print("=" * 60)
        return_code = run_training(args)
    
    elif args.command == 'inference':
        print("=" * 60)
        print("Starting DinoV3 Segmentation Inference")
        print("=" * 60)
        return_code = run_inference(args)
    
    elif args.command == 'evaluate':
        print("=" * 60)
        print("Starting DinoV3 Segmentation Evaluation")
        print("=" * 60)
        return_code = run_evaluation(args)
    
    else:
        print(f"Unknown command: {args.command}")
        return 1
    
    if return_code == 0:
        print("\n" + "=" * 60)
        print("Task completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Task failed!")
        print("=" * 60)
    
    return return_code


if __name__ == '__main__':
    sys.exit(main())