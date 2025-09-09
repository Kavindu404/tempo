import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
from tqdm import tqdm

from config import Config
from dataset import InstanceSegmentationDataset, get_transforms, collate_fn
from model import DINOv3Mask2Former
from criterion import build_criterion
from engine import train_one_epoch, evaluate, save_checkpoint, log_metrics, is_main_process
from visualizer import Visualizer

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_data_loaders(config, rank=0, world_size=1):
    """Create data loaders for training and evaluation"""
    
    # Create datasets
    train_transform = get_transforms(config, is_train=True)
    eval_transform = get_transforms(config, is_train=False)
    
    train_dataset = InstanceSegmentationDataset(
        config.train_json_path,
        config.image_dir,
        transforms=train_transform,
        is_train=True
    )
    
    eval_dataset = InstanceSegmentationDataset(
        config.test_json_path,
        config.image_dir,
        transforms=eval_transform,
        is_train=False
    )
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        eval_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, eval_loader, train_sampler, eval_sampler

def main_worker(rank, world_size, config):
    """Main worker function for distributed training"""
    
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        print(f"Using device: {device}")
        print(f"Training with {world_size} GPUs")
        print(f"Experiment name: {config.exp_name}")
    
    # Create directories
    if is_main_process():
        os.makedirs(config.exp_checkpoint_dir, exist_ok=True)
        os.makedirs(config.exp_log_dir, exist_ok=True)
        os.makedirs(config.exp_viz_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, eval_loader, train_sampler, eval_sampler = create_data_loaders(
        config, rank, world_size
    )
    
    if is_main_process():
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Evaluation samples: {len(eval_loader.dataset)}")
    
    # Create model
    model = DINOv3Mask2Former(config)
    model = model.to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Create criterion
    criterion, weight_dict = build_criterion(config)
    criterion = criterion.to(device)
    
    # Create optimizer
    # Separate parameters for backbone and head
    backbone_params = []
    head_params = []
    
    model_without_ddp = model.module if isinstance(model, DDP) else model
    
    for name, param in model_without_ddp.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    # Use different learning rates for backbone and head
    param_groups = [
        {'params': head_params, 'lr': config.learning_rate},
        {'params': backbone_params, 'lr': config.learning_rate * 0.1}  # Lower LR for backbone
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.num_epochs // 3, 
        gamma=0.1
    )
    
    # Create visualizer (only on main process)
    visualizer = Visualizer(config) if is_main_process() else None
    
    # Training loop
    best_mAP = 0.0
    
    if is_main_process():
        print("Starting training...")
    
    for epoch in range(1, config.num_epochs + 1):
        
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Training
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, config, epoch, visualizer
        )
        
        # Learning rate step
        scheduler.step()
        
        # Evaluation
        if epoch % 5 == 0 or epoch == config.num_epochs:  # Evaluate every 5 epochs
            eval_metrics = evaluate(
                model, criterion, eval_loader, device, config, epoch
            )
            
            # Save checkpoint if improved
            current_mAP = eval_metrics['mAP_segm']
            is_best = current_mAP > best_mAP
            
            if is_best:
                best_mAP = current_mAP
            
            # Save checkpoint (only on main process)
            if is_main_process():
                save_checkpoint(
                    model_without_ddp, optimizer, epoch, current_mAP, config, is_best
                )
            
            # Log metrics
            metrics = {
                'train': train_metrics,
                'eval': eval_metrics
            }
            log_metrics(metrics, epoch, config)
            
        else:
            # Log only training metrics
            metrics = {'train': train_metrics}
            log_metrics(metrics, epoch, config)
        
        # Synchronize processes
        if world_size > 1:
            dist.barrier()
    
    if is_main_process():
        print(f"Training completed! Best mAP: {best_mAP:.4f}")
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train DINOv3 Mask2Former')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--world_size', type=int, default=8, help='Number of GPUs')
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override config with command line arguments if provided
    if args.config:
        # Load config from file if needed
        pass
    
    config.world_size = args.world_size
    
    # Validate config
    assert os.path.exists(config.train_json_path), f"Train JSON not found: {config.train_json_path}"
    assert os.path.exists(config.test_json_path), f"Test JSON not found: {config.test_json_path}"
    assert os.path.exists(config.image_dir), f"Image directory not found: {config.image_dir}"
    assert os.path.exists(config.dinov3_repo_dir), f"DINOv3 repo not found: {config.dinov3_repo_dir}"
    assert os.path.exists(config.backbone_weights_path), f"Backbone weights not found: {config.backbone_weights_path}"
    
    if is_main_process():
        print("Configuration:")
        for key, value in config.__dict__.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        print()
    
    # Launch distributed training
    if config.world_size > 1:
        mp.spawn(
            main_worker,
            args=(config.world_size, config),
            nprocs=config.world_size,
            join=True
        )
    else:
        main_worker(0, 1, config)

if __name__ == '__main__':
    main()
