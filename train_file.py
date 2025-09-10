import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import random
import numpy as np
from config import Config
from dataset import InstanceSegmentationDataset, get_transforms, collate_fn
from model import DINOv3Mask2Former
from criterion import SetCriterion
from matcher import HungarianMatcher
from engine import train_one_epoch, evaluate, save_checkpoint, save_logs

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def set_seed(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def main_worker(rank, world_size, config):
    setup(rank, world_size)
    set_seed(config.seed, rank)
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create datasets
    train_dataset = InstanceSegmentationDataset(
        config.train_json,
        config.image_dir,
        transforms=get_transforms(config, is_train=True),
        is_train=True
    )
    
    val_dataset = InstanceSegmentationDataset(
        config.test_json,
        config.image_dir,
        transforms=get_transforms(config, is_train=False),
        is_train=False
    )
    
    # Create data loaders with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = DINOv3Mask2Former(config).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # Create matcher and criterion
    matcher = HungarianMatcher(
        cost_class=config.loss_ce,
        cost_mask=config.loss_mask,
        cost_dice=config.loss_dice,
        cost_bbox=config.loss_bbox,
        cost_giou=config.loss_giou
    )
    
    weight_dict = {
        'loss_ce': config.loss_ce,
        'loss_mask': config.loss_mask,
        'loss_dice': config.loss_dice,
        'loss_bbox': config.loss_bbox,
        'loss_giou': config.loss_giou
    }
    
    # Add auxiliary loss weights
    aux_weight_dict = {}
    for i in range(config.dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    
    criterion = SetCriterion(
        config.num_classes,
        matcher,
        weight_dict,
        eos_coef=0.1
    ).to(device)
    
    # Create optimizer
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'backbone' not in n and p.requires_grad],
            'lr': config.lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'backbone' in n and p.requires_grad],
            'lr': config.backbone_lr
        }
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_mAP = 0.0
    
    if rank == 0:
        print(f"Starting training for {config.num_epochs} epochs")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
    
    for epoch in range(1, config.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, config
        )
        
        if rank == 0:
            print(f"Epoch {epoch} Training: {train_metrics}")
        
        # Validate
        if epoch % config.val_freq == 0:
            val_metrics = evaluate(
                model, criterion, val_loader, device, config, 
                epoch=epoch, save_viz=(epoch % 5 == 0)
            )
            
            if rank == 0:
                print(f"Epoch {epoch} Validation: {val_metrics}")
                
                # Save logs
                save_logs(epoch, val_metrics, config)
                
                # Save checkpoint if improved
                current_mAP = val_metrics.get_avg('mAP_segm')
                if current_mAP > best_mAP:
                    best_mAP = current_mAP
                    save_checkpoint(model, optimizer, epoch, val_metrics, config)
                    print(f"New best mAP: {best_mAP:.4f}")
        
        scheduler.step()
    
    cleanup()

def main():
    config = Config()
    
    # Create necessary directories
    os.makedirs(config.viz_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, config)

if __name__ == "__main__":
    main()