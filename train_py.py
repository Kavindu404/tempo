import os
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Optional
import argparse

from dataset import create_dataloaders
from model import build_model
from evaluation import SegmentationEvaluator, MetricTracker, log_evaluation_results
from utils import (
    setup_logging, set_seed, setup_distributed, cleanup_distributed,
    save_checkpoint, load_checkpoint, AverageMeter, cosine_lr_scheduler,
    visualize_predictions, log_model_info, EarlyStopping
)


class Trainer:
    """Main trainer class for segmentation model"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup distributed training
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main_process = self.rank == 0
        
        # Set device
        self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(self.local_rank)
        
        # Setup logging (only on main process)
        if self.is_main_process:
            log_file = os.path.join('logs', f'{args.model_name}.log')
            setup_logging(log_file)
            self.logger = logging.getLogger(__name__)
        
        # Set random seed
        set_seed(args.seed)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_evaluator()
        self._setup_tracking()
        
        if self.is_main_process:
            log_model_info(self.model, self.logger)
    
    def _setup_data(self):
        """Setup data loaders"""
        # Calculate effective batch size
        if self.world_size > 1:
            assert self.args.batch_size % self.world_size == 0
            per_gpu_batch_size = self.args.batch_size // self.world_size
        else:
            per_gpu_batch_size = self.args.batch_size
        
        # Create datasets and loaders
        train_loader, val_loader = create_dataloaders(
            train_img_dir=self.args.train_img_dir,
            train_ann_file=self.args.train_ann_file,
            val_img_dir=self.args.val_img_dir,
            val_ann_file=self.args.val_ann_file,
            batch_size=per_gpu_batch_size,
            num_workers=self.args.num_workers,
            target_size=self.args.target_size,
            max_objects=self.args.max_objects,
            use_satellite_norm=self.args.use_satellite_norm
        )
        
        # Setup distributed samplers
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                train_loader.dataset, 
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.val_sampler = DistributedSampler(
                val_loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            # Recreate loaders with distributed samplers
            self.train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=per_gpu_batch_size,
                sampler=self.train_sampler,
                num_workers=self.args.num_workers,
                collate_fn=train_loader.collate_fn,
                pin_memory=True,
                drop_last=True
            )
            
            self.val_loader = torch.utils.data.DataLoader(
                val_loader.dataset,
                batch_size=per_gpu_batch_size,
                sampler=self.val_sampler,
                num_workers=self.args.num_workers,
                collate_fn=val_loader.collate_fn,
                pin_memory=True
            )
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.train_sampler = None
            self.val_sampler = None
        
        # Get number of classes from dataset
        self.num_classes = train_loader.dataset.num_classes
        self.class_names = train_loader.dataset.get_category_names()
        
        if self.is_main_process:
            self.logger.info(f"Training samples: {len(train_loader.dataset)}")
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
            self.logger.info(f"Number of classes: {self.num_classes}")
    
    def _setup_model(self):
        """Setup model"""
        self.model = build_model(
            backbone_name=self.args.backbone_name,
            repo_dir=self.args.repo_dir,
            weights=self.args.weights,
            num_classes=self.num_classes,
            num_queries=self.args.num_queries,
            hidden_dim=self.args.hidden_dim
        )
        
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if using multiple GPUs
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Different learning rates for backbone and head
        backbone_params = []
        head_params = []
        
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': self.args.lr * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.args.lr}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        
        self.scheduler = cosine_lr_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=0.1
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
    
    def _setup_evaluator(self):
        """Setup evaluator"""
        if self.is_main_process:
            self.evaluator = SegmentationEvaluator(
                gt_file=self.args.val_ann_file,
                class_names=self.class_names,
                confidence_threshold=self.args.confidence_threshold,
                use_fast_eval=True
            )
        else:
            self.evaluator = None
    
    def _setup_tracking(self):
        """Setup metric tracking and early stopping"""
        self.metric_tracker = MetricTracker()
        self.early_stopping = EarlyStopping(
            patience=self.args.patience,
            mode='max'
        )
        
        # Best metric tracking
        self.best_map = 0.0
        self.start_epoch = 0
        
        # Resume from checkpoint if provided
        if self.args.resume_from:
            self._load_checkpoint(self.args.resume_from)
    
    def train(self):
        """Main training loop"""
        if self.is_main_process:
            self.logger.info("Starting training...")
            self.logger.info(f"Total epochs: {self.args.epochs}")
            self.logger.info(f"Effective batch size: {self.args.batch_size}")
            self.logger.info(f"Learning rate: {self.args.lr}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Set epoch for distributed sampler
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Validation
            if (epoch + 1) % self.args.eval_n == 0:
                val_metrics = self._validate_epoch(epoch)
                
                # Update metric tracker
                if self.is_main_process:
                    self.metric_tracker.update(train_metrics, 'train')
                    self.metric_tracker.update(val_metrics, 'val')
                    self.metric_tracker.log_metrics(epoch + 1, self.logger)
                    
                    # Check for best model
                    current_map = val_metrics.get('mAP', 0.0)
                    is_best = current_map > self.best_map
                    if is_best:
                        self.best_map = current_map
                    
                    # Save checkpoint
                    if (epoch + 1) % self.args.save_n == 0:
                        self._save_checkpoint(epoch + 1, train_metrics['loss'], current_map, is_best)
                    
                    # Early stopping
                    if self.early_stopping(current_map):
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
            else:
                # Update only training metrics
                if self.is_main_process:
                    self.metric_tracker.update(train_metrics, 'train')
                    
                    # Save periodic checkpoint
                    if (epoch + 1) % self.args.save_n == 0:
                        self._save_checkpoint(epoch + 1, train_metrics['loss'], 0.0, False)
            
            # Generate visualizations
            if self.is_main_process and (epoch + 1) % self.args.vis_n == 0:
                self._generate_visualizations(epoch + 1)
        
        if self.is_main_process:
            self.logger.info("Training completed!")
            self.logger.info(f"Best mAP: {self.best_map:.4f}")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        loss_class_meter = AverageMeter()
        loss_bbox_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        loss_dice_meter = AverageMeter()
        
        if self.is_main_process:
            self.logger.info(f"\nEpoch {epoch + 1}/{self.args.epochs}")
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            images = batch['images']
            targets = {
                'boxes': batch['boxes'],
                'masks': batch['masks'],
                'labels': batch['labels'],
                'valid': batch['valid']
            }
            
            # Forward pass
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss_dict = self.model.module.compute_loss(outputs, targets) \
                        if isinstance(self.model, DDP) else self.model.compute_loss(outputs, targets)
            else:
                outputs = self.model(images)
                loss_dict = self.model.module.compute_loss(outputs, targets) \
                    if isinstance(self.model, DDP) else self.model.compute_loss(outputs, targets)
            
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            loss_class_meter.update(loss_dict['loss_class'].item(), batch_size)
            loss_bbox_meter.update(loss_dict['loss_bbox'].item(), batch_size)
            loss_mask_meter.update(loss_dict['loss_mask'].item(), batch_size)
            loss_dice_meter.update(loss_dict['loss_dice'].item(), batch_size)
            
            # Logging
            if self.is_main_process and batch_idx % self.args.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                
                self.logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss_meter.avg:.4f} | "
                    f"Class: {loss_class_meter.avg:.4f} | "
                    f"BBox: {loss_bbox_meter.avg:.4f} | "
                    f"Mask: {loss_mask_meter.avg:.4f} | "
                    f"Dice: {loss_dice_meter.avg:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        return {
            'loss': loss_meter.avg,
            'loss_class': loss_class_meter.avg,
            'loss_bbox': loss_bbox_meter.avg,
            'loss_mask': loss_mask_meter.avg,
            'loss_dice': loss_dice_meter.avg
        }
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate one epoch"""
        if not self.is_main_process:
            return {}
        
        self.logger.info("Running validation...")
        
        # Evaluate model
        val_metrics = self.evaluator.evaluate(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device
        )
        
        # Log results
        log_evaluation_results(val_metrics, epoch + 1, self.logger)
        
        return val_metrics
    
    def _generate_visualizations(self, epoch: int):
        """Generate prediction visualizations"""
        if not self.is_main_process:
            return
        
        self.logger.info("Generating visualizations...")
        
        self.model.eval()
        
        # Get a batch from validation set
        val_iter = iter(self.val_loader)
        batch = next(val_iter)
        
        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model(batch['images'])
        
        # Generate visualizations
        viz_dir = os.path.join('viz', f'{self.args.model_name}_viz')
        
        visualize_predictions(
            images=batch['images'],
            predictions=outputs,
            targets={
                'boxes': batch['boxes'],
                'masks': batch['masks'],
                'labels': batch['labels'],
                'valid': batch['valid']
            },
            class_names=self.class_names,
            save_dir=viz_dir,
            epoch=epoch,
            n_samples=self.args.n_viz_samples,
            use_satellite_norm=self.args.use_satellite_norm
        )
    
    def _save_checkpoint(self, epoch: int, loss: float, metric: float, is_best: bool):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        checkpoint_dir = os.path.join('checkpoints', self.args.model_name)
        filename = f'{self.args.model_name}_epoch_{epoch}_mAP_{metric:.4f}.pt'
        filepath = os.path.join(checkpoint_dir, filename)
        
        save_checkpoint(
            model=model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            loss=loss,
            metric=metric,
            filepath=filepath,
            is_best=is_best
        )
        
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def _load_checkpoint(self, filepath: str):
        """Load checkpoint for resuming training"""
        if self.is_main_process:
            self.logger.info(f"Loading checkpoint: {filepath}")
        
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        
        checkpoint = load_checkpoint(
            filepath=filepath,
            model=model_to_load,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.start_epoch = checkpoint['epoch']
        self.best_map = checkpoint.get('metric', 0.0)
        
        if self.is_main_process:
            self.logger.info(f"Resumed from epoch {self.start_epoch}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DinoV3 Segmentation Training')
    
    # Data arguments
    parser.add_argument('--train_img_dir', type=str, default='data/train/images',
                       help='Training images directory')
    parser.add_argument('--train_ann_file', type=str, default='data/train/annotations.json',
                       help='Training annotations file')
    parser.add_argument('--val_img_dir', type=str, default='data/val/images',
                       help='Validation images directory')
    parser.add_argument('--val_ann_file', type=str, default='data/val/annotations.json',
                       help='Validation annotations file')
    
    # Model arguments
    parser.add_argument('--backbone_name', type=str, default='dinov3_vitl16',
                       help='DinoV3 backbone name')
    parser.add_argument('--repo_dir', type=str, default='.',
                       help='DinoV3 repository directory')
    parser.add_argument('--weights', type=str, default=None,
                       help='Pretrained weights path/URL')
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Number of object queries')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='Hidden dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Total batch size across all GPUs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    
    # Data processing arguments
    parser.add_argument('--target_size', type=int, default=512,
                       help='Target image size')
    parser.add_argument('--max_objects', type=int, default=100,
                       help='Maximum objects per image')
    parser.add_argument('--use_satellite_norm', action='store_true', default=True,
                       help='Use satellite imagery normalization')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for evaluation')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Logging and saving arguments
    parser.add_argument('--model_name', type=str, default='dinov3_segmentation',
                       help='Model name for saving')
    parser.add_argument('--save_n', type=int, default=2,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_n', type=int, default=2,
                       help='Evaluate every N epochs')
    parser.add_argument('--vis_n', type=int, default=5,
                       help='Generate visualizations every N epochs')
    parser.add_argument('--n_viz_samples', type=int, default=10,
                       help='Number of visualization samples')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log interval for training')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        if trainer.is_main_process:
            trainer.logger.info("Training interrupted by user")
    except Exception as e:
        if trainer.is_main_process:
            trainer.logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()