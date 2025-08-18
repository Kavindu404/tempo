import os
import json
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2


def setup_logging(log_file: str, level: int = logging.INFO):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    metric: float,
    filepath: str,
    is_best: bool = False
):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metric': metric
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_filepath = filepath.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Dict:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.430, 0.411, 0.296),
    std: Tuple[float, float, float] = (0.213, 0.156, 0.143)
) -> np.ndarray:
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def visualize_predictions(
    images: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    class_names: List[str],
    save_dir: str,
    epoch: int,
    n_samples: int = 10,
    confidence_threshold: float = 0.5,
    use_satellite_norm: bool = True
):
    """Visualize predictions and ground truth"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples
    batch_size = images.shape[0]
    indices = random.sample(range(batch_size), min(n_samples, batch_size))
    
    # Normalization parameters
    if use_satellite_norm:
        mean = (0.430, 0.411, 0.296)
        std = (0.213, 0.156, 0.143)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    for i, idx in enumerate(indices):
        # Denormalize image
        image = denormalize_image(images[idx], mean, std)
        
        # Get predictions
        pred_logits = predictions['pred_logits'][idx]  # [num_queries, num_classes+1]
        pred_boxes = predictions['pred_boxes'][idx]    # [num_queries, 4]
        pred_masks = predictions['pred_masks'][idx]    # [num_queries, H, W]
        
        # Get ground truth
        gt_boxes = targets['boxes'][idx]      # [max_objects, 4]
        gt_masks = targets['masks'][idx]      # [max_objects, H, W]
        gt_labels = targets['labels'][idx]    # [max_objects]
        gt_valid = targets['valid'][idx]      # [max_objects]
        
        # Filter valid ground truth
        valid_gt = gt_valid.cpu().numpy()
        gt_boxes = gt_boxes[valid_gt].cpu().numpy()
        gt_masks = gt_masks[valid_gt].cpu().numpy()
        gt_labels = gt_labels[valid_gt].cpu().numpy()
        
        # Filter confident predictions
        pred_probs = torch.softmax(pred_logits, dim=-1)
        pred_scores, pred_classes = pred_probs[:, :-1].max(dim=-1)  # Exclude no-object class
        
        confident_mask = pred_scores > confidence_threshold
        if confident_mask.sum() > 0:
            pred_boxes_conf = pred_boxes[confident_mask].cpu().numpy()
            pred_masks_conf = pred_masks[confident_mask].sigmoid().cpu().numpy()
            pred_classes_conf = pred_classes[confident_mask].cpu().numpy()
            pred_scores_conf = pred_scores[confident_mask].cpu().numpy()
        else:
            pred_boxes_conf = np.empty((0, 4))
            pred_masks_conf = np.empty((0, pred_masks.shape[1], pred_masks.shape[2]))
            pred_classes_conf = np.empty((0,))
            pred_scores_conf = np.empty((0,))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(image)
        _draw_boxes_and_masks(
            axes[0, 1], image, gt_boxes, gt_masks, gt_labels, 
            class_names, title='Ground Truth'
        )
        
        # Predictions
        axes[1, 0].imshow(image)
        _draw_boxes_and_masks(
            axes[1, 0], image, pred_boxes_conf, pred_masks_conf, 
            pred_classes_conf, class_names, pred_scores_conf, 
            title='Predictions'
        )
        
        # Overlay comparison
        axes[1, 1].imshow(image)
        _draw_boxes_and_masks(
            axes[1, 1], image, gt_boxes, gt_masks, gt_labels, 
            class_names, title='GT (Red) vs Pred (Blue)', color_gt='red'
        )
        _draw_boxes_and_masks(
            axes[1, 1], image, pred_boxes_conf, pred_masks_conf, 
            pred_classes_conf, class_names, pred_scores_conf, 
            color_pred='blue', draw_labels=False
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f'sample_{i}_epoch_{epoch}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()


def _draw_boxes_and_masks(
    ax,
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    scores: Optional[np.ndarray] = None,
    title: str = '',
    color_gt: str = 'red',
    color_pred: str = 'blue',
    draw_labels: bool = True
):
    """Helper function to draw bounding boxes and masks"""
    h, w = image.shape[:2]
    
    if len(boxes) == 0:
        ax.set_title(title)
        ax.axis('off')
        return
    
    # Choose color
    color = color_gt if scores is None else color_pred
    
    for j, (box, mask, label) in enumerate(zip(boxes, masks, labels)):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Draw mask
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_colored = np.zeros((h, w, 4))
        if color == 'red':
            mask_colored[:, :, 0] = mask_resized
        else:  # blue
            mask_colored[:, :, 2] = mask_resized
        mask_colored[:, :, 3] = mask_resized * 0.3  # Alpha
        ax.imshow(mask_colored)
        
        # Draw label
        if draw_labels and label < len(class_names):
            text = class_names[label]
            if scores is not None:
                text += f' {scores[j]:.2f}'
            
            ax.text(
                x1, y1 - 5, text,
                fontsize=10, color=color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    ax.set_title(title)
    ax.axis('off')


def create_coco_submission(
    predictions: List[Dict],
    output_file: str,
    image_ids: List[int],
    category_mapping: Optional[Dict[int, int]] = None
):
    """Create COCO format submission file"""
    results = []
    
    for pred, img_id in zip(predictions, image_ids):
        pred_boxes = pred['boxes']
        pred_masks = pred['masks']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        for box, mask, label, score in zip(pred_boxes, pred_masks, pred_labels, pred_scores):
            # Convert box format from [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = box
            bbox = [x1, y1, x2 - x1, y2 - y1]
            
            # Convert mask to RLE format
            mask_rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            
            # Map category if needed
            category_id = category_mapping[label] if category_mapping else label
            
            result = {
                'image_id': img_id,
                'category_id': int(category_id),
                'bbox': [float(x) for x in bbox],
                'segmentation': mask_rle,
                'score': float(score)
            }
            results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f)


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    """Learning rate warmup scheduler"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1
):
    """Cosine annealing learning rate scheduler with warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_gpu_memory_usage() -> str:
    """Get GPU memory usage string"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
    return "CUDA not available"


def log_model_info(model: torch.nn.Module, logger: logging.Logger):
    """Log model information"""
    total_params, trainable_params = count_parameters(model)
    
    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"  {get_gpu_memory_usage()}")


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            self.is_better = lambda current, best: current < best - min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
