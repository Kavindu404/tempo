import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Config:
    # Experiment
    exp_name: str = "dinov3_mask2former"
    seed: int = 42
    device: str = "cuda"
    
    # Dataset paths
    train_json: str = "/path/to/train.json"
    test_json: str = "/path/to/test.json"
    image_dir: str = "/path/to/images"
    num_classes: int = 1  # Single class dataset
    
    # DINOv3 backbone
    dinov3_repo_dir: str = "/path/to/dinov3/repo"
    backbone_type: str = "vitl16"  # Options: vits16, vitb16, vitl16, vit7b16, etc.
    backbone_weights_dir: str = "/path/to/backbone/weights"
    freeze_backbone: bool = False
    
    # Model architecture
    hidden_dim: int = 256
    num_queries: int = 100
    nheads: int = 8
    dim_feedforward: int = 2048
    dec_layers: int = 6
    mask_dim: int = 256
    
    # Training
    batch_size: int = 2
    num_epochs: int = 50
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    
    # Loss weights
    loss_ce: float = 2.0
    loss_mask: float = 5.0
    loss_dice: float = 5.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    
    # Data augmentation
    img_size: Tuple[int, int] = (800, 800)
    min_size: int = 600
    max_size: int = 1000
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Visualization
    num_viz: int = 10
    viz_threshold: float = 0.5
    viz_dir: str = "viz"
    
    # Logging and checkpoints
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    print_freq: int = 50
    val_freq: int = 1
    
    # Distributed training
    world_size: int = 8
    dist_backend: str = "nccl"
    dist_url: str = "env://"
    
    def get_backbone_weight_path(self):
        backbone_map = {
            'vits16': 'dinov3_vits16.pth',
            'vitb16': 'dinov3_vitb16.pth',
            'vitl16': 'dinov3_vitl16.pth',
            'vit7b16': 'dinov3_vit7b16.pth',
            'convnext_tiny': 'dinov3_convnext_tiny.pth',
            'convnext_small': 'dinov3_convnext_small.pth',
            'convnext_base': 'dinov3_convnext_base.pth',
            'convnext_large': 'dinov3_convnext_large.pth',
        }
        return os.path.join(self.backbone_weights_dir, backbone_map.get(self.backbone_type, f'{self.backbone_type}.pth'))