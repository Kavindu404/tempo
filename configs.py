from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class Config:
    # --- Data ---
    train_json: str = "/path/to/train.json"
    val_json: str = "/path/to/val.json"
    image_dir: str = "/path/to/images"

    # --- Experiment ---
    exp_name: str = "dinov3_m2f_singlecls"
    output_root: str = "./"
    seed: int = 42

    # --- Backbone ---
    dinov3_repo_dir: str = "/path/to/dinov3_repo"
    backbone_name: str = "dinov3_vitl16"   # e.g., dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16, convnext variants
    backbone_weights: str = "/path/to/checkpoint.pth"
    freeze_backbone: bool = True
    out_stride: int = 16   # feature stride target (used in simple pixel decoder)

    # --- Model/Head ---
    hidden_dim: int = 256
    num_queries: int = 100
    num_decoder_layers: int = 6
    num_classes: int = 1   # single foreground class (we add a no-object internally)
    mask_dim: int = 256    # channels for mask embeddings
    use_bias: bool = True

    # --- Loss ---
    cls_alpha: float = 2.0   # focal
    cls_gamma: float = 2.0
    cls_weight: float = 2.0
    bbox_l1_weight: float = 5.0
    bbox_giou_weight: float = 2.0
    mask_focal_weight: float = 2.0
    mask_dice_weight: float = 5.0
    no_object_weight: float = 0.1

    # --- Optimization ---
    epochs: int = 50
    batch_size: int = 2
    lr: float = 1e-4
    wd: float = 1e-4
    backbone_lr_mult: float = 0.1
    warmup_epochs: int = 1
    num_workers: int = 8
    grad_clip_norm: float = 1.0
    amp: bool = True

    # --- Augmentations ---
    image_size: Tuple[int, int] = (1024, 1024)
    augment: bool = True

    # --- Viz ---
    viz_n_per_epoch: int = 10
    viz_score_thresh: float = 0.5

    # --- DDP ---
    backend: str = "nccl"
    find_unused_params: bool = False
    dist_url: Optional[str] = None  # set by launcher

    # --- Eval ---
    eval_every_epoch: bool = True
