from dataclasses import dataclass

@dataclass
class Config:
    # ===== Experiment =====
    experiment: str = "dinov3_vitl16_custom_seg"
    seed: int = 42

    # ===== Data (COCO-format) =====
    train_img_dir: str = "data/train/images"
    train_ann_file: str = "data/train/instances_train.json"
    val_img_dir: str   = "data/val/images"
    val_ann_file: str  = "data/val/instances_val.json"

    # Will be auto-inferred from train_ann_file categories (override to force)
    num_categories: int | None = None

    # ===== Augmentations =====
    img_size: int = 1024
    use_mosaic: bool = True
    mosaic_prob: float = 0.5
    use_mixup: bool = True
    mixup_prob: float = 0.3
    mixup_alpha: float = 0.5  # image blend alpha; masks/instances are concatenated (no blending)

    # ===== Model / Hub =====
    repo_dir: str = "path/to/local/facebookresearch/dinov3"  # local clone
    hub_entry: str = "dinov3_vitl16_ms"        # update if your hub key differs
    head_weights: str | None = None            # optional path/URL
    backbone_weights: str | None = None        # optional path/URL

    # Freeze→Unfreeze
    freeze_backbone_epochs: int = 5

    # ===== Train =====
    epochs: int = 50
    batch_size: int = 2              # per-GPU
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 8
    amp: bool = True

    # ===== Matcher costs =====
    cost_class: float = 2.0
    cost_mask: float = 5.0
    cost_dice: float = 5.0

    # ===== DDP =====
    dist_backend: str = "nccl"

    # ===== I/O =====
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    viz_dir: str = "viz"
    num_viz_samples: int = 10      # <-- your default
