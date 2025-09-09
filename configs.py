import os

class Config:
    # Experiment settings
    exp_name = "dinov3_mask2former_exp1"
    
    # Dataset settings
    train_json_path = "/path/to/train_annotations.json"
    test_json_path = "/path/to/test_annotations.json"
    image_dir = "/path/to/images"
    num_classes = 1  # Single class dataset
    
    # DINOv3 backbone settings
    dinov3_repo_dir = "/path/to/dinov3/repo"
    backbone_weights_dir = "/path/to/backbone/weights"
    backbone_type = "dinov3_vitl16"  # Options: dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16, dinov3_convnext_*
    freeze_backbone = False  # Set to True to freeze backbone parameters
    
    # Model settings
    hidden_dim = 256
    num_queries = 100
    num_encoder_layers = 6
    num_decoder_layers = 6
    num_heads = 8
    dropout = 0.1
    activation = "relu"
    
    # Training settings
    batch_size = 2  # Per GPU
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    gradient_clip_max_norm = 0.1
    
    # Data augmentation settings
    image_size = (1024, 1024)
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    
    # Loss weights
    mask_loss_weight = 20.0
    dice_loss_weight = 1.0
    cls_loss_weight = 2.0
    bbox_loss_weight = 5.0
    giou_loss_weight = 2.0
    
    # Evaluation settings
    eval_threshold = 0.5
    
    # Visualization settings
    num_viz_samples = 10  # Number of visualization samples to save per epoch
    viz_threshold = 0.5
    
    # Distributed training
    world_size = 8  # Number of GPUs
    
    # Paths (will be created if they don't exist)
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    viz_dir = "viz"
    
    @property
    def exp_checkpoint_dir(self):
        return os.path.join(self.checkpoint_dir, self.exp_name)
    
    @property
    def exp_log_dir(self):
        return os.path.join(self.log_dir, self.exp_name)
    
    @property
    def exp_viz_dir(self):
        return os.path.join(self.viz_dir, self.exp_name)
    
    @property
    def backbone_weights_path(self):
        # Map backbone type to expected weight file
        weight_mapping = {
            "dinov3_vits16": "dinov3_vits16_300ep.pth",
            "dinov3_vitb16": "dinov3_vitb16_300ep.pth", 
            "dinov3_vitl16": "dinov3_vitl16_300ep.pth",
            "dinov3_vith16plus": "dinov3_vith16plus_300ep.pth",
            "dinov3_vit7b16": "dinov3_vit7b16_300ep.pth",
            "dinov3_convnext_tiny": "dinov3_convnext_tiny_300ep.pth",
            "dinov3_convnext_small": "dinov3_convnext_small_300ep.pth",
            "dinov3_convnext_base": "dinov3_convnext_base_300ep.pth",
            "dinov3_convnext_large": "dinov3_convnext_large_300ep.pth",
        }
        weight_file = weight_mapping.get(self.backbone_type, f"{self.backbone_type}.pth")
        return os.path.join(self.backbone_weights_dir, weight_file)
