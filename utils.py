import os
import torch
import torch.distributed as dist
import numpy as np
import random
import json
from typing import Dict, List, Any
import shutil
import logging

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file=None, level=logging.INFO):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_config(config, save_path):
    """Save configuration to JSON file"""
    config_dict = {}
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            # Convert paths to strings and handle non-serializable objects
            if hasattr(value, '__dict__'):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_checkpoint(checkpoint_path, model, optimizer=None, strict=True):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    mAP = checkpoint.get('mAP', 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch} with mAP {mAP:.4f}")
    return epoch, mAP

def get_model_size(model):
    """Calculate model size in parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }

def print_model_info(model, config):
    """Print model information"""
    model_info = get_model_size(model)
    
    print("Model Information:")
    print("-" * 40)
    print(f"Backbone: {config.backbone_type}")
    print(f"Backbone frozen: {config.freeze_backbone}")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    print(f"Frozen parameters: {model_info['frozen_params']:,}")
    print(f"Model size: {model_info['total_params'] * 4 / 1024 / 1024:.2f} MB")
    print("-" * 40)

def create_experiment_dirs(config):
    """Create all necessary experiment directories"""
    dirs_to_create = [
        config.exp_checkpoint_dir,
        config.exp_log_dir,
        config.exp_viz_dir
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def cleanup_old_checkpoints(checkpoint_dir, keep_best=5):
    """Clean up old checkpoints, keeping only the best ones"""
    if not os.path.exists(checkpoint_dir):
        return
    
    # Find all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') and 'mAP_' in file and file != 'best_model.pt':
            # Extract mAP from filename
            try:
                mAP_str = file.split('mAP_')[1].split('.pt')[0]
                mAP = float(mAP_str)
                checkpoint_files.append((file, mAP))
            except:
                continue
    
    # Sort by mAP (descending)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old checkpoints
    for i, (filename, mAP) in enumerate(checkpoint_files):
        if i >= keep_best:
            file_path = os.path.join(checkpoint_dir, filename)
            os.remove(file_path)
            print(f"Removed old checkpoint: {filename}")

def validate_dataset_format(json_path, image_dir):
    """Validate dataset format"""
    print(f"Validating dataset: {json_path}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check required keys
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in dataset: {key}")
    
    # Check images
    print(f"Number of images: {len(data['images'])}")
    missing_images = []
    for img_info in data['images'][:10]:  # Check first 10 images
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img_info['file_name'])
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images missing from first 10 checked")
        print("Missing images:", missing_images[:5])
    
    # Check annotations
    print(f"Number of annotations: {len(data['annotations'])}")
    
    # Check categories
    print(f"Categories: {data['categories']}")
    
    print("Dataset validation completed!")

def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(optimizer, epoch, config, warmup_epochs=5):
    """Adjust learning rate with warmup"""
    if epoch < warmup_epochs:
        # Warmup
        lr = config.learning_rate * (epoch + 1) / warmup_epochs
    else:
        # Regular schedule
        lr = config.learning_rate * (0.1 ** (epoch // (config.num_epochs // 3)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def count_parameters(model):
    """Count model parameters by layer"""
    layer_params = {}
    
    for name, param in model.named_parameters():
        layer_name = '.'.join(name.split('.')[:-1])  # Remove parameter name (weight/bias)
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += param.numel()
    
    return layer_params

def log_system_info():
    """Log system information"""
    print("System Information:")
    print("-" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("-" * 40)

def format_time(seconds):
    """Format time in seconds to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_dict(input_dict, average=True):
    """Reduce dictionary values across distributed processes"""
    if not dist.is_available() or not dist.is_initialized():
        return input_dict
    
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict

def all_gather(data):
    """Gather data from all processes"""
    if not dist.is_available() or not dist.is_initialized():
        return [data]
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    
    # Serialize data
    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_bytes = buffer.getvalue()
    
    # Get size of data
    local_size = torch.tensor([len(data_bytes)], dtype=torch.long, device='cuda')
    size_list = [torch.tensor([0], dtype=torch.long, device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # Gather data
    max_size = max(size_list).item()
    padded_data = data_bytes + b'\0' * (max_size - len(data_bytes))
    
    gathered_data = [torch.empty(max_size, dtype=torch.uint8, device='cuda') for _ in range(world_size)]
    dist.all_gather(gathered_data, torch.frombuffer(padded_data, dtype=torch.uint8).cuda())
    
    # Deserialize
    result = []
    for i, size in enumerate(size_list):
        data_bytes = gathered_data[i][:size.item()].cpu().numpy().tobytes()
        buffer = io.BytesIO(data_bytes)
        result.append(torch.load(buffer))
    
    return result

def check_dinov3_installation(repo_dir):
    """Check if DINOv3 repository is properly set up"""
    required_files = [
        'dinov3/models/vision_transformer.py',
        'dinov3/models/convnext.py',
        'hubconf.py'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(repo_dir, file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Required DINOv3 file not found: {full_path}")
    
    print("DINOv3 repository validation passed!")

def verify_config(config):
    """Verify configuration settings"""
    print("Verifying configuration...")
    
    # Check paths exist
    paths_to_check = [
        ('train_json_path', config.train_json_path),
        ('test_json_path', config.test_json_path),
        ('image_dir', config.image_dir),
        ('dinov3_repo_dir', config.dinov3_repo_dir),
        ('backbone_weights_path', config.backbone_weights_path)
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    
    # Check DINOv3 setup
    check_dinov3_installation(config.dinov3_repo_dir)
    
    # Validate dataset
    validate_dataset_format(config.train_json_path, config.image_dir)
    validate_dataset_format(config.test_json_path, config.image_dir)
    
    print("Configuration verification completed!")

# For backwards compatibility
import io
