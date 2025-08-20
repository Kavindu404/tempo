import os, random, numpy as np, torch
from torch.utils.tensorboard import SummaryWriter

def init_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_exp_dirs(cfg):
    ckpt_dir = os.path.join(cfg.checkpoints_dir, cfg.experiment)
    log_dir  = os.path.join(cfg.logs_dir, cfg.experiment)
    viz_dir  = os.path.join(cfg.viz_dir, cfg.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    return ckpt_dir, log_dir, viz_dir

def tb_writer(log_dir):
    return SummaryWriter(log_dir)

def save_checkpoint(model_module, optimizer, epoch, metrics, ckpt_dir, experiment):
    fname = f"{experiment}_mAP_{metrics['mAP']:.4f}_epoch_{epoch}.pt"
    path = os.path.join(ckpt_dir, fname)
    torch.save({
        "epoch": epoch,
        "model": model_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics
    }, path)
    return path

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def setup_ddp(backend="nccl"):
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
