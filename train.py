import os
import json
import math
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from dataset import CocoSingleClassDataset, collate_fn
from matcher import HungarianMatcher
from criterion import SetCriterion
from model import DINOv3Backbone, Mask2FormerLite
from engine import train_one_epoch, evaluate, postprocess
from utils import init_distributed_mode, is_main_process, mkdir_p, set_seed
from visualizer import save_visualizations

def build_model(cfg: Config, device):
    backbone = DINOv3Backbone(
        repo_dir=cfg.dinov3_repo_dir,
        backbone_name=cfg.backbone_name,
        weights_path=cfg.backbone_weights,
        freeze=cfg.freeze_backbone,
        out_stride=cfg.out_stride,
    )
    model = Mask2FormerLite(
        backbone=backbone,
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        num_queries=cfg.num_queries,
        num_decoder_layers=cfg.num_decoder_layers,
        mask_dim=cfg.mask_dim,
        use_bias=cfg.use_bias,
    )
    return model.to(device)

def main():
    cfg = Config()
    set_seed(cfg.seed)

    # DDP
    try:
        rank, world_size, local_gpu = init_distributed_mode(cfg.backend)
    except Exception:
        rank, world_size, local_gpu = (0,1,0)

    device = torch.device("cuda", local_gpu) if torch.cuda.is_available() else torch.device("cpu")

    # Dirs
    out_logs = os.path.join(cfg.output_root, "logs", cfg.exp_name)
    out_ckpt = os.path.join(cfg.output_root, "checkpoints", cfg.exp_name)
    out_viz  = os.path.join(cfg.output_root, "viz", cfg.exp_name)
    if is_main_process():
        mkdir_p(out_logs); mkdir_p(out_ckpt); mkdir_p(out_viz)

    # Data
    train_set = CocoSingleClassDataset(cfg.train_json, cfg.image_dir, image_size=cfg.image_size, augment=cfg.augment)
    val_set   = CocoSingleClassDataset(cfg.val_json,   cfg.image_dir, image_size=cfg.image_size, augment=False)

    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size>1 else None
    val_sampler   = DistributedSampler(val_set, shuffle=False) if world_size>1 else None

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                              sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Model/loss
    model = build_model(cfg, device)
    matcher = HungarianMatcher(cls_cost=cfg.cls_weight, bbox_cost=cfg.bbox_l1_weight,
                               giou_cost=cfg.bbox_giou_weight, mask_cost=cfg.mask_focal_weight, dice_cost=cfg.mask_dice_weight)
    weights = {"cls": cfg.cls_weight, "bbox": cfg.bbox_l1_weight, "giou": cfg.bbox_giou_weight,
               "mask_focal": cfg.mask_focal_weight, "mask_dice": cfg.mask_dice_weight}
    criterion = SetCriterion(cfg.num_classes, matcher, weights, no_object_weight=cfg.no_object_weight,
                             cls_alpha=cfg.cls_alpha, cls_gamma=cfg.cls_gamma).to(device)

    # Optimizer: lower LR on (possibly frozen) backbone
    bb_params = []
    head_params = []
    for n,p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if "backbone" in n:
            bb_params.append(p)
        else:
            head_params.append(p)
    optim = AdamW([
        {"params": head_params, "lr": cfg.lr, "weight_decay": cfg.wd},
        {"params": bb_params, "lr": cfg.lr * cfg.backbone_lr_mult, "weight_decay": cfg.wd},
    ])

    def lr_lambda(ep):
        if ep < cfg.warmup_epochs:
            return float(ep + 1) / float(max(1, cfg.warmup_epochs))
        return 1.0
    sched = LambdaLR(optim, lr_lambda=lr_lambda)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # Wrap DDP
    if dist.is_initialized():
        find_unused = cfg.find_unused_params
        model = DDP(model, device_ids=[local_gpu], output_device=local_gpu, find_unused_parameters=find_unused)

    best_segm_ap = -1.0

    for epoch in range(1, cfg.epochs+1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, optim, train_loader, device, epoch, scaler, grad_clip=cfg.grad_clip_norm)
        sched.step()

        # Eval
        bbox_stats = segm_stats = {"AP": 0.0}
        if cfg.eval_every_epoch:
            # model.module if DDP
            eval_model = model.module if isinstance(model, DDP) else model
            bbox_stats, segm_stats = evaluate(eval_model, val_loader, device, cfg.val_json)

        # Logging per epoch
        if is_main_process():
            log_path = os.path.join(out_logs, f"{epoch}_logs.txt")
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "train": train_stats,
                    "bbox_eval": bbox_stats,
                    "segm_eval": segm_stats
                }, indent=2))

        # Save best checkpoints by segm AP
        cur_ap = float(segm_stats.get("AP", 0.0))
        is_best = cur_ap > best_segm_ap
        if is_main_process() and is_best:
            best_segm_ap = cur_ap
            base = os.path.join(out_ckpt, f"{epoch}_{best_segm_ap:.4f}.pt")
            to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model": to_save,
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "best_segm_AP": best_segm_ap,
                "config": cfg.__dict__,
            }, base)

        # Save visualizations (n samples from *this* epoch's first val batch)
        if is_main_process():
            eval_model = model.module if isinstance(model, DDP) else model
            eval_model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    outputs = eval_model(images)
                    preds = postprocess(outputs, targets)
                    # format for visualizer
                    pack = []
                    for p in preds:
                        pack.append({
                            "scores": p["scores"],
                            "labels": p["labels"],
                            "masks": p["masks"],  # (Q,H,W)
                        })
                    save_visualizations(images, targets, pack, out_viz, epoch, n_limit=cfg.viz_n_per_epoch, score_thresh=cfg.viz_score_thresh)
                    break  # only one batch per epoch
            eval_model.train()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
