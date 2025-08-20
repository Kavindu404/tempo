import os, json, torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from configs import Config
from dataset import COCOMaskDataset, collate_fn
from matcher import HungarianMatcher
from criterion import SetCriterion
from model import DINOv3Mask2Former
from engine import train_one_epoch, evaluate_epoch
from utils import init_seeds, setup_exp_dirs, tb_writer, save_checkpoint, is_main_process, setup_ddp

def infer_num_categories(cfg):
    with open(cfg.train_ann_file, "r") as f:
        data = json.load(f)
    # COCO categories might be sparse ids; we treat "num_classes" as max index + 1 or len unique ids.
    # For mask classification, we need contiguous class indices; the head expects num_classes.
    cat_ids = sorted({c["id"] for c in data["categories"]})
    return len(cat_ids)

def main():
    cfg = Config()
    setup_ddp(cfg.dist_backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    init_seeds(cfg.seed + local_rank)

    # Infer categories if not set
    if cfg.num_categories is None:
        cfg.num_categories = infer_num_categories(cfg)

    ckpt_dir, log_dir, _ = setup_exp_dirs(cfg)
    writer = tb_writer(log_dir) if is_main_process() else None

    # Data
    train_set = COCOMaskDataset(cfg.train_img_dir, cfg.train_ann_file, img_size=cfg.img_size, is_train=True,
                                use_mosaic=cfg.use_mosaic, mosaic_prob=cfg.mosaic_prob,
                                use_mixup=cfg.use_mixup, mixup_prob=cfg.mixup_prob, mixup_alpha=cfg.mixup_alpha)
    val_set   = COCOMaskDataset(cfg.val_img_dir, cfg.val_ann_file, img_size=cfg.img_size, is_train=False,
                                use_mosaic=False, use_mixup=False)

    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler   = DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=cfg.batch_size, sampler=val_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Model
    model = DINOv3Mask2Former(cfg.repo_dir, cfg.hub_entry, cfg.num_categories,
                              cfg.head_weights, cfg.backbone_weights).to(device)

    # Freeze backbone for warmup epochs
    model.freeze_backbone(True)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Matcher + criterion
    matcher = HungarianMatcher(cfg.cost_class, cfg.cost_mask, cfg.cost_dice)
    criterion = SetCriterion(cfg.num_categories, matcher).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_map = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)

        # Freeze→Unfreeze trick
        if epoch == cfg.freeze_backbone_epochs + 1 and is_main_process():
            print(f"[epoch {epoch}] unfreezing backbone")
        if epoch <= cfg.freeze_backbone_epochs:
            model.module.freeze_backbone(True)
        else:
            model.module.freeze_backbone(False)

        train_loss = train_one_epoch(cfg, model, criterion, optimizer, train_loader, device, epoch, scaler, writer)

        # Sync before eval
        torch.distributed.barrier()

        metrics = {}
        if is_main_process():
            metrics = evaluate_epoch(cfg, model, criterion, matcher, val_loader, device, epoch, writer)
            path = save_checkpoint(model.module, optimizer, epoch, metrics, ckpt_dir, cfg.experiment)
            print(f"[checkpoint] {path}")

            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                best_path = os.path.join(ckpt_dir, f"{cfg.experiment}_BEST_mAP_{best_map:.4f}_epoch_{epoch}.pt")
                torch.save(torch.load(path), best_path)
                print(f"[best] {best_path}")

        torch.distributed.barrier()

    if writer and is_main_process(): writer.close()

if __name__ == "__main__":
    main()
