import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataset import CustomDataset
from model import DINOv3Mask2Former
from criterion import SetCriterion
from matcher import HungarianMatcher
from visualizer import Visualizer
from engine import coco_evaluate
from config import Config
from utils import save_checkpoint, setup_logger

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def main(rank, world_size, config):
    setup_ddp(rank, world_size)

    # Logger setup
    logger = setup_logger(os.path.join("logs", config.experiment_name))
    if rank == 0:
        logger.info("Starting training with DINOv3 + Mask2Former")

    # Dataset & Dataloader
    train_dataset = CustomDataset(
        json_path=config.train_json,
        image_dir=config.image_dir,
        transforms=True
    )
    val_dataset = CustomDataset(
        json_path=config.val_json,
        image_dir=config.image_dir,
        transforms=False
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Model
    model = DINOv3Mask2Former(config)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Matcher & Criterion
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    criterion = SetCriterion(
        config.num_classes,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2, "loss_mask": 1},
        eos_coef=0.1,
    ).to(rank)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_drop, gamma=0.1)

    # Visualizer
    if rank == 0:
        viz_dir = os.path.join("viz", config.experiment_name)
        os.makedirs(viz_dir, exist_ok=True)
        visualizer = Visualizer(config, viz_dir)
    else:
        visualizer = None

    # Training Loop
    best_map = 0.0
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        criterion.train()

        pbar = tqdm(train_loader, disable=(rank != 0))
        epoch_loss = 0.0

        for i, batch in enumerate(pbar):
            images = batch["images"].to(rank, non_blocking=True)
            targets = [{k: v.to(rank) for k, v in t.items()} for t in batch["targets"]]

            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            if rank == 0:
                pbar.set_description(f"Epoch [{epoch+1}/{config.epochs}] Loss: {losses.item():.4f}")

        scheduler.step()

        # Evaluate on Validation Set
        if rank == 0:
            eval_results = coco_evaluate(model, val_loader, device=rank)
            segm_map = eval_results["segm"]["mAP"]
            bbox_map = eval_results["bbox"]["mAP"]

            # Log results
            log_path = os.path.join("logs", config.experiment_name, f"{epoch+1}_logs.txt")
            with open(log_path, "w") as f:
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Train Loss: {epoch_loss/len(train_loader):.4f}\n")
                f.write(f"Segm mAP: {segm_map:.4f}\n")
                f.write(f"BBox mAP: {bbox_map:.4f}\n")

            # Save best checkpoint
            if segm_map > best_map:
                best_map = segm_map
                ckpt_path = os.path.join("checkpoints", config.experiment_name)
                os.makedirs(ckpt_path, exist_ok=True)
                save_checkpoint(
                    model.module,
                    optimizer,
                    epoch,
                    segm_map,
                    ckpt_path
                )

            # Save visualization samples
            visualizer.save_predictions(model, val_loader, epoch)

    cleanup_ddp()

if __name__ == "__main__":
    config = Config()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, config), nprocs=world_size)
