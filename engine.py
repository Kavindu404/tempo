import os, json, torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from visualization import save_epoch_samples

def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

@torch.no_grad()
def evaluate_epoch(cfg, model, criterion, matcher, loader, device, epoch, writer=None):
    model.eval()
    total_loss = 0.0
    iou_all, dice_all = [], []

    coco_results = []
    for ib, (images, targets) in enumerate(tqdm(loader, desc=f"Eval {epoch}")):
        images = images.to(device, non_blocking=True)
        targets = [{k:(v.to(device) if torch.is_tensor(v) else v) for k,v in t.items()} for t in targets]

        outputs = model(images)
        losses, indices = criterion(outputs, targets)
        total_loss += float(losses["loss_total"])

        # IoU/Dice on matched pairs
        pred_masks = outputs["pred_masks"]             # [B,Q,H,W] logits
        for b,(src_idx,tgt_idx) in enumerate(indices):
            if tgt_idx.numel()==0: continue
            p = pred_masks[b][src_idx].sigmoid() > 0.5
            g = targets[b]["masks"][tgt_idx].bool()
            # flatten per-instance IoU, Dice
            for pi,gi in zip(p, g):
                inter = (pi & gi).sum().item()
                u = (pi | gi).sum().item()
                iou = inter / (u + 1e-6)
                dice = (2*inter) / (pi.sum().item() + gi.sum().item() + 1e-6)
                iou_all.append(iou); dice_all.append(dice)

        # One batch of visualizations per eval pass
        if ib == 0 and cfg.num_viz_samples > 0:
            save_epoch_samples(cfg, epoch, images, targets, outputs, writer)

        # COCO segm predictions
        probs = outputs["pred_logits"].softmax(-1)
        masks = outputs["pred_masks"].sigmoid()
        B,Q,H,W = masks.shape
        for b in range(B):
            image_id = int(targets[b]["image_id"])
            scores, labels = probs[b, :, :-1].max(-1)
            keep = scores > 0.05
            for s, c, m in zip(scores[keep], labels[keep], masks[b][keep]):
                m_bin = (m > 0.5).to(torch.uint8).cpu().numpy()
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(c),
                    "score": float(s),
                    "segmentation": _mask_to_rle(m_bin)
                })

    avg_loss = total_loss / max(1, len(loader))
    mean_iou = float(np.mean(iou_all)) if len(iou_all)>0 else 0.0
    mean_dice = float(np.mean(dice_all)) if len(dice_all)>0 else 0.0

    # COCO mAP
    tmp_json = os.path.join(cfg.checkpoints_dir, cfg.experiment, f"_tmp_results_epoch{epoch}.json")
    os.makedirs(os.path.dirname(tmp_json), exist_ok=True)
    with open(tmp_json, "w") as f:
        json.dump(coco_results, f)

    coco_gt = COCO(cfg.val_ann_file)
    coco_dt = coco_gt.loadRes(tmp_json) if len(coco_results) else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    mAP = float(coco_eval.stats[0]) if coco_eval.stats is not None else 0.0

    if writer is not None:
        writer.add_scalar("eval/loss", avg_loss, epoch)
        writer.add_scalar("eval/mAP_segm", mAP, epoch)
        writer.add_scalar("eval/mIoU", mean_iou, epoch)
        writer.add_scalar("eval/mDice", mean_dice, epoch)

    return {"loss": avg_loss, "mAP": mAP, "mIoU": mean_iou, "mDice": mean_dice}

def train_one_epoch(cfg, model, criterion, optimizer, loader, device, epoch, scaler, writer=None):
    model.train()
    total = 0.0
    for images, targets in tqdm(loader, desc=f"Train {epoch}"):
        images = images.to(device, non_blocking=True)
        targets = [{k:(v.to(device) if torch.is_tensor(v) else v) for k,v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', enabled=cfg.amp):
            outputs = model(images)
            losses, _ = criterion(outputs, targets)
            loss = losses["loss_total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())

    avg = total / max(1, len(loader))
    if writer is not None:
        writer.add_scalar("train/loss", avg, epoch)
    return avg
