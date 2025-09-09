import os
import json
import torch
from tqdm.auto import tqdm
from utils import reduce_dict, is_main_process, mkdir_p
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

@torch.no_grad()
def postprocess(outputs, targets):
    """
    Convert outputs to per-image predictions for viz/eval.
    Returns list of dicts with 'scores','labels','boxes','masks'
    """
    logits = outputs["pred_logits"]        # (B,Q,C+1)
    boxes = outputs["pred_boxes"]          # (B,Q,4) cxcywh norm
    masks = outputs["pred_masks"].sigmoid()# (B,Q,H,W)

    B, Q, _ = logits.shape
    prob = logits.softmax(-1)
    scores, labels = prob[..., :-1].max(-1)  # ignore no-object
    pred_list = []
    for b in range(B):
        s = scores[b].detach().cpu()
        l = labels[b].detach().cpu()
        m = masks[b].detach().cpu().numpy()
        bx = boxes[b].detach().cpu()
        # convert boxes to xyxy in pixels using mask size
        H, W = m.shape[-2:]
        xyxy = torch.zeros_like(bx)
        xyxy[:, 0] = (bx[:, 0] - bx[:, 2] / 2) * W
        xyxy[:, 1] = (bx[:, 1] - bx[:, 3] / 2) * H
        xyxy[:, 2] = (bx[:, 0] + bx[:, 2] / 2) * W
        xyxy[:, 3] = (bx[:, 1] + bx[:, 3] / 2) * H
        pred_list.append({
            "scores": s.numpy(),
            "labels": l.numpy(),
            "boxes": xyxy.numpy(),
            "masks": m,  # (Q,H,W)
        })
    return pred_list

def coco_eval_from_preds(coco_gt, preds, targets, iou_type="bbox"):
    """
    Build COCO-format predictions and run COCOeval.
    """
    results = []
    for img_tgt, p in zip(targets, preds):
        image_id = int(img_tgt["image_id"])
        scores = p["scores"]
        labels = p["labels"]
        boxes = p["boxes"]  # (Q,4) xyxy
        masks = p["masks"]  # (Q,H,W)
        H, W = masks.shape[-2:]

        for q in range(len(scores)):
            if iou_type == "bbox":
                x0,y0,x1,y1 = boxes[q].tolist()
                w = max(0., x1 - x0)
                h = max(0., y1 - y0)
                results.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x0,y0,w,h],
                    "score": float(scores[q]),
                })
            else:
                # segm: RLE via pycocotools
                import pycocotools.mask as mask_utils
                m_bin = (masks[q] >= 0.5).astype('uint8')
                rle = mask_utils.encode(np.asfortranarray(m_bin))
                rle["counts"] = rle["counts"].decode("ascii")
                results.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": float(scores[q]),
                })

    if len(results)==0:
        return {"AP": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats  # [12] for bbox, segm
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, scaler=None, grad_clip=1.0):
    model.train()
    running = {}
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        # move masks/boxes to device
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)
            t["masks"]  = t["masks"].to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                indices = criterion.matcher(outputs, targets)
                losses = criterion(outputs, targets, indices)
                loss = losses["loss_total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            indices = criterion.matcher(outputs, targets)
            losses = criterion(outputs, targets, indices)
            loss = losses["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.detach().cpu().item())

        pbar.set_postfix({k: f"{running[k]/(pbar.n+1):.3f}" for k in ["loss_total","loss_cls","loss_bbox","loss_giou","loss_mask_focal","loss_mask_dice"] if k in running})

    # average over iterations
    for k in running:
        running[k] /= max(len(data_loader),1)
    running = reduce_dict({k: torch.tensor(v, device=device) for k,v in running.items()})
    return {k: float(v.item()) for k, v in running.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, coco_json_path):
    model.eval()
    from pycocotools.coco import COCO
    coco_gt = COCO(coco_json_path)

    all_preds = []
    all_targets = []
    for images, targets in tqdm(data_loader, desc="Eval", leave=False):
        images = images.to(device)
        outputs = model(images)
        preds = postprocess(outputs, targets)
        all_preds.extend(preds)
        all_targets.extend(targets)

    # COCO bbox & segm
    bbox_stats = coco_eval_from_preds(coco_gt, all_preds, all_targets, iou_type="bbox")
    segm_stats = coco_eval_from_preds(coco_gt, all_preds, all_targets, iou_type="segm")
    return bbox_stats, segm_stats
