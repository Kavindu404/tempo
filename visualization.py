import os, cv2, random
import numpy as np
import torch
from typing import List

def _color_map(n=256):
    np.random.seed(123)
    return np.random.randint(0, 255, size=(n,3), dtype=np.uint8)

CMAP = _color_map(1024)

def overlay_masks_and_boxes(img: np.ndarray, masks: torch.Tensor, boxes: torch.Tensor, alpha=0.45):
    """
    img: uint8 [H,W,3] (RGB)
    masks: [N,H,W] bool
    boxes: [N,4] xyxy
    """
    out = img.copy()
    H,W = out.shape[:2]
    if masks is not None and masks.numel()>0:
        m = masks.cpu().numpy().astype(bool)
        for i in range(m.shape[0]):
            color = CMAP[i % len(CMAP)].tolist()
            colored = np.zeros_like(out); colored[m[i]] = color
            out = cv2.addWeighted(out, 1.0, colored, alpha, 0)
    if boxes is not None and boxes.numel()>0:
        for b in boxes.cpu().numpy().astype(int):
            x0,y0,x1,y1 = b
            x0 = np.clip(x0, 0, W-1); x1 = np.clip(x1, 0, W-1)
            y0 = np.clip(y0, 0, H-1); y1 = np.clip(y1, 0, H-1)
            cv2.rectangle(out, (x0,y0), (x1,y1), (255,255,255), 2)
    return out

def save_epoch_samples(cfg, epoch: int, batch_images: torch.Tensor, batch_targets: List[dict],
                       batch_outputs: dict, writer=None):
    """
    Saves up to cfg.num_viz_samples images from the provided batch.
    Path: viz/<exp>/<image_name>_<epoch>_<k>.png
    Also logs a grid to TensorBoard if writer is provided.
    """
    os.makedirs(os.path.join(cfg.viz_dir, cfg.experiment), exist_ok=True)
    B = batch_images.size(0)
    k = min(cfg.num_viz_samples, B)
    idxs = random.sample(range(B), k=k)

    tb_imgs = []
    for si, i in enumerate(idxs):
        img_t = batch_images[i].detach().cpu()  # [3,H,W], normalized by A.Normalize(mean=0,std=1) — display by clipping
        # Convert back to 0..255 approximate
        img = (np.clip(img_t.numpy().transpose(1,2,0), 0, 1) * 255).astype(np.uint8)

        # preds
        logits = batch_outputs["pred_logits"][i].softmax(-1)[:,:-1]  # exclude no-object
        labels = torch.argmax(logits, dim=-1)
        scores = torch.max(logits, dim=-1).values
        keep = scores > 0.5
        pred_masks = (batch_outputs["pred_masks"][i][keep].sigmoid() > 0.5)
        # crude boxes from masks
        if pred_masks.numel()>0:
            bxs = []
            for m in pred_masks:
                ys,xs = torch.where(m)
                if len(xs)==0:
                    bxs.append(torch.tensor([0,0,0,0], device=m.device))
                else:
                    bxs.append(torch.tensor([xs.min(), ys.min(), xs.max()+1, ys.max()+1], device=m.device))
            pred_boxes = torch.stack(bxs,0).float()
        else:
            pred_boxes = torch.zeros((0,4), device=batch_images.device)

        # gts
        gt_masks = batch_targets[i]["masks"]
        gt_boxes = batch_targets[i]["boxes"]

        pred_overlay = overlay_masks_and_boxes(img, pred_masks, pred_boxes)
        gt_overlay   = overlay_masks_and_boxes(img, gt_masks, gt_boxes)

        vis = np.concatenate([img, gt_overlay, pred_overlay], axis=1)  # [H, 3W, 3]

        fname = f"{batch_targets[i]['file_name'].split('/')[-1].split('.')[0]}_{epoch}_{si}.png"
        path = os.path.join(cfg.viz_dir, cfg.experiment, fname)
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        if writer is not None:
            # to TensorBoard as CHW in 0..1
            tb = vis.astype(np.float32)/255.0
            tb_imgs.append(torch.from_numpy(tb.transpose(2,0,1)))

    if writer is not None and len(tb_imgs)>0:
        grid = torch.stack(tb_imgs[:min(8, len(tb_imgs))],0)  # up to 8
        writer.add_images(f"viz/epoch_{epoch}", grid, epoch)
