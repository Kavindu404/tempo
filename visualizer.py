import os
import cv2
import numpy as np
from utils import mkdir_p, denorm_img

def contours_from_mask(mask):
    # mask: (H,W) uint8 {0,1}
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(img_rgb, contours, color=(0,255,0), thickness=2):
    canvas = img_rgb.copy()
    cv2.drawContours(canvas, contours, -1, color, thickness)
    return canvas

def save_visualizations(batch_imgs, batch_targets, batch_preds, save_dir, epoch, n_limit=10, score_thresh=0.5):
    """
    batch_imgs: (B,3,H,W) tensor normalized
    batch_targets: list of target dicts (with 'masks','file_name')
    batch_preds: list of dicts with 'scores','labels','masks' (H,W) float 0..1
    """
    mkdir_p(save_dir)
    B = min(len(batch_targets), n_limit)
    for i in range(B):
        img = denorm_img(batch_imgs[i])
        gt_masks = batch_targets[i]["masks"].cpu().numpy()  # (N,H,W)
        # combine GT contours
        gt_canvas = img.copy()
        for m in gt_masks:
            cnts = contours_from_mask(m)
            gt_canvas = draw_contours(gt_canvas, cnts, color=(255,0,0), thickness=2)

        # predictions over threshold
        preds = batch_preds[i]
        pmasks = preds["masks"]  # list of (H,W) float
        pscores = preds["scores"]
        pred_canvas = img.copy()
        for m, s in zip(pmasks, pscores):
            if s < score_thresh: 
                continue
            m_bin = (m >= 0.5).astype(np.uint8)
            cnts = contours_from_mask(m_bin)
            pred_canvas = draw_contours(pred_canvas, cnts, color=(0,255,0), thickness=2)

        # stack: original | GT contours | Pred contours
        vis = np.concatenate([img, gt_canvas, pred_canvas], axis=1)
        fn = batch_targets[i]["file_name"]
        base = os.path.splitext(os.path.basename(fn))[0]
        out_path = os.path.join(save_dir, f"{base}_{epoch}_{i}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
