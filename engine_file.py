import os
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import numpy as np
from visualizer import save_predictions
from collections import defaultdict

class MetricLogger:
    def __init__(self):
        self.meters = defaultdict(SmoothedValue)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.global_avg:.4f}")
        return " | ".join(loss_str)
    
    def get_avg(self, name):
        return self.meters[name].global_avg

class SmoothedValue:
    def __init__(self):
        self.total = 0.0
        self.count = 0
    
    def update(self, value, n=1):
        self.total += value * n
        self.count += n
    
    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, config):
    model.train()
    metric_logger = MetricLogger()
    
    header = f'Epoch: [{epoch}]'
    print_freq = config.print_freq
    
    progress_bar = tqdm(data_loader, desc=header) if dist.get_rank() == 0 else data_loader
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimizer.zero_grad()
        losses.backward()
        
        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
        
        optimizer.step()
        
        metric_logger.update(loss=losses, **loss_dict)
        
        if dist.get_rank() == 0 and batch_idx % print_freq == 0:
            progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return metric_logger

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, config, epoch=0, save_viz=False):
    model.eval()
    metric_logger = MetricLogger()
    
    coco_results = []
    img_ids = []
    
    viz_dir = None
    if save_viz and dist.get_rank() == 0:
        viz_dir = os.path.join(config.viz_dir, config.exp_name)
        os.makedirs(viz_dir, exist_ok=True)
    
    saved_viz = 0
    
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc='Validation')):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        metric_logger.update(**loss_dict)
        
        # Save visualizations for first few batches
        if save_viz and saved_viz < config.num_viz and dist.get_rank() == 0:
            n_to_save = min(config.num_viz - saved_viz, images.shape[0])
            save_predictions(
                images[:n_to_save],
                targets[:n_to_save],
                outputs,
                config,
                epoch,
                viz_dir
            )
            saved_viz += n_to_save
        
        # Process predictions for COCO eval
        pred_scores = outputs['pred_logits'].softmax(-1)[:, :, :-1]
        pred_masks = outputs['pred_masks'].sigmoid()
        pred_boxes = outputs['pred_boxes']
        
        for i, target in enumerate(targets):
            img_id = target['image_id'].item()
            img_ids.append(img_id)
            
            scores = pred_scores[i].max(-1)[0]
            keep = scores > 0.05
            
            if keep.any():
                boxes = pred_boxes[i][keep]
                masks = pred_masks[i][keep]
                scores = scores[keep]
                
                # Convert to COCO format
                h, w = target['orig_size'].tolist()
                boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
                boxes = box_cxcywh_to_xyxy(boxes).cpu().numpy()
                
                masks = F.interpolate(
                    masks.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                masks = (masks > 0.5).cpu().numpy()
                
                for j in range(len(scores)):
                    rle = mask_utils.encode(np.asfortranarray(masks[j].astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    
                    coco_results.append({
                        'image_id': img_id,
                        'category_id': 1,  # Single class
                        'bbox': boxes[j].tolist(),
                        'score': scores[j].item(),
                        'segmentation': rle
                    })
    
    # Gather results from all processes
    if dist.is_initialized():
        all_coco_results = [None] * dist.get_world_size()
        dist.all_gather_object(all_coco_results, coco_results)
        coco_results = sum(all_coco_results, [])
    
    # Compute COCO metrics
    if dist.get_rank() == 0 and len(coco_results) > 0:
        coco_eval_results = compute_coco_metrics(coco_results, data_loader.dataset, ['bbox', 'segm'])
        metric_logger.meters.update(coco_eval_results)
    
    return metric_logger

def compute_coco_metrics(results, dataset, iou_types):
    coco_gt = convert_to_coco_format(dataset)
    metrics = {}
    
    for iou_type in iou_types:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics[f'mAP_{iou_type}'] = coco_eval.stats[0]
        metrics[f'mAP_50_{iou_type}'] = coco_eval.stats[1]
        metrics[f'mAP_75_{iou_type}'] = coco_eval.stats[2]
    
    return metrics

def convert_to_coco_format(dataset):
    coco_format = {
        'images': dataset.images,
        'annotations': dataset.annotations,
        'categories': [{'id': 1, 'name': 'object'}]
    }
    return COCO(coco_format)

def save_checkpoint(model, optimizer, epoch, metric_logger, config):
    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if not is_main_process:
        return
    
    checkpoint_dir = os.path.join(config.checkpoint_dir, config.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    mAP = metric_logger.get_avg('mAP_segm')
    checkpoint_path = os.path.join(checkpoint_dir, f'{epoch}_{mAP:.4f}.pt')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mAP': mAP,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

def save_logs(epoch, metric_logger, config):
    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if not is_main_process:
        return
    
    log_dir = os.path.join(config.log_dir, config.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f'{epoch}_logs.txt')
    
    with open(log_path, 'w') as f:
        f.write(f'Epoch: {epoch}\n')
        f.write(f'{metric_logger}\n')
    
    print(f'Saved logs: {log_path}')

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

import torch.nn.functional as F
from pycocotools import mask as mask_utils