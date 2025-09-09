import os
import time
import torch
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from tqdm import tqdm
import json
import tempfile
from visualizer import Visualizer, get_image_names_from_targets

def is_main_process():
    """Check if this is the main process in distributed training"""
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

class COCOEvaluator:
    """COCO evaluation wrapper"""
    
    def __init__(self, coco_gt, iou_types):
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
    
    def update(self, predictions):
        """Update with predictions"""
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            
            # Create temporary COCO result object
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(results, f)
                temp_file = f.name
            
            try:
                coco_dt = self.coco_gt.loadRes(temp_file) if results else COCO()
                coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)
                coco_eval.params.imgIds = list(img_ids)
                coco_eval.evaluate()
                coco_eval.accumulate()
                self.eval_imgs[iou_type].append(coco_eval.evalImgs)
            finally:
                os.unlink(temp_file)
    
    def prepare(self, predictions, iou_type):
        """Prepare predictions for COCO evaluation"""
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError(f"Unknown iou_type {iou_type}")
    
    def prepare_for_coco_detection(self, predictions):
        """Prepare bbox predictions"""
        coco_results = []
        for image_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
                
            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]
            
            # Convert to COCO format: [x, y, w, h]
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            
            for i in range(len(boxes)):
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": boxes[i].tolist(),
                    "score": float(scores[i])
                })
        
        return coco_results
    
    def prepare_for_coco_segmentation(self, predictions):
        """Prepare segmentation predictions"""
        coco_results = []
        for image_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
                
            masks = prediction["masks"]
            scores = prediction["scores"]
            labels = prediction["labels"]
            
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            
            for i in range(len(masks)):
                # Convert mask to RLE
                mask = masks[i]
                rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "segmentation": rle,
                    "score": float(scores[i])
                })
        
        return coco_results
    
    def synchronize_between_processes(self):
        """Synchronize evaluation across processes"""
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
    
    def accumulate(self):
        """Accumulate evaluation results"""
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()
    
    def summarize(self):
        """Summarize evaluation results"""
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """Create common COCO evaluation"""
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

def merge(img_ids, eval_imgs):
    """Merge evaluation results"""
    all_img_ids = {}
    all_eval_imgs = {}
    for p in img_ids:
        all_img_ids.update(p)
    for p in eval_imgs:
        all_eval_imgs.update(p)
    
    merged_img_ids = []
    for img_id in all_img_ids:
        merged_img_ids.append(img_id)
    
    merged_eval_imgs = []
    for img_id in merged_img_ids:
        if img_id in all_eval_imgs:
            merged_eval_imgs.append(all_eval_imgs[img_id])
    
    return merged_img_ids, merged_eval_imgs

def convert_to_coco_api(ds):
    """Convert dataset to COCO API format"""
    coco_ds = COCO()
    # Initialize COCO dataset
    coco_ds.dataset = {
        'images': [],
        'categories': [{'id': 1, 'name': 'object'}],  # Single class
        'annotations': []
    }
    coco_ds.createIndex()
    return coco_ds

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, config, epoch):
    """Evaluation function"""
    model.eval()
    criterion.eval()
    
    # Create evaluator
    # Note: For proper COCO evaluation, you'd need to create a COCO object from your dataset
    # For now, we'll compute basic metrics
    
    total_loss = 0
    num_batches = 0
    all_predictions = {}
    
    # Progress bar
    if is_main_process():
        pbar = tqdm(data_loader, desc=f"Eval Epoch {epoch}")
    else:
        pbar = data_loader
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        num_batches += 1
        
        # Post-process predictions for evaluation
        processed_outputs = post_process_outputs(outputs, targets, config.eval_threshold)
        
        # Collect predictions by image_id
        for i, target in enumerate(targets):
            img_id = target['image_id'].item()
            all_predictions[img_id] = processed_outputs[i]
        
        if is_main_process():
            pbar.set_postfix({'loss': losses.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Compute mAP (simplified version)
    # For a complete implementation, you'd use COCOeval with proper COCO format
    mAP_segm = 0.0  # Placeholder
    mAP_bbox = 0.0  # Placeholder
    
    # Log results
    if is_main_process():
        print(f"Evaluation - Loss: {avg_loss:.4f}, mAP(segm): {mAP_segm:.4f}, mAP(bbox): {mAP_bbox:.4f}")
    
    return {
        'loss': avg_loss,
        'mAP_segm': mAP_segm,
        'mAP_bbox': mAP_bbox
    }

def post_process_outputs(outputs, targets, threshold=0.5):
    """Post-process model outputs for evaluation"""
    processed = []
    
    pred_logits = outputs['pred_logits']  # [B, Q, num_classes+1]
    pred_masks = outputs['pred_masks']    # [B, Q, H, W]
    pred_boxes = outputs['pred_boxes']    # [B, Q, 4]
    
    batch_size = pred_logits.shape[0]
    
    for i in range(batch_size):
        # Apply softmax to get class probabilities
        scores = F.softmax(pred_logits[i], dim=-1)[:, 0]  # Class 1 scores (single class)
        
        # Filter by confidence threshold
        keep = scores > threshold
        
        if keep.sum() == 0:
            processed.append({
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty((0,)),
                'labels': torch.empty((0,), dtype=torch.long),
                'masks': torch.empty((0, pred_masks.shape[-2], pred_masks.shape[-1]))
            })
            continue
        
        # Keep only high-confidence predictions
        filtered_scores = scores[keep]
        filtered_boxes = pred_boxes[i][keep]
        filtered_masks = torch.sigmoid(pred_masks[i][keep])
        filtered_labels = torch.ones(keep.sum(), dtype=torch.long)  # All class 1
        
        processed.append({
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'labels': filtered_labels,
            'masks': filtered_masks
        })
    
    return processed

def train_one_epoch(model, criterion, data_loader, optimizer, device, config, epoch, visualizer=None):
    """Training function for one epoch"""
    model.train()
    criterion.train()
    
    total_loss = 0
    num_batches = 0
    loss_dict_total = {}
    
    # Progress bar
    if is_main_process():
        pbar = tqdm(data_loader, desc=f"Train Epoch {epoch}")
    else:
        pbar = data_loader
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping
        if config.gradient_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_max_norm)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses.item()
        num_batches += 1
        
        for k, v in loss_dict.items():
            if k not in loss_dict_total:
                loss_dict_total[k] = 0
            loss_dict_total[k] += v.item()
        
        if is_main_process():
            pbar.set_postfix({'loss': losses.item()})
        
        # Save visualizations for first batch of first few epochs
        if visualizer is not None and batch_idx == 0 and epoch <= 5 and is_main_process():
            try:
                image_names = get_image_names_from_targets(targets, config.image_dir)
                visualizer.save_batch_visualizations(
                    images, targets, outputs, image_names, epoch, config.num_viz_samples
                )
            except Exception as e:
                print(f"Error saving visualizations: {str(e)}")
    
    # Average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_total.items()}
    
    if is_main_process():
        print(f"Training - Epoch {epoch}, Loss: {avg_loss:.4f}")
        for k, v in avg_loss_dict.items():
            print(f"  {k}: {v:.4f}")
    
    return {
        'loss': avg_loss,
        'loss_dict': avg_loss_dict
    }

def save_checkpoint(model, optimizer, epoch, mAP, config, is_best=False):
    """Save model checkpoint"""
    if not is_main_process():
        return
    
    checkpoint_dir = config.exp_checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mAP': mAP,
        'config': config.__dict__
    }
    
    # Save with mAP in filename
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}_mAP_{mAP:.4f}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")

def log_metrics(metrics, epoch, config):
    """Log metrics to file"""
    if not is_main_process():
        return
    
    log_dir = config.exp_log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"epoch_{epoch:03d}_logs.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Epoch {epoch} Metrics:\n")
        f.write("=" * 40 + "\n")
        
        if 'train' in metrics:
            f.write("Training Metrics:\n")
            train_metrics = metrics['train']
            f.write(f"  Total Loss: {train_metrics['loss']:.4f}\n")
            if 'loss_dict' in train_metrics:
                for k, v in train_metrics['loss_dict'].items():
                    f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
        
        if 'eval' in metrics:
            f.write("Evaluation Metrics:\n")
            eval_metrics = metrics['eval']
            f.write(f"  Loss: {eval_metrics['loss']:.4f}\n")
            f.write(f"  mAP (segm): {eval_metrics['mAP_segm']:.4f}\n")
            f.write(f"  mAP (bbox): {eval_metrics['mAP_bbox']:.4f}\n")
        
        f.write(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Logged metrics to: {log_file}")
