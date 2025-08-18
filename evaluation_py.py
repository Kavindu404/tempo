import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from fastcocoeval import FastCOCOeval
import json
import tempfile
import os


class SegmentationEvaluator:
    """Evaluator for segmentation tasks using COCO metrics"""
    
    def __init__(
        self,
        gt_file: str,
        class_names: List[str],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        use_fast_eval: bool = True
    ):
        self.gt_file = gt_file
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.use_fast_eval = use_fast_eval
        
        # Load ground truth
        self.coco_gt = COCO(gt_file)
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """Evaluate model on dataset"""
        model.eval()
        predictions = []
        
        self.logger.info("Starting evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    self.logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
                
                # Move to device
                images = batch['images'].to(device)
                image_ids = batch['image_ids']
                orig_sizes = batch['orig_sizes']
                
                # Forward pass
                outputs = model(images)
                
                # Post-process predictions
                batch_predictions = self._post_process_predictions(
                    outputs, image_ids, orig_sizes
                )
                predictions.extend(batch_predictions)
        
        # Evaluate predictions
        metrics = self._compute_coco_metrics(predictions)
        
        self.logger.info("Evaluation completed")
        return metrics
    
    def _post_process_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        image_ids: List[int],
        orig_sizes: torch.Tensor
    ) -> List[Dict]:
        """Post-process model outputs to COCO format"""
        pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [batch, num_queries, 4]
        pred_masks = outputs['pred_masks']    # [batch, num_queries, H, W]
        
        batch_size = pred_logits.shape[0]
        predictions = []
        
        for i in range(batch_size):
            img_id = image_ids[i]
            orig_h, orig_w = orig_sizes[i].tolist()
            
            # Get predictions for this image
            logits = pred_logits[i]  # [num_queries, num_classes+1]
            boxes = pred_boxes[i]    # [num_queries, 4]
            masks = pred_masks[i]    # [num_queries, H, W]
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            scores, labels = probs[:, :-1].max(dim=-1)  # Exclude no-object class
            
            # Filter by confidence
            keep = scores > self.confidence_threshold
            if not keep.any():
                continue
            
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            masks = masks[keep]
            
            # Convert normalized boxes to pixel coordinates
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
            
            # Convert masks to original size
            masks_resized = F.interpolate(
                masks.unsqueeze(1),
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Apply sigmoid and threshold
            masks_binary = (masks_resized.sigmoid() > 0.5).cpu().numpy()
            
            # Convert to COCO format
            for j in range(len(scores)):
                # Convert box format from [x1, y1, x2, y2] to [x, y, w, h]
                box = boxes[j].cpu().numpy()
                x1, y1, x2, y2 = box
                bbox = [x1, y1, x2 - x1, y2 - y1]
                
                # Convert mask to RLE
                mask = masks_binary[j].astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                prediction = {
                    'image_id': img_id,
                    'category_id': int(labels[j]) + 1,  # COCO categories start from 1
                    'bbox': [float(x) for x in bbox],
                    'segmentation': rle,
                    'score': float(scores[j])
                }
                predictions.append(prediction)
        
        return predictions
    
    def _compute_coco_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        """Compute COCO evaluation metrics"""
        if len(predictions) == 0:
            self.logger.warning("No predictions to evaluate")
            return {
                'mAP': 0.0,
                'mAP_50': 0.0,
                'mAP_75': 0.0,
                'mAP_s': 0.0,
                'mAP_m': 0.0,
                'mAP_l': 0.0
            }
        
        # Save predictions to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f)
            pred_file = f.name
        
        try:
            if self.use_fast_eval:
                # Use FastCOCOeval for faster evaluation
                evaluator = FastCOCOeval(self.gt_file, pred_file, 'segm')
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                
                stats = evaluator.stats
            else:
                # Use standard COCOeval
                coco_dt = self.coco_gt.loadRes(pred_file)
                coco_eval = COCOeval(self.coco_gt, coco_dt, 'segm')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                stats = coco_eval.stats
            
            # Extract metrics
            metrics = {
                'mAP': stats[0],        # AP @ IoU=0.50:0.95
                'mAP_50': stats[1],     # AP @ IoU=0.50
                'mAP_75': stats[2],     # AP @ IoU=0.75
                'mAP_s': stats[3],      # AP @ IoU=0.50:0.95 (small)
                'mAP_m': stats[4],      # AP @ IoU=0.50:0.95 (medium)
                'mAP_l': stats[5],      # AP @ IoU=0.50:0.95 (large)
            }
            
        finally:
            # Clean up temporary file
            os.unlink(pred_file)
        
        return metrics
    
    def evaluate_single_image(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        image_id: int,
        orig_size: Tuple[int, int],
        device: str = 'cuda'
    ) -> Dict:
        """Evaluate model on single image"""
        model.eval()
        
        with torch.no_grad():
            # Add batch dimension
            images = image.unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Post-process
            orig_sizes = torch.tensor([orig_size]).to(device)
            predictions = self._post_process_predictions(
                outputs, [image_id], orig_sizes
            )
        
        return predictions


class MetricTracker:
    """Track training and validation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, metrics: Dict[str, float], split: str = 'train'):
        """Update metrics for current epoch"""
        if split not in self.metrics:
            self.metrics[split] = {}
            self.history[split] = {}
        
        for key, value in metrics.items():
            self.metrics[split][key] = value
            
            if key not in self.history[split]:
                self.history[split][key] = []
            self.history[split][key].append(value)
    
    def get_current(self, split: str = 'train') -> Dict[str, float]:
        """Get current epoch metrics"""
        return self.metrics.get(split, {})
    
    def get_history(self, split: str = 'train') -> Dict[str, List[float]]:
        """Get metric history"""
        return self.history.get(split, {})
    
    def get_best(self, metric: str, split: str = 'val', mode: str = 'max') -> float:
        """Get best value for a metric"""
        if split not in self.history or metric not in self.history[split]:
            return 0.0 if mode == 'max' else float('inf')
        
        values = self.history[split][metric]
        return max(values) if mode == 'max' else min(values)
    
    def log_metrics(self, epoch: int, logger: logging.Logger):
        """Log current metrics"""
        for split, metrics in self.metrics.items():
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Epoch {epoch} [{split}] - {metric_str}")


class ConfusionMatrix:
    """Compute and track confusion matrix for segmentation"""
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update confusion matrix with batch predictions"""
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Remove ignored pixels
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        # Compute confusion matrix
        n = self.num_classes
        cm = np.bincount(n * target + pred, minlength=n**2).reshape(n, n)
        self.confusion_matrix += cm
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute segmentation metrics from confusion matrix"""
        cm = self.confusion_matrix
        
        # Per-class IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        
        # Mean IoU
        mean_iou = np.nanmean(iou)
        
        # Pixel accuracy
        pixel_acc = intersection.sum() / cm.sum()
        
        # Mean accuracy
        class_acc = intersection / (cm.sum(axis=1) + 1e-10)
        mean_acc = np.nanmean(class_acc)
        
        # Frequency weighted IoU
        freq = cm.sum(axis=1) / cm.sum()
        freq_weighted_iou = (freq * iou).sum()
        
        return {
            'mIoU': mean_iou,
            'pixel_acc': pixel_acc,
            'mean_acc': mean_acc,
            'freq_weighted_iou': freq_weighted_iou,
            'per_class_iou': iou.tolist()
        }


def compute_mask_iou(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """Compute IoU between predicted and ground truth masks"""
    # Apply threshold to predictions
    pred_binary = (pred_masks > threshold).float()
    gt_binary = gt_masks.float()
    
    # Compute intersection and union
    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + gt_binary.sum(dim=(-2, -1)) - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-10)
    
    return iou


def compute_dice_score(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """Compute Dice score between predicted and ground truth masks"""
    # Apply threshold to predictions
    pred_binary = (pred_masks > threshold).float()
    gt_binary = gt_masks.float()
    
    # Compute Dice score
    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
    dice = (2 * intersection) / (pred_binary.sum(dim=(-2, -1)) + gt_binary.sum(dim=(-2, -1)) + 1e-10)
    
    return dice


def evaluate_detection_metrics(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute detection metrics (AP, AR) for single image"""
    from torchvision.ops import box_iou
    
    if len(pred_boxes) == 0:
        return {'AP': 0.0, 'AR': 0.0}
    
    if len(gt_boxes) == 0:
        return {'AP': 0.0, 'AR': 0.0}
    
    # Compute IoU matrix
    ious = box_iou(pred_boxes, gt_boxes)
    
    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    for i, idx in enumerate(sorted_indices):
        pred_label = pred_labels[idx]
        
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_matched[gt_idx] or gt_labels[gt_idx] != pred_label:
                continue
            
            iou = ious[idx, gt_idx]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / (len(gt_boxes) + 1e-10)
    
    # Compute AP using trapezoidal rule
    ap = torch.trapz(precision, recall)
    
    # Compute AR (average recall)
    ar = recall[-1] if len(recall) > 0 else 0.0
    
    return {
        'AP': ap.item(),
        'AR': ar.item()
    }


def log_evaluation_results(
    metrics: Dict[str, float],
    epoch: int,
    logger: logging.Logger,
    prefix: str = "Validation"
):
    """Log evaluation results in a formatted way"""
    logger.info(f"\n{prefix} Results - Epoch {epoch}")
    logger.info("=" * 50)
    
    # Main metrics
    main_metrics = ['mAP', 'mAP_50', 'mAP_75']
    for metric in main_metrics:
        if metric in metrics:
            logger.info(f"{metric:>10s}: {metrics[metric]:.4f}")
    
    # Size-specific metrics
    size_metrics = ['mAP_s', 'mAP_m', 'mAP_l']
    logger.info("\nSize-specific metrics:")
    for metric in size_metrics:
        if metric in metrics:
            size_name = metric.split('_')[1]
            logger.info(f"  {size_name:>6s}: {metrics[metric]:.4f}")
    
    # Additional metrics
    other_metrics = {k: v for k, v in metrics.items() 
                    if k not in main_metrics + size_metrics}
    if other_metrics:
        logger.info("\nAdditional metrics:")
        for metric, value in other_metrics.items():
            logger.info(f"  {metric:>10s}: {value:.4f}")
    
    logger.info("=" * 50)