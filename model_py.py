import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
from scipy.optimize import linear_sum_assignment


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer decoder"""
    
    def __init__(self, hidden_dim: int, max_len: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HungarianMatcher(nn.Module):
    """Hungarian matching for object queries to ground truth assignment"""
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_mask: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_mask = cost_mask
    
    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_masks: torch.Tensor,
        valid: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            pred_logits: [batch_size, num_queries, num_classes + 1]
            pred_boxes: [batch_size, num_queries, 4]
            pred_masks: [batch_size, num_queries, H, W]
            gt_labels: [batch_size, max_objects]
            gt_boxes: [batch_size, max_objects, 4]
            gt_masks: [batch_size, max_objects, H, W]
            valid: [batch_size, max_objects] - mask for valid objects
        
        Returns:
            List of (pred_indices, gt_indices) for each batch item
        """
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Flatten to compute cost matrices
        pred_logits_flat = pred_logits.flatten(0, 1)  # [batch_size * num_queries, num_classes + 1]
        pred_boxes_flat = pred_boxes.flatten(0, 1)    # [batch_size * num_queries, 4]
        pred_masks_flat = pred_masks.flatten(0, 1)    # [batch_size * num_queries, H, W]
        
        # Compute class probabilities
        pred_probs = F.softmax(pred_logits_flat, dim=-1)
        
        indices = []
        
        for i in range(batch_size):
            # Get valid ground truth objects for this batch item
            valid_mask = valid[i]
            num_gt = valid_mask.sum().item()
            
            if num_gt == 0:
                # No ground truth objects, assign empty matching
                indices.append((
                    torch.empty(0, dtype=torch.long, device=pred_logits.device),
                    torch.empty(0, dtype=torch.long, device=pred_logits.device)
                ))
                continue
            
            gt_labels_i = gt_labels[i, :num_gt]
            gt_boxes_i = gt_boxes[i, :num_gt]
            gt_masks_i = gt_masks[i, :num_gt]
            
            # Classification cost
            cost_class = -pred_probs[i * num_queries:(i + 1) * num_queries, gt_labels_i]
            
            # Box cost (L1 + GIoU)
            pred_boxes_i = pred_boxes[i]
            cost_bbox = torch.cdist(pred_boxes_i, gt_boxes_i, p=1)
            cost_giou = -self._generalized_box_iou(pred_boxes_i, gt_boxes_i)
            cost_bbox = cost_bbox + cost_giou
            
            # Mask cost (Dice loss)
            pred_masks_i = pred_masks[i]
            cost_mask = self._dice_loss_cost(pred_masks_i, gt_masks_i)
            
            # Total cost
            C = (self.cost_class * cost_class + 
                 self.cost_bbox * cost_bbox + 
                 self.cost_mask * cost_mask)
            
            # Hungarian algorithm
            pred_indices, gt_indices = linear_sum_assignment(C.cpu().numpy())
            
            indices.append((
                torch.tensor(pred_indices, dtype=torch.long, device=pred_logits.device),
                torch.tensor(gt_indices, dtype=torch.long, device=pred_logits.device)
            ))
        
        return indices
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute generalized IoU between two sets of boxes"""
        # boxes format: [x1, y1, x2, y2] (normalized)
        
        # Compute intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # left-top
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # right-bottom
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        # Compute areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1[:, None] + area2[None, :] - inter
        
        # IoU
        iou = inter / (union + 1e-6)
        
        # Generalized IoU
        lt_max = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
        rb_max = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh_max = (rb_max - lt_max).clamp(min=0)
        area_max = wh_max[:, :, 0] * wh_max[:, :, 1]
        
        giou = iou - (area_max - union) / (area_max + 1e-6)
        
        return giou
    
    def _dice_loss_cost(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss between predicted and ground truth masks"""
        pred_masks = pred_masks.sigmoid()
        pred_masks = pred_masks.flatten(1)  # [num_queries, H*W]
        gt_masks = gt_masks.flatten(1).float()  # [num_gt, H*W]
        
        numerator = 2 * torch.einsum('nc,mc->nm', pred_masks, gt_masks)
        denominator = pred_masks.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        
        dice = numerator / (denominator + 1e-6)
        return 1 - dice


class SegmentationHead(nn.Module):
    """Segmentation head with mask prediction"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_queries: int = 100,
        mask_dim: int = 256
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=6
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = nn.MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Mask prediction
        self.mask_embed = nn.Linear(hidden_dim, mask_dim)
        
        # Positional encoding for spatial features
        self.pos_embed = PositionalEncoding(hidden_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        feature_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, hidden_dim] - global features from backbone
            feature_map: [batch_size, hidden_dim, H, W] - spatial feature map
        
        Returns:
            Dictionary with predicted logits, boxes, and masks
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Prepare spatial features for transformer
        h, w = feature_map.shape[-2:]
        spatial_features = feature_map.flatten(2).permute(2, 0, 1)  # [H*W, batch_size, hidden_dim]
        spatial_features = self.pos_embed(spatial_features)
        
        # Object queries
        query_embeds = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, batch_size, hidden_dim]
        
        # Transformer decoder
        decoded_features = self.transformer_decoder(
            query_embeds, spatial_features
        )  # [num_queries, batch_size, hidden_dim]
        
        decoded_features = decoded_features.transpose(0, 1)  # [batch_size, num_queries, hidden_dim]
        
        # Predictions
        pred_logits = self.class_embed(decoded_features)
        pred_boxes = self.bbox_embed(decoded_features).sigmoid()
        
        # Mask prediction
        mask_embeds = self.mask_embed(decoded_features)  # [batch_size, num_queries, mask_dim]
        
        # Project feature map for mask prediction
        mask_features = F.conv2d(
            feature_map,
            self.mask_embed.weight.view(-1, self.hidden_dim, 1, 1),
            bias=self.mask_embed.bias
        )  # [batch_size, mask_dim, H, W]
        
        # Compute masks by dot product
        pred_masks = torch.einsum('bqc,bchw->bqhw', mask_embeds, mask_features)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_masks': pred_masks
        }


class MLP(nn.Module):
    """Simple MLP with ReLU activation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DinoV3Segmentation(nn.Module):
    """Complete segmentation model with DinoV3 backbone"""
    
    def __init__(
        self,
        backbone_name: str = 'dinov3_vitl16',
        repo_dir: str = '.',
        weights: Optional[str] = None,
        num_classes: int = 80,
        num_queries: int = 100,
        hidden_dim: int = 1024,
        mask_dim: int = 256,
        aux_loss: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        
        # Load DinoV3 backbone
        self.backbone = torch.hub.load(
            repo_dir, backbone_name, 
            source='local', 
            weights=weights
        )
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get backbone output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone.forward_features(dummy_input)
            backbone_dim = dummy_features['x_norm_patchtokens'].shape[-1]
            
            # Get spatial dimensions for feature map
            patch_tokens = dummy_features['x_norm_patchtokens']
            num_patches = patch_tokens.shape[1]
            spatial_size = int(math.sqrt(num_patches))
        
        self.backbone_dim = backbone_dim
        self.spatial_size = spatial_size
        
        # Projection to hidden dimension
        self.input_proj = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            mask_dim=mask_dim
        )
        
        # Hungarian matcher
        self.matcher = HungarianMatcher()
        
        # Loss weights
        self.weight_class = 1.0
        self.weight_bbox = 5.0
        self.weight_mask = 2.0
        self.weight_dice = 1.0
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size = images.shape[0]
        
        # Extract features from backbone
        with torch.no_grad():
            backbone_features = self.backbone.forward_features(images)
        
        # Get patch tokens and reshape to spatial format
        patch_tokens = backbone_features['x_norm_patchtokens']  # [batch, num_patches, dim]
        patch_tokens = patch_tokens.view(
            batch_size, self.spatial_size, self.spatial_size, self.backbone_dim
        ).permute(0, 3, 1, 2)  # [batch, dim, h, w]
        
        # Project to hidden dimension
        feature_map = self.input_proj(patch_tokens)
        
        # Global features (class token)
        global_features = backbone_features['x_norm_clstoken']  # [batch, dim]
        global_features = F.linear(
            global_features, 
            self.input_proj.weight.view(self.input_proj.out_channels, -1),
            self.input_proj.bias
        )
        
        # Segmentation prediction
        outputs = self.segmentation_head(global_features, feature_map)
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute segmentation loss"""
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_masks = outputs['pred_masks']
        
        gt_labels = targets['labels']
        gt_boxes = targets['boxes']
        gt_masks = targets['masks']
        valid = targets['valid']
        
        # Hungarian matching
        indices = self.matcher(
            pred_logits, pred_boxes, pred_masks,
            gt_labels, gt_boxes, gt_masks, valid
        )
        
        # Compute losses
        loss_class = self._classification_loss(pred_logits, gt_labels, indices, valid)
        loss_bbox = self._bbox_loss(pred_boxes, gt_boxes, indices, valid)
        loss_mask = self._mask_loss(pred_masks, gt_masks, indices, valid)
        loss_dice = self._dice_loss(pred_masks, gt_masks, indices, valid)
        
        # Total loss
        total_loss = (
            self.weight_class * loss_class +
            self.weight_bbox * loss_bbox +
            self.weight_mask * loss_mask +
            self.weight_dice * loss_dice
        )
        
        return {
            'loss': total_loss,
            'loss_class': loss_class,
            'loss_bbox': loss_bbox,
            'loss_mask': loss_mask,
            'loss_dice': loss_dice
        }
    
    def _classification_loss(
        self,
        pred_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        valid: torch.Tensor
    ) -> torch.Tensor:
        """Classification loss with focal loss"""
        batch_size = pred_logits.shape[0]
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.long, device=pred_logits.device
        )
        
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[i, pred_idx] = gt_labels[i, gt_idx]
        
        return F.cross_entropy(
            pred_logits.view(-1, self.num_classes + 1),
            target_classes.view(-1)
        )
    
    def _bbox_loss(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        valid: torch.Tensor
    ) -> torch.Tensor:
        """Bounding box loss (L1 + GIoU)"""
        total_loss = 0.0
        num_boxes = 0
        
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            pred_boxes_i = pred_boxes[i, pred_idx]
            gt_boxes_i = gt_boxes[i, gt_idx]
            
            # L1 loss
            l1_loss = F.l1_loss(pred_boxes_i, gt_boxes_i, reduction='sum')
            
            # GIoU loss
            giou = self.matcher._generalized_box_iou(pred_boxes_i, gt_boxes_i)
            giou_loss = (1 - giou.diag()).sum()
            
            total_loss += l1_loss + giou_loss
            num_boxes += len(pred_idx)
        
        return total_loss / max(num_boxes, 1)
    
    def _mask_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        valid: torch.Tensor
    ) -> torch.Tensor:
        """Mask focal loss"""
        total_loss = 0.0
        num_masks = 0
        
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            pred_masks_i = pred_masks[i, pred_idx]
            gt_masks_i = gt_masks[i, gt_idx].float()
            
            # Focal loss
            pred_masks_sigmoid = pred_masks_i.sigmoid()
            focal_loss = self._focal_loss(pred_masks_sigmoid, gt_masks_i)
            
            total_loss += focal_loss
            num_masks += len(pred_idx)
        
        return total_loss / max(num_masks, 1)
    
    def _dice_loss(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        valid: torch.Tensor
    ) -> torch.Tensor:
        """Dice loss for masks"""
        total_loss = 0.0
        num_masks = 0
        
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            pred_masks_i = pred_masks[i, pred_idx].sigmoid()
            gt_masks_i = gt_masks[i, gt_idx].float()
            
            pred_flat = pred_masks_i.flatten(1)
            gt_flat = gt_masks_i.flatten(1)
            
            numerator = 2 * (pred_flat * gt_flat).sum(1)
            denominator = pred_flat.sum(1) + gt_flat.sum(1)
            
            dice_loss = 1 - (numerator / (denominator + 1e-6))
            total_loss += dice_loss.sum()
            num_masks += len(pred_idx)
        
        return total_loss / max(num_masks, 1)
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """Focal loss implementation"""
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            loss = alpha_t * loss
        
        return loss.mean()


def build_model(
    backbone_name: str = 'dinov3_vitl16',
    repo_dir: str = '.',
    weights: Optional[str] = None,
    num_classes: int = 80,
    num_queries: int = 100,
    hidden_dim: int = 1024
) -> DinoV3Segmentation:
    """Build segmentation model"""
    return DinoV3Segmentation(
        backbone_name=backbone_name,
        repo_dir=repo_dir,
        weights=weights,
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=hidden_dim
    )