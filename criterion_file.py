import torch
import torch.nn as nn
import torch.nn.functional as F
from matcher import HungarianMatcher, generalized_box_iou, box_xyxy_to_cxcywh

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        
        masks = [t["masks"] for t in targets]
        target_masks, valid = self._nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        
        target_masks = target_masks.flatten(1)
        
        losses = {
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks.float()),
            "loss_dice": self._dice_loss(src_masks, target_masks)
        }
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_masks):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        target_boxes = box_xyxy_to_cxcywh(target_boxes)
        sizes = torch.cat([t["orig_size"] for t in targets])
        target_boxes = target_boxes / torch.cat([sizes, sizes], dim=1)[tgt_idx[1]]
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_masks
        
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_masks
        return losses
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def _dice_loss(self, inputs, targets):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1).float()
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()
    
    def _nested_tensor_from_tensor_list(self, tensor_list):
        if tensor_list[0].ndim == 3:
            max_size = [max([img.shape[i] for img in tensor_list]) for i in range(3)]
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError('not supported')
        return NestedTensor(tensor, mask)
    
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        indices = self.matcher(outputs_without_aux, targets)
        
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_masks)
            num_masks = torch.clamp(num_masks / torch.distributed.get_world_size(), min=1).item()
        else:
            num_masks = torch.clamp(num_masks, min=1).item()
        
        losses = {}
        for loss in ['labels', 'masks', 'boxes']:
            losses.update(getattr(self, f'loss_{loss}')(outputs_without_aux, targets, indices, num_masks))
        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in ['labels', 'masks', 'boxes']:
                    l_dict = getattr(self, f'loss_{loss}')(aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses

class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask
    
    def decompose(self):
        return self.tensors, self.mask