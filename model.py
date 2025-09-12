import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math
import copy

class DINOv3Mask2Former(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load DINOv3 backbone
        self.backbone = self._load_dinov3_backbone()
        
        # Get backbone configuration
        self.patch_size = getattr(self.backbone, 'patch_size', 16)
        
        # Determine backbone output dimension
        if hasattr(self.backbone, 'embed_dim'):
            self.backbone_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, 'num_features'):
            self.backbone_dim = self.backbone.num_features  
        else:
            # Fallback: probe with dummy input
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = self._extract_features(dummy)
                self.backbone_dim = out.shape[-1]
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Multi-scale feature projection
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone_dim, config.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, config.hidden_dim)
            )
        ])
        
        # Positional encoding
        self.pe_layer = PositionEmbeddingSine(config.hidden_dim // 2, normalize=True)
        
        # Mask embedder
        self.mask_embed = MLP(config.hidden_dim, config.hidden_dim, config.mask_dim, 3)
        
        # Query embeddings
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_dim)
        self.query_feat = nn.Embedding(config.num_queries, config.hidden_dim)
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            normalize_before=True
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=config.dec_layers,
            norm=nn.LayerNorm(config.hidden_dim),
            return_intermediate=True
        )
        
        # Output heads
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)
        self.bbox_embed = MLP(config.hidden_dim, config.hidden_dim, 4, 3)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        
        # Special initialization for class embed
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.config.num_classes + 1) * bias_value
        
        # Initialize bbox regression to predict centered boxes
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
    
    def _load_dinov3_backbone(self):
        backbone_name = f'dinov3_{self.config.backbone_type}'
        weights_path = self.config.get_backbone_weight_path()
        
        backbone = torch.hub.load(
            self.config.dinov3_repo_dir,
            backbone_name,
            source='local',
            weights=weights_path
        )
        
        # Set to eval mode if frozen
        if self.config.freeze_backbone:
            backbone.eval()
        
        return backbone
    
    def _extract_features(self, x):
        """Extract features from DINOv3 backbone"""
        if hasattr(self.backbone, 'forward_features'):
            # For Vision Transformer models
            features = self.backbone.forward_features(x)
            
            if isinstance(features, dict):
                # DINOv3 returns dict with different token types
                if 'x_norm_patchtokens' in features:
                    return features['x_norm_patchtokens']
                elif 'x' in features:
                    return features['x']
            else:
                # Remove CLS token if present
                if hasattr(self.backbone, 'global_pool'):
                    if self.backbone.global_pool == 'token':
                        return features[:, 1:]  # Remove CLS token
                return features
        else:
            # For ConvNeXt or other models
            features = self.backbone(x)
            if isinstance(features, dict):
                return features.get('features', features.get('x', features))
            return features
    
    def forward(self, images):
        batch_size = images.shape[0]
        h_orig, w_orig = images.shape[-2:]
        
        # Extract features from backbone
        features = self._extract_features(images)
        
        # Calculate spatial dimensions
        h = h_orig // self.patch_size
        w = w_orig // self.patch_size
        
        # Reshape to spatial format (B, C, H, W)
        if features.dim() == 3:  # (B, N, C) format from ViT
            features = features.transpose(1, 2).reshape(batch_size, self.backbone_dim, h, w)
        elif features.dim() == 2:  # (B, C) global pooled
            # Expand to spatial
            features = features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        # Project features
        srcs = []
        masks = []
        pos_embeds = []
        
        for level, proj in enumerate(self.input_proj):
            src = proj(features)
            mask = torch.zeros((batch_size, h, w), dtype=torch.bool, device=src.device)
            pos_embed = self.pe_layer(src, mask)
            
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos_embed)
        
        # Flatten for transformer
        src_flatten = []
        mask_flatten = []
        pos_embed_flatten = []
        
        for src, mask, pos_embed in zip(srcs, masks, pos_embeds):
            bs, c, h, w = src.shape
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed_flatten.append(pos_embed.flatten(2).transpose(1, 2))
        
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        pos_embed_flatten = torch.cat(pos_embed_flatten, 1)
        
        # Prepare queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Decoder forward
        hs, memory = self.decoder(
            query_feat.transpose(0, 1),
            src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=pos_embed_flatten.transpose(0, 1),
            query_pos=query_embed.transpose(0, 1)
        )
        
        # Generate outputs for each decoder layer
        outputs_class = []
        outputs_coord = []
        
        for lvl in range(hs.shape[0]):
            hs_lvl = hs[lvl]
            outputs_class.append(self.class_embed(hs_lvl))
            outputs_coord.append(self.bbox_embed(hs_lvl).sigmoid())
        
        outputs_class = torch.stack(outputs_class)
        outputs_coord = torch.stack(outputs_coord)
        
        # Generate masks using dot product
        mask_embed = self.mask_embed(hs[-1])  # (B, Q, C)
        
        # Reshape memory back to spatial
        memory_reshaped = src_flatten.transpose(1, 2).reshape(batch_size, self.config.hidden_dim, h, w)
        
        # Generate mask predictions via dot product
        outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, memory_reshaped)
        
        # Upsample masks to original resolution
        outputs_seg_masks = F.interpolate(
            outputs_seg_masks,
            size=(h_orig, w_orig),
            mode="bilinear",
            align_corners=False
        )
        
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_masks': outputs_seg_masks,
            'aux_outputs': [
                {'pred_logits': a, 'pred_boxes': b, 'pred_masks': outputs_seg_masks}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        }
        
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
    
    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(
                output, memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos
            )
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate), memory.transpose(0, 1)
        
        return output.unsqueeze(0), memory.transpose(0, 1)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        # Self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
