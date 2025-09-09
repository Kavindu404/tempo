import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional


# -------------------------
# Position Embedding (Sine)
# -------------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
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


# -------------------------
# Multi-Head Attention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        bs = query.size(0)

        # Linear projections
        q = self.q_linear(query).view(bs, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(bs, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(bs, -1, self.nhead, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(out)


# -------------------------
# Transformer Decoder Layer
# -------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# -------------------------
# Mask Head
# -------------------------
class MaskHead(nn.Module):
    def __init__(self, hidden_dim, fpn_dims, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.mask_embed = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_layers = nn.ModuleList([nn.Conv2d(fpn_dims[i], hidden_dim, 1) for i in range(len(fpn_dims))])

        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, features, query_embed):
        batch_size, num_queries, _ = query_embed.shape
        mask_embed = self.mask_embed(query_embed)

        processed_features = [self.decoder_layers[i](feat) for i, feat in enumerate(features)]
        mask_features = processed_features[-1]

        B, H, H_feat, W_feat = mask_features.shape
        mask_features_flat = mask_features.view(B, H, -1).transpose(1, 2)

        attended_features = self.attention(mask_embed, mask_features_flat, mask_features_flat)

        masks = []
        for q in range(num_queries):
            query_feat = attended_features[:, q:q + 1, :].transpose(1, 2)
            query_feat = query_feat.view(B, H, 1, 1)
            mask_feat = mask_features * query_feat
            mask = self.mask_predictor(mask_feat)
            masks.append(mask.squeeze(1))

        masks = torch.stack(masks, dim=1)
        return masks


# -------------------------
# Mask2Former Head
# -------------------------
class Mask2FormerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_queries = config.num_queries
        self.num_classes = config.num_classes

        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        decoder_layer = TransformerDecoderLayer(config.hidden_dim, config.num_heads, dropout=config.dropout)
        self.decoder = nn.ModuleList([decoder_layer for _ in range(config.num_decoder_layers)])
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes + 1)
        fpn_dims = [self.hidden_dim] * 4
        self.mask_head = MaskHead(self.hidden_dim, fpn_dims)
        self.bbox_embed = nn.Linear(self.hidden_dim, 4)

    def forward(self, features, pos_embed=None):
        memory = features[-1]
        B, C, H, W = memory.shape
        memory_flat = memory.flatten(2).transpose(1, 2)

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        output = query_embed
        for layer in self.decoder:
            output = layer(output, memory_flat)

        outputs_class = self.class_embed(output)
        outputs_coord = self.bbox_embed(output).sigmoid()
        outputs_mask = self.mask_head(features, output)

        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "pred_masks": outputs_mask
        }


# -------------------------
# Full DINOv3 + Mask2Former Model
# -------------------------
class DINOv3Mask2Former(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self._load_dinov3_backbone()
        backbone_dim = self._get_backbone_output_dim()
        self.input_proj = nn.Conv2d(backbone_dim, config.hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(config.hidden_dim // 2, normalize=True)
        self.head = Mask2FormerHead(config)

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _load_dinov3_backbone(self):
        backbone = torch.hub.load(
            self.config.dinov3_repo_dir,
            self.config.backbone_type,
            source="local",
            weights=self.config.backbone_weights_path
        )
        return backbone

    def _get_backbone_output_dim(self):
        backbone_dims = {
            "dinov3_vits16": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
            "dinov3_vith16plus": 1280,
            "dinov3_vit7b16": 3584,
            "dinov3_convnext_tiny": 768,
            "dinov3_convnext_small": 768,
            "dinov3_convnext_base": 1024,
            "dinov3_convnext_large": 1536,
        }
        return backbone_dims.get(self.config.backbone_type, 768)

    def forward(self, images):
        with torch.set_grad_enabled(not self.config.freeze_backbone):
            backbone_features = self.backbone.forward_features(images)

        B, num_patches, embed_dim = backbone_features["x_norm_patchtokens"].shape
        patch_size = int(math.sqrt(num_patches))
        features = backbone_features["x_norm_patchtokens"].transpose(1, 2).view(
            B, embed_dim, patch_size, patch_size
        )

        features = self.input_proj(features)
        pos_embed = self.position_embedding(features)
        multi_scale_features = [features]
        outputs = self.head(multi_scale_features, pos_embed)

        return outputs
