import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

class DINOv3Mask2Former(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load DINOv3 backbone
        self.backbone = self._load_dinov3_backbone()
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_features = self.backbone(dummy_input)
            if isinstance(backbone_features, dict):
                backbone_dim = backbone_features['x_norm_patchtokens'].shape[-1]
            else:
                backbone_dim = backbone_features.shape[-1]
        
        # Projection layers
        self.input_proj = nn.Conv2d(backbone_dim, config.hidden_dim, kernel_size=1)
        self.mask_embed = MLP(config.hidden_dim, config.hidden_dim, config.mask_dim, 3)
        
        # Transformer decoder
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_dim)
        self.transformer_decoder = TransformerDecoder(
            config.hidden_dim,
            config.nheads,
            config.dim_feedforward,
            config.dec_layers
        )
        
        # Output heads
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)
        self.bbox_embed = MLP(config.hidden_dim, config.hidden_dim, 4, 3)
        
    def _load_dinov3_backbone(self):
        backbone_name = f'dinov3_{self.config.backbone_type}'
        weights_path = self.config.get_backbone_weight_path()
        
        backbone = torch.hub.load(
            self.config.dinov3_repo_dir,
            backbone_name,
            source='local',
            weights=weights_path
        )
        return backbone
    
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Extract features from backbone
        features = self.backbone(images)
        
        if isinstance(features, dict):
            features = features['x_norm_patchtokens']
        
        # Reshape features to spatial format
        h = w = int(math.sqrt(features.shape[1]))
        features = features.transpose(1, 2).reshape(batch_size, -1, h, w)
        
        # Project features
        features = self.input_proj(features)
        
        # Flatten for transformer
        features_flat = features.flatten(2).permute(2, 0, 1)
        
        # Generate queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        # Transformer decoder
        hs, memory = self.transformer_decoder(query_embed, features_flat)
        
        # Generate outputs
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        # Generate masks
        mask_embed = self.mask_embed(hs)
        outputs_mask = torch.einsum("lbq,bqc->lbc", memory, mask_embed)
        outputs_mask = outputs_mask.reshape(batch_size, self.config.num_queries, h, w)
        
        # Upsample masks
        outputs_mask = F.interpolate(
            outputs_mask,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_masks': outputs_mask,
            'aux_outputs': [
                {'pred_logits': a, 'pred_boxes': b, 'pred_masks': outputs_mask}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        }
        
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory):
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory)
            intermediate.append(self.norm(output))
        
        return torch.stack(intermediate), memory

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=0.1)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=0.1)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
        self.activation = nn.ReLU()
    
    def forward(self, tgt, memory):
        tgt2 = self.norm1(tgt)
        q = k = tgt2
        tgt2 = self.self_attn(q, k, tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt2, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
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