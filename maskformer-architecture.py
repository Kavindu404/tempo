"""
MaskFormer: A mask-based segmentation model based on ContourFormer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from collections import OrderedDict

from ...core import register

@register()
class MaskFormer(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


@register()
class MaskDecoder(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size', 'mask_dim']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=100,
                 mask_dim=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=0,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 eval_spatial_size=None,
                 aux_loss=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.mask_dim = mask_dim
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        
        # Create transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, 
            nhead, 
            dim_feedforward, 
            dropout, 
            activation
        )
        self.decoder = TransformerDecoder(
            decoder_layer, 
            num_layers
        )
        
        # Create query embeddings that will be used as input to the decoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Class prediction head
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        
        # Mask prediction head
        self.mask_embed = MaskHead(hidden_dim, mask_dim)
        
        # Pixel-level decoder that converts mask embeddings to binary masks
        self.pixel_decoder = PixelDecoder(mask_dim, hidden_dim)
        
        # Denoising part for training
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim)
            
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize weights with standard methods
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize the class embeddings with bias favorable to predicting background
        nn.init.constant_(self.class_embed.bias, -4.0)
    
    def forward(self, features, targets=None):
        # Process multi-scale features from encoder
        memory, spatial_shapes = self._prepare_features(features)
        
        # Get query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            memory.shape[0], 1, 1
        )
        
        # Run the transformer decoder
        hs, mask_features = self.decoder(
            query_embed,  # queries
            memory,       # encoder outputs
            spatial_shapes,
            memory_key_padding_mask=None
        )
        
        # Predict class logits for each query
        outputs_class = self.class_embed(hs)
        
        # Predict mask embeddings for each query
        outputs_mask_embed = self.mask_embed(hs)
        
        # Convert mask embeddings to actual masks using the pixel decoder
        outputs_masks = self.pixel_decoder(mask_features, outputs_mask_embed, spatial_shapes)
        
        # Prepare outputs dictionary
        out = {
            'pred_logits': outputs_class[-1],
            'pred_masks': outputs_masks[-1],
        }
        
        # Add auxiliary outputs for deep supervision during training
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': outputs_class[i], 'pred_masks': outputs_masks[i]}
                for i in range(self.num_layers - 1)
            ]
            
        return out
    
    def _prepare_features(self, features):
        # Flatten all feature maps and prepare spatial shapes
        feat_flattens = []
        spatial_shapes = []
        
        for feat in features:
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat_flattens.append(feat.flatten(2).transpose(1, 2))  # [B, H*W, C]
            
        # Concatenate all flattened features
        memory = torch.cat(feat_flattens, dim=1)  # [B, sum(H*W), C]
        
        return memory, spatial_shapes


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = self._get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    
    def forward(self, query, memory, query_pos=None, memory_pos=None, attn_mask=None):
        # Self-attention
        q = k = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(q, k, value=query, attn_mask=attn_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        
        # Cross-attention
        q = self.with_pos_embed(query, query_pos)
        k = self.with_pos_embed(memory, memory_pos)
        query2 = self.cross_attn(q, k, value=memory)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        
        # FFN
        query2 = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = query + self.dropout4(query2)
        query = self.norm3(query)
        
        return query


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, query, memory, spatial_shapes, memory_key_padding_mask=None):
        output = query
        
        # Initialize list to store intermediate outputs for deep supervision
        intermediate = []
        intermediate_mask_features = []
        
        # Process through each decoder layer
        for layer in self.layers:
            output = layer(
                output,
                memory,
                query_pos=None,
                memory_pos=None,
                attn_mask=None
            )
            
            intermediate.append(output)
            # Here we would also compute intermediate mask features
            mask_feature = self._compute_mask_feature(output, memory, spatial_shapes)
            intermediate_mask_features.append(mask_feature)
        
        return torch.stack(intermediate), torch.stack(intermediate_mask_features)
    
    def _compute_mask_feature(self, query, memory, spatial_shapes):
        # This method would generate mask features from the current query output
        # In a real implementation, this would involve more sophisticated processing
        # to create high-quality mask embeddings
        
        # Simplified implementation - just a placeholder
        return query


class MaskHead(nn.Module):
    def __init__(self, hidden_dim, mask_dim):
        super().__init__()
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mask_dim)
        )
    
    def forward(self, x):
        # x: [num_layers, batch_size, num_queries, hidden_dim]
        return self.mask_head(x)


class PixelDecoder(nn.Module):
    def __init__(self, mask_dim, hidden_dim):
        super().__init__()
        # This module takes mask embeddings and produces binary masks
        self.conv_layers = nn.Sequential(
            nn.Conv2d(mask_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1)
        )
    
    def forward(self, mask_features, mask_embeddings, spatial_shapes):
        # mask_features: features suitable for mask prediction
        # mask_embeddings: per-object mask embeddings from the decoder
        
        # Number of layers, batch size, number of queries, mask dimension
        num_layers, bs, num_queries, mask_dim = mask_embeddings.shape
        
        # Initialize list to store masks for all layers
        all_masks = []
        
        # Process each layer
        for layer_idx in range(num_layers):
            # Get current layer's mask embeddings
            curr_mask_embeddings = mask_embeddings[layer_idx]  # [bs, num_queries, mask_dim]
            
            # Reshape mask features to spatial dimensions
            # This is a simplified implementation - a real version would process per feature level
            h, w = spatial_shapes[0]  # Using first level's spatial shape for simplicity
            
            # Process mask embeddings to create binary masks
            # In a real implementation, this would use mask_features more effectively
            masks = self._generate_masks(curr_mask_embeddings, mask_features[layer_idx], h, w)
            all_masks.append(masks)
            
        # Stack masks from all layers
        return torch.stack(all_masks)
    
    def _generate_masks(self, mask_embeddings, mask_features, height, width):
        # Generate binary masks from mask embeddings
        # This is a simplified implementation
        
        bs, num_queries, mask_dim = mask_embeddings.shape
        
        # Reshape mask embeddings for processing
        mask_embeddings = mask_embeddings.view(bs * num_queries, mask_dim, 1, 1)
        
        # Apply convolutions to generate masks
        masks = self.conv_layers(mask_embeddings)
        
        # Resize to target spatial dimensions
        masks = F.interpolate(masks, size=(height, width), mode='bilinear', align_corners=False)
        
        # Reshape to [bs, num_queries, h, w]
        masks = masks.view(bs, num_queries, height, width)
        
        return masks
