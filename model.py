import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from deformable_attention import MSDeformAttn, get_reference_points

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
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = self._extract_features(dummy)
                self.backbone_dim = out.shape[-1]
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Pixel Decoder (FPN-like structure for multi-scale features)
        self.pixel_decoder = PixelDecoder(
            backbone_dim=self.backbone_dim,
            hidden_dim=config.hidden_dim,
            num_feature_levels=3
        )
        
        # Positional encoding
        self.pe_layer = PositionEmbeddingSine(config.hidden_dim // 2, normalize=True)
        
        # Query initialization with learnable content and position
        self.query_feat = nn.Embedding(config.num_queries, config.hidden_dim)
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_dim)
        
        # Query diversity enhancement
        self.query_diversity = QueryDiversityModule(config.hidden_dim, config.num_queries)
        
        # Transformer decoder with deformable attention
        self.decoder = Mask2FormerDecoder(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            num_layers=config.dec_layers,
            num_feature_levels=3
        )
        
        # Mask prediction heads (one per decoder layer)
        self.mask_embed = nn.ModuleList([
            MLP(config.hidden_dim, config.hidden_dim, config.mask_dim, 3)
            for _ in range(config.dec_layers)
        ])
        
        # Classification and box heads
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)
        self.bbox_embed = MLP(config.hidden_dim, config.hidden_dim, 4, 3)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        
        # Class embedding initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.config.num_classes + 1) * bias_value
        
        # Bbox head initialization
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        # Initialize query embeddings with normal distribution
        torch.nn.init.normal_(self.query_feat.weight, std=0.01)
        torch.nn.init.normal_(self.query_embed.weight, std=0.01)
    
    def _load_dinov3_backbone(self):
        backbone_name = f'dinov3_{self.config.backbone_type}'
        weights_path = self.config.get_backbone_weight_path()
        
        backbone = torch.hub.load(
            self.config.dinov3_repo_dir,
            backbone_name,
            source='local',
            weights=weights_path
        )
        
        if self.config.freeze_backbone:
            backbone.eval()
        
        return backbone
    
    def _extract_features(self, x):
        """Extract features from DINOv3 backbone"""
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
            
            if isinstance(features, dict):
                if 'x_norm_patchtokens' in features:
                    return features['x_norm_patchtokens']
                elif 'x' in features:
                    return features['x']
            else:
                if hasattr(self.backbone, 'global_pool'):
                    if self.backbone.global_pool == 'token':
                        return features[:, 1:]
                return features
        else:
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
        
        # Reshape to spatial format
        if features.dim() == 3:
            features = features.transpose(1, 2).reshape(batch_size, self.backbone_dim, h, w)
        elif features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        # Multi-scale feature extraction via pixel decoder
        multi_scale_features, multi_scale_masks, multi_scale_pos = self.pixel_decoder(features)
        
        # Prepare for transformer
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_scale_features, multi_scale_masks, multi_scale_pos)):
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            
            src_flatten.append(feat)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(pos_embed)
        
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        
        # Enhance query diversity
        query_embeds = self.query_diversity(self.query_embed.weight, self.query_feat.weight, spatial_shapes)
        query_feat = query_embeds['content'].unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = query_embeds['position'].unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Reference points for deformable attention
        reference_points = get_reference_points(spatial_shapes, device=src_flatten.device)
        
        # Decoder forward pass
        hs, memory, inter_references = self.decoder(
            query=query_feat,
            key=src_flatten,
            value=src_flatten,
            query_pos=query_pos,
            key_pos=lvl_pos_embed_flatten,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes
        )
        
        # Generate outputs for each decoder layer
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        
        # Compute mask features from pixel decoder output
        mask_features = self.pixel_decoder.get_mask_features(multi_scale_features)
        
        for lvl in range(hs.shape[0]):
            hs_lvl = hs[lvl]
            
            # Classification
            outputs_class = self.class_embed(hs_lvl)
            outputs_classes.append(outputs_class)
            
            # Bbox regression
            outputs_coord = self.bbox_embed(hs_lvl).sigmoid()
            outputs_coords.append(outputs_coord)
            
            # Mask prediction via dynamic convolution
            mask_embed = self.mask_embed[lvl](hs_lvl)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            
            # Upsample to original resolution
            outputs_mask = F.interpolate(
                outputs_mask,
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False
            )
            outputs_masks.append(outputs_mask)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_masks = torch.stack(outputs_masks)
        
        out = {
            'pred_logits': outputs_classes[-1],
            'pred_boxes': outputs_coords[-1],
            'pred_masks': outputs_masks[-1],
            'aux_outputs': [
                {'pred_logits': c, 'pred_boxes': b, 'pred_masks': m}
                for c, b, m in zip(outputs_classes[:-1], outputs_coords[:-1], outputs_masks[:-1])
            ]
        }
        
        return out


class PixelDecoder(nn.Module):
    """FPN-like pixel decoder for multi-scale feature processing"""
    def __init__(self, backbone_dim, hidden_dim, num_feature_levels=3):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        
        # Input projection from backbone
        self.input_proj = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        
        # Lateral connections (1x1 convs)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in range(num_feature_levels)
        ])
        
        # Output convolutions (3x3 convs)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_feature_levels)
        ])
        
        # Mask feature projection
        self.mask_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
        # Position encoders for each level
        self.pe_layers = nn.ModuleList([
            PositionEmbeddingSine(hidden_dim // 2, normalize=True)
            for _ in range(num_feature_levels)
        ])
        
        self._mask_features = None
    
    def forward(self, x):
        # Project backbone features
        x = self.input_proj(x)
        
        # Generate multi-scale features via strided convolutions
        features = [x]
        for i in range(1, self.num_feature_levels):
            features.append(F.max_pool2d(features[-1], kernel_size=2, stride=2))
        
        # Apply FPN-style processing
        outputs = []
        masks = []
        pos_embeds = []
        
        # Top-down path
        for i in range(self.num_feature_levels - 1, -1, -1):
            feat = self.lateral_convs[i](features[i])
            
            if i < self.num_feature_levels - 1:
                # Upsample and add
                feat_up = F.interpolate(outputs[-1], size=feat.shape[-2:], mode='bilinear', align_corners=False)
                feat = feat + feat_up
            
            # Apply output convolution
            feat = self.output_convs[i](feat)
            outputs.insert(0, feat)
            
            # Generate mask for padding
            mask = torch.zeros((feat.shape[0], feat.shape[2], feat.shape[3]), 
                              dtype=torch.bool, device=feat.device)
            masks.insert(0, mask)
            
            # Generate position embedding
            pos_embed = self.pe_layers[i](feat, mask)
            pos_embeds.insert(0, pos_embed)
        
        # Store mask features (highest resolution)
        self._mask_features = self.mask_proj(outputs[0])
        
        return outputs, masks, pos_embeds
    
    def get_mask_features(self, multi_scale_features=None):
        return self._mask_features


class QueryDiversityModule(nn.Module):
    """Enhance query diversity to prevent mode collapse"""
    def __init__(self, hidden_dim, num_queries):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Spatial prior for queries
        self.spatial_prior = nn.Linear(2, hidden_dim)
        
        # Content diversity loss
        self.content_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query_pos, query_content, spatial_shapes):
        # Generate spatial priors for queries
        # Handle non-square number of queries
        grid_size = int(math.ceil(math.sqrt(self.num_queries)))
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, grid_size),
            torch.linspace(0, 1, grid_size),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        grid = grid[:self.num_queries]  # Take only the needed number
        grid = grid.to(query_pos.device)
        
        # Add spatial bias to queries
        spatial_bias = self.spatial_prior(grid)
        
        # Enhance content diversity
        query_content = self.content_proj(query_content)
        
        # Combine with spatial prior
        enhanced_pos = query_pos + spatial_bias
        
        return {
            'position': enhanced_pos,
            'content': query_content
        }


class Mask2FormerDecoder(nn.Module):
    """Transformer decoder with multi-scale deformable attention"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers, num_feature_levels):
        super().__init__()
        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        
        # Decoder layers
        self.layers = nn.ModuleList([
            Mask2FormerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, num_feature_levels
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Reference point head
        self.reference_point_head = MLP(d_model, d_model, 2, 2)
    
    def forward(self, query, key, value, query_pos, key_pos, key_padding_mask,
                reference_points, spatial_shapes):
        output = query
        intermediate = []
        intermediate_ref_points = []
        
        bs = query.shape[0]
        
        # Initial reference points for queries
        query_ref_points = self.reference_point_head(query_pos).sigmoid()
        query_ref_points = query_ref_points.unsqueeze(2).repeat(1, 1, self.num_feature_levels, 1)
        
        for layer_idx, layer in enumerate(self.layers):
            output = layer(
                output, key, value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_ref_points=query_ref_points,
                key_ref_points=reference_points,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes
            )
            
            # Update reference points
            new_query_ref_points = self.reference_point_head(output).sigmoid()
            query_ref_points = new_query_ref_points.unsqueeze(2).repeat(1, 1, self.num_feature_levels, 1)
            
            intermediate.append(self.norm(output))
            intermediate_ref_points.append(query_ref_points)
        
        return torch.stack(intermediate), key, torch.stack(intermediate_ref_points)


class Mask2FormerDecoderLayer(nn.Module):
    """Decoder layer with self-attention and multi-scale deformable cross-attention"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_feature_levels):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multi-scale deformable cross attention
        self.cross_attn = MSDeformAttn(d_model, num_feature_levels, nhead, n_points=4)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.activation = nn.ReLU(inplace=True)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory, value, query_pos, key_pos, 
                query_ref_points, key_ref_points, key_padding_mask, spatial_shapes):
        
        # Self attention with pre-norm
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, _ = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), 
                                  tgt2.transpose(0, 1))
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        
        # Multi-scale deformable cross attention with pre-norm
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(
            tgt2, query_ref_points, 
            memory, spatial_shapes, key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        
        # FFN with pre-norm
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
