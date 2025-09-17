import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module"""
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads')
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # Compute sampling locations
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + \
                           sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        # Sample features
        output = self._sample_features(
            value, sampling_locations, attention_weights,
            input_spatial_shapes, input_padding_mask
        )
        
        output = self.output_proj(output)
        
        return output
    
    def _sample_features(self, value, sampling_locations, attention_weights, 
                         input_spatial_shapes, input_padding_mask):
        N, Len_q, n_heads, n_levels, n_points = sampling_locations.shape[:5]
        
        # Split by levels
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        
        for level, (H_, W_) in enumerate(input_spatial_shapes):
            value_l_ = value_list[level].view(N, H_, W_, self.n_heads, -1)
            value_l_ = value_l_.permute(0, 3, 4, 1, 2).contiguous()  # (N, n_heads, C, H, W)
            value_l_ = value_l_.flatten(0, 1)  # (N*n_heads, C, H, W)
            
            sampling_grid_l_ = sampling_grids[:, :, :, level]  # (N, Len_q, n_heads, n_points, 2)
            sampling_grid_l_ = sampling_grid_l_.transpose(1, 2)  # (N, n_heads, Len_q, n_points, 2)
            sampling_grid_l_ = sampling_grid_l_.flatten(0, 1)  # (N*n_heads, Len_q, n_points, 2)
            
            # Bilinear sampling
            sampling_value_l_ = F.grid_sample(
                value_l_, sampling_grid_l_,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )  # (N*n_heads, C, Len_q, n_points)
            
            sampling_value_list.append(sampling_value_l_)
        
        # Weighted sum
        attention_weights = attention_weights.transpose(1, 2).reshape(N * self.n_heads, Len_q, n_levels, n_points)
        output = torch.zeros(N * self.n_heads, self.d_model // self.n_heads, Len_q, device=value.device)
        
        for level, sampling_value in enumerate(sampling_value_list):
            # (N*n_heads, C, Len_q, n_points) * (N*n_heads, Len_q, 1, n_points) -> (N*n_heads, C, Len_q)
            level_output = (sampling_value * attention_weights[:, :, level, :].unsqueeze(1)).sum(-1)
            output += level_output
        
        output = output.view(N, self.n_heads, -1, Len_q).transpose(2, 3).contiguous()
        output = output.view(N, Len_q, -1)
        
        return output


def get_reference_points(spatial_shapes, device):
    """Get reference points for each feature level"""
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if isinstance(H_, torch.Tensor):
            H_ = H_.item()
            W_ = W_.item()
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, int(H_), dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, int(W_), dtype=torch.float32, device=device),
            indexing='ij'  # Explicit indexing to avoid warning
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    
    return reference_points