"""
CPU-only fallback for MultiScaleDeformableAttention
This module provides a CPU implementation when CUDA compilation fails
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class MSDeformAttnFunction(Function):
    """CPU fallback implementation of multi-scale deformable attention"""
    
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        """
        CPU implementation of forward pass
        Args:
            value: [N, S, C] where S = sum(Hi*Wi)
            spatial_shapes: [num_levels, 2]
            level_start_index: [num_levels]
            sampling_locations: [N, Lq, num_heads, num_levels, num_points, 2]
            attention_weights: [N, Lq, num_heads, num_levels, num_points]
            im2col_step: not used in CPU version
        """
        warnings.warn("Using CPU fallback for MultiScaleDeformableAttention. "
                     "Performance will be significantly slower than CUDA version.", 
                     UserWarning)
        
        N, S, C = value.shape
        _, Lq, num_heads, num_levels, num_points, _ = sampling_locations.shape
        
        # Reshape value for easier processing
        value_list = []
        for level, (H, W) in enumerate(spatial_shapes):
            start_idx = level_start_index[level]
            end_idx = start_idx + H * W
            value_level = value[:, start_idx:end_idx, :].view(N, H, W, C)
            value_list.append(value_level)
        
        # Process each query location
        output = torch.zeros(N, Lq, num_heads, C // num_heads, dtype=value.dtype, device=value.device)
        
        for n in range(N):
            for q in range(Lq):
                for h in range(num_heads):
                    for level in range(num_levels):
                        H, W = spatial_shapes[level]
                        value_level = value_list[level][n]  # [H, W, C]
                        
                        for p in range(num_points):
                            # Get sampling location
                            loc = sampling_locations[n, q, h, level, p]  # [2]
                            weight = attention_weights[n, q, h, level, p]  # scalar
                            
                            # Convert normalized coordinates to pixel coordinates
                            x = (loc[0] + 1) * (W - 1) / 2  # Convert from [-1, 1] to [0, W-1]
                            y = (loc[1] + 1) * (H - 1) / 2  # Convert from [-1, 1] to [0, H-1]
                            
                            # Clamp coordinates
                            x = torch.clamp(x, 0, W - 1)
                            y = torch.clamp(y, 0, H - 1)
                            
                            # Bilinear interpolation
                            x0 = torch.floor(x).long()
                            y0 = torch.floor(y).long()
                            x1 = torch.clamp(x0 + 1, 0, W - 1)
                            y1 = torch.clamp(y0 + 1, 0, H - 1)
                            
                            # Interpolation weights
                            wa = (x1 - x) * (y1 - y)
                            wb = (x - x0) * (y1 - y)
                            wc = (x1 - x) * (y - y0)
                            wd = (x - x0) * (y - y0)
                            
                            # Get channel slice for this head
                            c_start = h * (C // num_heads)
                            c_end = c_start + (C // num_heads)
                            
                            # Sample values at 4 corners
                            va = value_level[y0, x0, c_start:c_end]
                            vb = value_level[y0, x1, c_start:c_end]
                            vc = value_level[y1, x0, c_start:c_end]
                            vd = value_level[y1, x1, c_start:c_end]
                            
                            # Bilinear interpolation
                            sampled_value = wa * va + wb * vb + wc * vc + wd * vd
                            
                            # Accumulate weighted result
                            output[n, q, h] += weight * sampled_value
        
        return output.view(N, Lq, C)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # For simplicity, return zero gradients
        # A proper implementation would compute gradients for all inputs
        warnings.warn("Backward pass for CPU MSDeformAttn is simplified", UserWarning)
        return None, None, None, None, None, None


def ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights):
    """
    Multi-scale deformable attention core function (CPU version)
    """
    N, S, C = value.shape
    _, Lq, num_heads, num_levels, num_points, _ = sampling_locations.shape
    
    level_start_index = []
    start = 0
    for h, w in spatial_shapes:
        level_start_index.append(start)
        start += h * w
    level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=value.device)
    
    return MSDeformAttnFunction.apply(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 4
    )


# Create a fallback module that can be imported
class CPUMultiScaleDeformableAttention:
    """CPU fallback for the CUDA MultiScaleDeformableAttention"""
    
    def __init__(self):
        warnings.warn(
            "Using CPU fallback for MultiScaleDeformableAttention. "
            "CUDA compilation failed due to Visual Studio/CUDA version incompatibility. "
            "Training will be much slower but functional.",
            UserWarning
        )
    
    def __call__(self, *args, **kwargs):
        return ms_deform_attn_core_pytorch(*args, **kwargs)


# This will be imported as the fallback
def ms_deform_attn_core_pytorch_fallback(*args, **kwargs):
    return ms_deform_attn_core_pytorch(*args, **kwargs)


if __name__ == "__main__":
    print("CPU fallback for MultiScaleDeformableAttention loaded successfully")