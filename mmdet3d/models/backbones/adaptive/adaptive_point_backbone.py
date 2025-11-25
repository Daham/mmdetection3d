"""
Backbone for processing adaptive octree voxels
Uses attention mechanism instead of sparse convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


class SizeAwareAttention(nn.Module):
    """
    Multi-head attention that considers voxel sizes
    
    Key innovation: Similar-sized voxels attend more to each other
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
    
    def forward(
        self,
        x: torch.Tensor,
        centers: torch.Tensor,
        sizes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) features
            centers: (N, 3) voxel centers
            sizes: (N, 1) voxel sizes
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Size-based modulation: similar-sized voxels attend more to each other
        size_diff = torch.abs(sizes - sizes.T)  # (N, N)
        size_weight = torch.exp(-size_diff / 0.1)  # Gaussian kernel
        attn = attn * size_weight.unsqueeze(0).unsqueeze(0)  # Broadcast to (B, num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return self.norm(x + out)


@MODELS.register_module()
class AdaptivePointBackbone(BaseModule):
    """
    Process adaptive voxels with attention mechanism
    
    Cannot use sparse convolutions (require fixed grid)
    Instead uses size-aware attention to process variable-sized voxels
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: List[int] = [256, 512, 512],
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        init_cfg: Optional[dict] = None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.num_layers = num_layers
        
        print(f"🔧 Initializing AdaptivePointBackbone:")
        print(f"   Layers: {num_layers}")
        print(f"   Feature channels: {feat_channels}")
        print(f"   Attention heads: {num_heads}")
        
        self.layers = nn.ModuleList()
        
        curr_channels = in_channels
        for i in range(num_layers):
            out_channels = feat_channels[min(i, len(feat_channels) - 1)]
            
            layer = nn.ModuleDict({
                'attn': SizeAwareAttention(curr_channels, num_heads),
                'ffn': nn.Sequential(
                    nn.Linear(curr_channels, out_channels * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(out_channels * 4, out_channels),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(curr_channels),
                'norm2': nn.LayerNorm(out_channels)
            })
            
            # Projection if channels change
            if curr_channels != out_channels:
                layer['proj'] = nn.Linear(curr_channels, out_channels)
            
            self.layers.append(layer)
            curr_channels = out_channels
        
        self.out_channels = curr_channels
    
    def forward(self, voxel_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process adaptive voxels
        
        Args:
            voxel_dict: From AdaptiveOctreeVFE with:
                - voxel_features: (M, C)
                - voxel_centers: (M, 3)
                - voxel_sizes: (M, 1)
        
        Returns:
            features: (M, C_out) processed features
        """
        x = voxel_dict['voxel_features'].unsqueeze(0)  # (1, M, C) add batch dim
        centers = voxel_dict['voxel_centers']
        sizes = voxel_dict['voxel_sizes']
        
        # Progressive feature extraction
        for layer in self.layers:
            # Size-aware attention
            x_norm = layer['norm1'](x)
            x = x + layer['attn'](x_norm, centers, sizes)
            
            # Feed-forward network with residual
            x_norm = x if 'proj' not in layer else layer['proj'](x)
            x_ffn = layer['ffn'](x_norm)
            x = layer['norm2'](x_norm + x_ffn)
        
        return x.squeeze(0)  # (M, C) remove batch dim
