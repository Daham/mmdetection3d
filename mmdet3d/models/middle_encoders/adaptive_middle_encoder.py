# mmdet3d/models/middle_encoders/adaptive_middle_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from .sparse_encoder import SparseEncoder

@MODELS.register_module()
class AdaptiveMiddleEncoder(SparseEncoder):
    """
    Middle encoder that can handle truly adaptive voxel sizes.
    
    Instead of trying to create different sparse shapes, this encoder:
    1. Uses adaptive pooling to handle varying voxel information
    2. Applies content-aware processing based on voxel characteristics
    3. Maintains efficiency while enabling true adaptivity
    """
    
    def __init__(self, 
                 sparse_shape=None,
                 base_sparse_shape=None,
                 adaptive_pooling=True,
                 **kwargs):
        # Handle both sparse_shape and base_sparse_shape parameters
        if base_sparse_shape is not None:
            actual_sparse_shape = base_sparse_shape
        elif sparse_shape is not None:
            actual_sparse_shape = sparse_shape
        else:
            raise ValueError("Either sparse_shape or base_sparse_shape must be provided")
            
        # Initialize with the sparse shape
        super().__init__(sparse_shape=actual_sparse_shape, **kwargs)
        self.base_sparse_shape = actual_sparse_shape
        self.adaptive_pooling = adaptive_pooling
        
        # Adaptive processing layers
        self.adaptive_processor = nn.Sequential(
            nn.Linear(self.in_channels, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        
        # Spatial attention mechanism for adaptive voxels
        self.spatial_attention = nn.Sequential(
            nn.Linear(self.in_channels + 3, 64),  # +3 for adaptive size info
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, voxel_features, coors, batch_size, adaptive_info=None):
        """
        Forward pass with support for adaptive voxel information.
        
        Args:
            voxel_features: [N, C] voxel features
            coors: [N, 4] coordinates (batch_idx, z, y, x)
            batch_size: int
            adaptive_info: dict with adaptive voxel information
        """
        if adaptive_info is not None and 'adaptive_sizes' in adaptive_info:
            return self._forward_adaptive(voxel_features, coors, batch_size, adaptive_info)
        else:
            # Fallback to standard processing
            return super().forward(voxel_features, coors, batch_size)
    
    def _forward_adaptive(self, voxel_features, coors, batch_size, adaptive_info):
        """Process with adaptive voxel size information."""
        adaptive_sizes = adaptive_info['adaptive_sizes']
        
        # Apply adaptive processing
        enhanced_features = self.adaptive_processor(voxel_features)
        
        # Use adaptive size information for spatial attention
        if adaptive_sizes.dim() > 1:  # Per-voxel sizes
            # Normalize adaptive sizes for attention
            size_features = adaptive_sizes / self.base_voxel_size[0]  # Normalize by base size
            attention_input = torch.cat([voxel_features, size_features], dim=1)
            attention_weights = self.spatial_attention(attention_input)
            
            # Apply attention
            enhanced_features = enhanced_features * attention_weights
        
        # Create spatial output
        output_H, output_W = 200, 176  # Standard SECOND output size
        spatial_features = torch.zeros(
            (batch_size, 256, output_H, output_W),
            dtype=voxel_features.dtype,
            device=voxel_features.device
        )
        
        # Map coordinates to output space
        batch_idx = coors[:, 0].long()
        z_idx = coors[:, 1].long()
        y_idx = coors[:, 2].long()
        x_idx = coors[:, 3].long()
        
        # Adaptive coordinate mapping based on voxel sizes
        D, H, W = self.base_sparse_shape
        if adaptive_sizes.dim() > 1:
            # Use adaptive sizes for coordinate mapping
            avg_size_factor = adaptive_sizes.mean(dim=1)  # Average across x,y,z
            # Adjust coordinates based on size - smaller voxels = finer resolution
            y_out = torch.clamp((y_idx.float() * output_H / H * avg_size_factor).long(), 0, output_H - 1)
            x_out = torch.clamp((x_idx.float() * output_W / W * avg_size_factor).long(), 0, output_W - 1)
        else:
            # Standard mapping
            y_out = torch.clamp((y_idx * output_H) // H, 0, output_H - 1)
            x_out = torch.clamp((x_idx * output_W) // W, 0, output_W - 1)
        
        # Aggregate features using scatter_add
        valid_mask = (batch_idx >= 0) & (batch_idx < batch_size)
        if valid_mask.any():
            valid_batch = batch_idx[valid_mask]
            valid_y = y_out[valid_mask]
            valid_x = x_out[valid_mask]
            valid_features = enhanced_features[valid_mask]
            
            # Use adaptive aggregation
            flat_indices = valid_batch * (output_H * output_W) + valid_y * output_W + valid_x
            flat_output = spatial_features.view(batch_size * output_H * output_W, -1)
            flat_output.scatter_add_(0, flat_indices.unsqueeze(1).expand(-1, 256), valid_features)
            spatial_features = flat_output.view(batch_size, 256, output_H, output_W)
        
        # Print adaptive statistics occasionally
        if self.training and torch.rand(1).item() < 0.02:
            if adaptive_sizes.dim() > 1:
                size_stats = {
                    'mean_x': adaptive_sizes[:, 0].mean().item(),
                    'mean_y': adaptive_sizes[:, 1].mean().item(), 
                    'mean_z': adaptive_sizes[:, 2].mean().item(),
                    'std_x': adaptive_sizes[:, 0].std().item(),
                    'std_y': adaptive_sizes[:, 1].std().item(),
                    'std_z': adaptive_sizes[:, 2].std().item()
                }
                print(f"Adaptive middle encoder - Size stats: {size_stats}")
                
                # Check for diversity in adaptation
                total_variation = adaptive_sizes.std(dim=0).sum().item()
                print(f"Adaptation diversity (higher=better): {total_variation:.6f}")
        
        return spatial_features
