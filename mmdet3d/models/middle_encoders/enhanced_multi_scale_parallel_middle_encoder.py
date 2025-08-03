"""
Enhanced Multi-Scale Parallel Middle Encoder for Advanced VFE
============================================================

Designed to work with MultiScaleVFEWithAttention module.
Handles multi-scale features with attention weights and scale embeddings.

Author: PhD Research Implementation
Date: August 3, 2025
"""

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from typing import Tuple


@MODELS.register_module()
class EnhancedMultiScaleParallelMiddleEncoder(BaseModule):
    """
    Enhanced middle encoder for multi-scale VFE output with attention
    
    Handles input features from MultiScaleVFEWithAttention:
    - Multi-scale features with scale embeddings
    - Attention-weighted features
    - Scale ID information
    """
    
    def __init__(self, 
                 in_channels: int = 81,  # 64 (features) + 16 (scale_emb) + 1 (scale_id)
                 output_channels: int = 256, 
                 sparse_shape: Tuple[int, int, int] = (41, 1600, 1408),
                 order=('conv', 'norm', 'act'),  # Compatible with base config
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.sparse_shape = sparse_shape
        self.order = order  # Store order parameter for compatibility
        
        # BEV dimensions
        self.bev_h, self.bev_w = sparse_shape[1] // 4, sparse_shape[2] // 4
        
        # Project from multi-scale features to standard feature size
        self.feature_projection = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        # Multi-scale aware processing
        self.scale_aware_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def _convert_to_bev(self, voxel_features, coors, batch_size):
        """Convert voxel features to BEV representation"""
        device = voxel_features.device
        dtype = voxel_features.dtype
        
        # Initialize BEV feature map
        bev_map = torch.zeros(
            (batch_size, 64, self.bev_h, self.bev_w),
            dtype=dtype, device=device
        )
        
        # Handle coordinate format
        if coors.shape[1] == 3:
            batch_indices = torch.zeros(coors.shape[0], dtype=torch.long, device=device)
            full_coors = torch.cat([batch_indices.unsqueeze(1), coors], dim=1)
        else:
            full_coors = coors
        
        # Extract coordinates
        batch_idx = full_coors[:, 0].long()
        z_idx = full_coors[:, 1].long()
        y_idx = full_coors[:, 2].long()
        x_idx = full_coors[:, 3].long()
        
        # Convert to BEV coordinates
        bev_y = y_idx // 4
        bev_x = x_idx // 4
        
        # Create valid mask
        valid_mask = (
            (batch_idx >= 0) & (batch_idx < batch_size) &
            (bev_y >= 0) & (bev_y < self.bev_h) &
            (bev_x >= 0) & (bev_x < self.bev_w)
        )
        
        if valid_mask.sum() > 0:
            # Filter valid coordinates and features
            valid_batch = batch_idx[valid_mask]
            valid_y = bev_y[valid_mask]
            valid_x = bev_x[valid_mask]
            valid_features = voxel_features[valid_mask]
            
            # Assign features to BEV map
            for i in range(batch_size):
                batch_mask = valid_batch == i
                if batch_mask.sum() > 0:
                    b_y = valid_y[batch_mask]
                    b_x = valid_x[batch_mask]
                    b_features = valid_features[batch_mask]
                    
                    # Use scatter_add for accumulation
                    indices = b_y * self.bev_w + b_x
                    flat_bev = bev_map[i].view(64, -1)
                    b_features_t = b_features.transpose(0, 1)
                    flat_bev.scatter_add_(1, indices.unsqueeze(0).expand(64, -1), b_features_t)
        
        return bev_map
    
    def forward(self, voxel_features, coors, batch_size):
        """
        Forward pass for enhanced multi-scale processing
        
        Args:
            voxel_features: Multi-scale features [N, in_channels]
            coors: Voxel coordinates [N, 3] or [N, 4]
            batch_size: Batch size
            
        Returns:
            BEV features [B, output_channels, H, W]
        """
        # Project multi-scale features to standard size
        projected_features = self.feature_projection(voxel_features)  # [N, 64]
        
        # Convert to BEV representation
        bev_features = self._convert_to_bev(projected_features, coors, batch_size)
        
        # Apply scale-aware convolutions
        output = self.scale_aware_conv(bev_features)
        
        return output
