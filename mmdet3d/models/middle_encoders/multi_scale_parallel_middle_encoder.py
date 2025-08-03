# 🎯 MULTI-SCALE PARALLEL MIDDLE ENCODER
# PhD Research: Process different voxel size tensors in parallel networks, then fuse

from typing import Tuple, Dict, List
import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS
from .sparse_encoder import SparseEncoder


@MODELS.register_module()
class MultiScaleParallelMiddleEncoder(nn.Module):
    """
    🎯 PhD Research Architecture: True Parallel Processing
    
    Instead of early fusion (concatenating then processing):
    1. Process each voxel scale with separate sparse conv networks
    2. Intelligently fuse the processed features
    3. Output consistent BEV feature maps for detection head
    """
    
    def __init__(self,
                 sparse_shape=[41, 1600, 1408],
                 in_channels=4,
                 order=('conv', 'norm', 'act'),  # Must be tuple
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
                 encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [0, 1, 1]], [0, 0]],
                 block_type='basicblock',
                 **kwargs):
        super().__init__()
        
        self.sparse_shape = sparse_shape
        
        # 🔬 PARALLEL SPARSE ENCODERS: One for each voxel scale
        self.fine_encoder = SparseEncoder(
            sparse_shape=sparse_shape,
            in_channels=in_channels,
            order=order,
            norm_cfg=norm_cfg,
            base_channels=base_channels,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            block_type=block_type,
            **kwargs
        )
        
        self.medium_encoder = SparseEncoder(
            sparse_shape=sparse_shape,
            in_channels=in_channels,
            order=order,
            norm_cfg=norm_cfg,
            base_channels=base_channels,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            block_type=block_type,
            **kwargs
        )
        
        self.coarse_encoder = SparseEncoder(
            sparse_shape=sparse_shape,
            in_channels=in_channels,
            order=order,
            norm_cfg=norm_cfg,
            base_channels=base_channels,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            block_type=block_type,
            **kwargs
        )
        
        # 🧠 INTELLIGENT FUSION MODULE  
        # After parallel processing, fuse features intelligently
        # Each sparse encoder outputs 256 channels, so 3 × 256 = 768 total channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1),  # 768 -> 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),  # 512 -> 256 (expected by backbone)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, voxel_features: Tensor, coors: Tensor, batch_size: int) -> Tensor:
        """
        🎯 Revolutionary Parallel Processing Pipeline
        
        Args:
            voxel_features: Combined features from multi-scale encoder
            coors: Combined coordinates with scale info in last column
            batch_size: Batch size
            
        Returns:
            Fused BEV feature map with consistent dimensions
        """
        # 🔍 SEPARATE TENSORS BY SCALE
        # Extract scale info from the last column of coordinates
        scale_ids = coors[:, -1]  # Last column contains scale ID (0=fine, 1=medium, 2=coarse)
        base_coors = coors[:, :-1]  # Remove scale column for sparse conv
        
        # Split features and coordinates by scale
        fine_mask = scale_ids == 0
        medium_mask = scale_ids == 1  
        coarse_mask = scale_ids == 2
        
        fine_features = voxel_features[fine_mask]
        medium_features = voxel_features[medium_mask]
        coarse_features = voxel_features[coarse_mask]
        
        fine_coors = base_coors[fine_mask]
        medium_coors = base_coors[medium_mask]
        coarse_coors = base_coors[coarse_mask]
        
        # 🚀 PARALLEL SPARSE CONVOLUTION PROCESSING
        processed_features = []
        
        # Process fine scale (if has voxels)
        if fine_features.shape[0] > 0:
            fine_bev = self.fine_encoder(fine_features, fine_coors, batch_size)
            processed_features.append(fine_bev)
        else:
            # Create zero tensor with 256 channels (matching sparse encoder output)
            zero_bev = torch.zeros(batch_size, 256, 200, 176, 
                                 device=voxel_features.device, dtype=voxel_features.dtype)
            processed_features.append(zero_bev)
            
        # Process medium scale (if has voxels)
        if medium_features.shape[0] > 0:
            medium_bev = self.medium_encoder(medium_features, medium_coors, batch_size)
            processed_features.append(medium_bev)
        else:
            zero_bev = torch.zeros(batch_size, 256, 200, 176,
                                 device=voxel_features.device, dtype=voxel_features.dtype)
            processed_features.append(zero_bev)
            
        # Process coarse scale (if has voxels)
        if coarse_features.shape[0] > 0:
            coarse_bev = self.coarse_encoder(coarse_features, coarse_coors, batch_size)
            processed_features.append(coarse_bev)
        else:
            # Create zero tensor with SAME CHANNELS as other scales (256, not 128)
            zero_bev = torch.zeros(batch_size, 256, 200, 176,
                                 device=voxel_features.device, dtype=voxel_features.dtype)
            processed_features.append(zero_bev)
        
        # 🧠 INTELLIGENT LATE FUSION
        # Concatenate along channel dimension for fusion - now guaranteed 3×256=768 channels
        concatenated_bev = torch.cat(processed_features, dim=1)  # [B, 768, 200, 176]
        
        # Apply fusion convolutions to reduce to standard output channels
        fused_bev = self.fusion_conv(concatenated_bev)  # [B, 256, 200, 176]
        
        return fused_bev
