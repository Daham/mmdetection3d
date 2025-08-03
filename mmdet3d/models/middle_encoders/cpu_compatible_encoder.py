"""
CPU-Compatible Dense Encoder for Adaptive Voxelization
====================================================

This module implements a CPU-compatible dense encoder that replaces SparseEncoder
to avoid CUDA sparse convolution issues while testing adaptive voxelization.

Author: PhD Research Implementation  
Date: August 3, 2025
"""

import torch
import torch.nn as nn
from typing import Tuple
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType
from mmengine.model import BaseModule


@MODELS.register_module()
@MODELS.register_module(name='CPUSparseEncoder')  # Register with alias for config compatibility
class CPUCompatibleDenseEncoder(BaseModule):
    """
    CPU-compatible dense encoder that processes voxel features without sparse convolutions.
    
    This encoder is designed to work around CUDA sparse convolution issues while
    preserving the same interface as SparseEncoder for seamless integration.
    """
    
    def __init__(self,
                 in_channels: int = 65,
                 output_channels: int = 128,
                 sparse_shape: Tuple[int, int, int] = (41, 1600, 1408),
                 order: Tuple[str, ...] = ('conv', 'norm', 'act'),
                 encoder_channels: Tuple = ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                 encoder_paddings: Tuple = ((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                 block_type: str = 'basicblock',
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.sparse_shape = sparse_shape
        self.order = order
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.block_type = block_type
        
        # Build dense convolution layers to process voxel features
        self.conv_layers = nn.ModuleList()
        
        # Input projection layer
        self.input_conv = nn.Sequential(
            nn.Linear(in_channels, encoder_channels[0][0]),
            nn.BatchNorm1d(encoder_channels[0][0]),
            nn.ReLU(inplace=True)
        )
        
        # Progressive feature encoding layers
        prev_channels = encoder_channels[0][0]
        for layer_channels in encoder_channels:
            layer_convs = nn.ModuleList()
            for out_channels in layer_channels:
                layer_convs.append(
                    nn.Sequential(
                        nn.Linear(prev_channels, out_channels),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1)
                    )
                )
                prev_channels = out_channels
            self.conv_layers.append(layer_convs)
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Linear(prev_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial feature processing (global pooling + expansion)
        self.spatial_processor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(output_channels, output_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels * 4, output_channels)
        )
        
    def forward(self, voxel_features: torch.Tensor, 
                coordinates: torch.Tensor, 
                batch_size: int) -> dict:
        """
        Forward pass processing voxel features in dense format.
        
        Args:
            voxel_features: (N, in_channels) - voxel features from VFE
            coordinates: (N, 4) - voxel coordinates [batch_idx, z, y, x]
            batch_size: batch size
            
        Returns:
            dict: Processed features in format compatible with SECOND backbone
        """
        device = voxel_features.device
        N, C = voxel_features.shape
        
        if N == 0:
            # Handle empty input
            dummy_features = torch.zeros(batch_size, self.output_channels, 
                                       200, 176, device=device, dtype=voxel_features.dtype)
            return dummy_features
        
        # === STEP 1: PROGRESSIVE FEATURE ENCODING ===
        x = self.input_conv(voxel_features)  # (N, initial_channels)
        
        # Apply encoder layers progressively
        for layer_convs in self.conv_layers:
            for conv in layer_convs:
                x = conv(x)  # Progressive feature transformation
        
        # Final output projection
        encoded_features = self.output_conv(x)  # (N, output_channels)
        
        # === STEP 2: SPATIAL AGGREGATION ===
        # Group features by batch for spatial processing
        batch_features = []
        for b in range(batch_size):
            batch_mask = coordinates[:, 0] == b
            if batch_mask.any():
                batch_voxel_features = encoded_features[batch_mask]  # (N_b, output_channels)
                
                # Global spatial aggregation
                aggregated = self.spatial_processor(
                    batch_voxel_features.unsqueeze(0).transpose(1, 2)
                ).squeeze(0)  # (output_channels,)
                
                batch_features.append(aggregated)
            else:
                # Empty batch
                zero_features = torch.zeros(self.output_channels, device=device)
                batch_features.append(zero_features)
        
        # Stack batch features
        batch_features = torch.stack(batch_features, dim=0)  # (batch_size, output_channels)
        
        # === STEP 3: FORMAT OUTPUT FOR SECOND BACKBONE ===
        # SECOND backbone expects features as direct tensor input (not dict)
        # Create spatial grid representation (simplified)
        spatial_height, spatial_width = 200, 176  # Reduced spatial dimensions
        spatial_features = batch_features.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        spatial_features = spatial_features.expand(-1, -1, spatial_height, spatial_width)
        
        # Return direct tensor (not dict) for SECOND backbone compatibility
        return spatial_features  # (batch_size, output_channels, H, W)
    
    def init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# Register the module
__all__ = ['CPUCompatibleDenseEncoder']
