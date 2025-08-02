"""
PhD Research: Multi-Scale Adaptive Sparse Convolution Encoder

This module implements a novel sparse convolution approach that can process
voxels of different sizes in parallel through dedicated processing paths.

Key Innovation:
- Each voxel size gets its own processing pathway
- Voxels are grouped by size and processed separately
- Results are combined at the end, maintaining full adaptivity
- No remapping to regular grid needed!

Author: PhD Research Implementation
Date: August 2025
"""

import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from typing import Dict, List, Tuple, Optional

from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential, SubMConv3d
else:
    from mmcv.ops import SparseConvTensor, SparseSequential, SubMConv3d


@MODELS.register_module()
class AdaptiveSparseEncoder(BaseModule):
    """
    PhD Research: Multi-Scale Adaptive Sparse Convolution
    
    This encoder processes adaptive voxel grids where each voxel can have
    different sizes. It groups voxels by size and processes each group
    through dedicated sparse convolution pathways.
    
    Architecture:
    1. Group voxels by their learned sizes
    2. Process each size group through dedicated conv layers
    3. Combine results from all size groups
    4. Apply final fusion layers
    
    This maintains true adaptivity without requiring remapping!
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 sparse_shape: List[int] = [41, 1600, 1408],
                 order: List[str] = ['conv', 'norm', 'act'],
                 norm_cfg: Dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels: int = 16,
                 output_channels: int = 128,
                 encoder_channels: List[int] = [16, 32, 64, 64, 64, 64],
                 encoder_paddings: List[int] = [1, 1, 1, 1, 1, 1],
                 block_type: str = 'conv_module',
                 # Adaptive parameters
                 num_size_groups: int = 4,
                 size_group_ranges: List[Tuple[float, float]] = [
                     (0.05, 0.15),   # Fine voxels
                     (0.15, 0.25),   # Medium-fine voxels  
                     (0.25, 0.35),   # Medium voxels
                     (0.35, 0.50)    # Coarse voxels
                 ],
                 fusion_type: str = 'attention',  # 'concat', 'attention', 'weighted'
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # No need to check spconv availability - handled by compatibility layer
        
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.norm_cfg = norm_cfg
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.block_type = block_type
        
        # Adaptive processing parameters
        self.num_size_groups = num_size_groups
        self.size_group_ranges = size_group_ranges
        self.fusion_type = fusion_type
        
        assert len(size_group_ranges) == num_size_groups, \
            "Number of size groups must match number of ranges"
        
        # Build multi-scale processing pathways
        self._build_size_specific_pathways()
        
        # Build fusion module
        self._build_fusion_module()
        
        # Build final output convolution (like standard SparseEncoder)
        self.conv_out = make_sparse_convmodule(
            self.output_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='adaptive_down2',
            conv_type='SparseConv3d')
        
        print(f"🔬 AdaptiveSparseEncoder initialized with {num_size_groups} size-specific pathways")
        print(f"   Size ranges: {size_group_ranges}")
        print(f"   Fusion type: {fusion_type}")
    
    def _build_size_specific_pathways(self):
        """
        Build dedicated sparse convolution pathways for each voxel size group
        """
        self.size_pathways = nn.ModuleList()
        
        for i, (min_size, max_size) in enumerate(self.size_group_ranges):
            pathway = self._build_single_pathway(f"pathway_{i}")
            self.size_pathways.append(pathway)
            
            print(f"   📍 Pathway {i}: voxel sizes [{min_size:.2f}, {max_size:.2f}]")
    
    def _build_single_pathway(self, pathway_name: str) -> nn.Module:
        """
        Build a single sparse convolution pathway for one size group
        """
        layers = []
        in_ch = self.in_channels
        
        for i, (out_ch, padding) in enumerate(zip(self.encoder_channels, self.encoder_paddings)):
            # Sparse convolution layer
            conv = SubMConv3d(
                in_ch, out_ch,
                kernel_size=3,
                padding=padding,
                bias=False,
                indice_key=f"{pathway_name}_conv{i}"
            )
            
            # Normalization
            norm_name, norm = build_norm_layer(
                dict(type='BN1d', eps=1e-3, momentum=0.01), 
                out_ch
            )
            
            # Activation
            act = nn.ReLU(inplace=True)
            
            # Create block
            if self.block_type == 'conv_module':
                block = SparseSequential(conv, norm, act)
            else:
                block = conv
            
            layers.append(block)
            in_ch = out_ch
        
        # Final output projection for this pathway
        final_conv = SubMConv3d(
            in_ch, self.output_channels,
            kernel_size=1,
            bias=False,
            indice_key=f"{pathway_name}_final"
        )
        layers.append(final_conv)
        
        return SparseSequential(*layers)
    
    def _build_fusion_module(self):
        """
        Build module to fuse outputs from all size-specific pathways
        """
        if self.fusion_type == 'concat':
            # Simple concatenation + projection
            total_channels = self.output_channels * self.num_size_groups
            self.fusion_proj = nn.Linear(total_channels, self.output_channels)
            
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            self.attention_weights = nn.Sequential(
                nn.Linear(self.output_channels, self.output_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.output_channels // 4, 1),
                nn.Sigmoid()
            )
            
        elif self.fusion_type == 'weighted':
            # Learnable weighted combination
            self.pathway_weights = nn.Parameter(torch.ones(self.num_size_groups))
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def group_voxels_by_size(self, 
                           features: torch.Tensor,
                           coordinates: torch.Tensor, 
                           voxel_sizes: torch.Tensor) -> Dict[int, Dict]:
        """
        Group voxels by their learned sizes into processing groups
        
        Args:
            features: [N, C] voxel features
            coordinates: [N, 4] voxel coordinates (batch, z, y, x)
            voxel_sizes: [N,] learned voxel sizes for each voxel
            
        Returns:
            Dictionary mapping group_id -> {features, coordinates, indices}
        """
        groups = {}
        
        for group_id, (min_size, max_size) in enumerate(self.size_group_ranges):
            # Find voxels in this size range
            mask = (voxel_sizes >= min_size) & (voxel_sizes < max_size)
            
            if mask.sum() > 0:
                group_features = features[mask]
                group_coords = coordinates[mask]
                group_indices = torch.where(mask)[0]
                
                groups[group_id] = {
                    'features': group_features,
                    'coordinates': group_coords,
                    'indices': group_indices,
                    'size_range': (min_size, max_size),
                    'count': mask.sum().item()
                }
        
        return groups
    
    def process_size_group(self, 
                          group_data: Dict,
                          pathway: nn.Module,
                          group_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one size group through its dedicated pathway
        """
        features = group_data['features']
        coordinates = group_data['coordinates']
        
        # Create sparse tensor for this group
        sparse_tensor = SparseConvTensor(
            features=features,
            indices=coordinates,
            spatial_shape=self.sparse_shape,
            batch_size=coordinates[:, 0].max().item() + 1
        )
        
        # Process through pathway
        output_tensor = pathway(sparse_tensor)
        
        return output_tensor.features, group_data['indices']
    
    def fuse_pathway_outputs(self, 
                           pathway_outputs: List[torch.Tensor],
                           all_indices: List[torch.Tensor],
                           total_voxels: int) -> torch.Tensor:
        """
        Fuse outputs from all size-specific pathways
        """
        # Reconstruct full feature tensor
        full_features = torch.zeros(
            total_voxels, self.output_channels,
            dtype=pathway_outputs[0].dtype,
            device=pathway_outputs[0].device
        )
        
        if self.fusion_type == 'concat':
            # Concatenate and project
            for features, indices in zip(pathway_outputs, all_indices):
                full_features[indices] = features
                
        elif self.fusion_type == 'attention':
            # Attention-weighted fusion
            pathway_weights = []
            for features, indices in zip(pathway_outputs, all_indices):
                weights = self.attention_weights(features)
                weighted_features = features * weights
                full_features[indices] = weighted_features
                
        elif self.fusion_type == 'weighted':
            # Learnable weighted combination
            for i, (features, indices) in enumerate(zip(pathway_outputs, all_indices)):
                weight = torch.softmax(self.pathway_weights, dim=0)[i]
                full_features[indices] = features * weight
        
        return full_features
    
    def forward(self, voxel_features, coors, batch_size=None, voxel_sizes=None):
        """
        PhD Research: Multi-Scale Adaptive Sparse Convolution Forward Pass
        
        Args:
            voxel_features: [N, C] features from adaptive voxel encoder
            coors: [N, 4] coordinates (batch, z, y, x)
            batch_size: batch size
            voxel_sizes: [N,] learned voxel sizes (REQUIRED for adaptive processing)
            
        Returns:
            spatial_features: [N, C*D, H, W] dense tensor for backbone
        """
        if voxel_sizes is None:
            raise ValueError("voxel_sizes is required for adaptive sparse convolution")
        
        # Group voxels by their learned sizes
        size_groups = self.group_voxels_by_size(voxel_features, coors, voxel_sizes)
        
        # Process each size group through its dedicated pathway
        pathway_outputs = []
        all_indices = []
        
        for group_id, pathway in enumerate(self.size_pathways):
            if group_id in size_groups:
                group_data = size_groups[group_id]
                output_features, indices = self.process_size_group(
                    group_data, pathway, group_id
                )
                pathway_outputs.append(output_features)
                all_indices.append(indices)
                
                # Research logging
                if self.training and torch.rand(1).item() < 0.05:  # 5% logging
                    size_range = group_data['size_range']
                    count = group_data['count']
                    print(f"🔬 Group {group_id} [{size_range[0]:.2f}, {size_range[1]:.2f}]: {count} voxels processed")
        
        # Fuse outputs from all pathways
        if pathway_outputs:
            final_features = self.fuse_pathway_outputs(
                pathway_outputs, all_indices, voxel_features.size(0)
            )
        else:
            # Fallback if no voxels (shouldn't happen in practice)
            final_features = torch.zeros_like(voxel_features[:, :self.output_channels])
        
        # Create final sparse tensor for conv_out and dense conversion
        final_sparse_tensor = SparseConvTensor(
            features=final_features,
            indices=coors,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Apply final convolution (like standard SparseEncoder)
        out = self.conv_out(final_sparse_tensor)
        
        # Convert to dense format [N, C, D, H, W]
        spatial_features = out.dense()
        
        # Reshape for 2D backbone: [N, C, D, H, W] -> [N, C*D, H, W]
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        
        # Research statistics
        if self.training and torch.rand(1).item() < 0.01:  # 1% detailed logging
            print(f"📊 Adaptive Sparse Encoder Stats:")
            print(f"   - Total voxels: {voxel_features.size(0)}")
            print(f"   - Active size groups: {len(pathway_outputs)}/{self.num_size_groups}")
            print(f"   - Final output shape: {spatial_features.shape}")
            print(f"   - Voxel size distribution:")
            for group_id, group_data in size_groups.items():
                print(f"     Group {group_id}: {group_data['count']} voxels ({group_data['count']/voxel_features.size(0)*100:.1f}%)")
        
        return spatial_features
