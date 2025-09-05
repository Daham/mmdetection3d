"""
Fixed Multi-Scale VFE for Baseline Comparison
============================================

This module implements a fixed (non-learnable) multi-scale voxel feature encoder
for comparison with the adaptive/learnable multi-scale approach.

Key Differences from Adaptive Version:
- Fixed voxel scales (no nn.Parameter)
- Fixed scale assignment (uniform/round-robin, no learning)
- No Gumbel-Softmax (deterministic assignment)
- Same architecture otherwise for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType


class FixedScaleAssignment(nn.Module):
    """Fixed scale assignment without learning"""
    
    def __init__(self, num_scales: int = 3, assignment_strategy: str = 'uniform'):
        super().__init__()
        self.num_scales = num_scales
        self.assignment_strategy = assignment_strategy
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Fixed scale assignment based on strategy
        
        Args:
            points: (N, 4) point cloud
            
        Returns:
            scale_assignment: (N, num_scales) one-hot assignment
        """
        N = points.shape[0]
        device = points.device
        
        if self.assignment_strategy == 'uniform':
            # Uniform distribution across scales
            assignment = torch.ones(N, self.num_scales, device=device) / self.num_scales
            
        elif self.assignment_strategy == 'round_robin':
            # Round-robin assignment
            assignment = torch.zeros(N, self.num_scales, device=device)
            for i in range(N):
                scale_idx = i % self.num_scales
                assignment[i, scale_idx] = 1.0
                
        elif self.assignment_strategy == 'distance_based':
            # Assign based on distance from origin (simple heuristic)
            distances = torch.norm(points[:, :3], dim=1)
            # Normalize distances to [0, 1] and map to scales
            norm_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            scale_indices = (norm_distances * (self.num_scales - 1)).long()
            assignment = torch.zeros(N, self.num_scales, device=device)
            assignment[torch.arange(N), scale_indices] = 1.0
            
        else:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
            
        return assignment


class FixedScaleVFE(nn.Module):
    """Fixed scale-specific VFE without learnable components"""
    
    def __init__(self, 
                 in_channels: int = 4,
                 feat_channels: List[int] = [32, 64],
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        
        # Build VFE layers
        self.vfe_layers = nn.ModuleList()
        prev_channels = in_channels
        
        for out_channels in feat_channels:
            self.vfe_layers.append(
                nn.Sequential(
                    nn.Linear(prev_channels, out_channels, bias=False),
                    nn.BatchNorm1d(out_channels, eps=norm_cfg['eps'], momentum=norm_cfg['momentum']),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = out_channels
            
    def forward(self, points: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Process points through fixed VFE layers
        
        Args:
            points: (M, max_points, C) voxel points
            num_points: (M,) number of points per voxel
            
        Returns:
            features: (M, feat_channels[-1]) voxel features
        """
        M, max_points, C = points.shape
        
        # Reshape for processing
        points_flat = points.view(-1, C)  # (M*max_points, C)
        
        # Pass through VFE layers
        features = points_flat
        for layer in self.vfe_layers:
            features = layer(features)
            
        # Reshape back to voxel format
        features = features.view(M, max_points, -1)  # (M, max_points, feat_dim)
        
        # Max pooling across points in each voxel
        voxel_features = torch.max(features, dim=1)[0]  # (M, feat_dim)
        
        return voxel_features


@MODELS.register_module()
class SimpleFixedMultiScaleVFE(BaseModule):
    """
    Fixed Multi-Scale VFE for Baseline Comparison
    
    This implements the same multi-scale architecture as the adaptive version
    but with fixed (non-learnable) scale assignment for fair comparison.
    """
    
    def __init__(self,
                 # Multi-scale configuration
                 voxel_scales: List[float] = [0.05, 0.1, 0.2],
                 num_scales: int = 3,
                 
                 # VFE configuration
                 vfe_channels: List[int] = [32, 64],
                 fusion_channels: int = 64,
                 output_channels: int = 3,
                 
                 # Standard VFE parameters
                 max_num_points: int = 5,
                 max_voxels: Tuple[int, int] = (16000, 40000),
                 point_cloud_range: List[float] = None,
                 
                 # Fixed assignment strategy
                 assignment_strategy: str = 'uniform',  # 'uniform', 'round_robin', 'distance_based'
                 
                 # Other parameters
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        
        # Store configuration
        self.voxel_scales = voxel_scales  # Fixed scales (not nn.Parameter)
        self.num_scales = num_scales
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.point_cloud_range = point_cloud_range
        self.assignment_strategy = assignment_strategy
        
        print(f"🎯 FixedMultiScaleVFEBaseline initialized:")
        print(f"   📏 Fixed scales: {[f'{s:.3f}m' for s in self.voxel_scales]}")
        print(f"   🔧 Assignment strategy: {assignment_strategy}")
        print(f"   📊 Output channels: {output_channels}")
        
        # 1. Fixed scale assignment (no learning)
        self.scale_assignment = FixedScaleAssignment(
            num_scales=num_scales,
            assignment_strategy=assignment_strategy
        )
        
        # 2. Scale-specific VFEs (same as adaptive version)
        self.scale_vfes = nn.ModuleList()
        for i in range(num_scales):
            vfe = FixedScaleVFE(
                in_channels=4,  # x, y, z, intensity
                feat_channels=vfe_channels,
                norm_cfg=norm_cfg
            )
            self.scale_vfes.append(vfe)
            
        # 3. Feature fusion (simple concatenation + MLP)
        fusion_input_dim = vfe_channels[-1] * num_scales
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_channels),
            nn.BatchNorm1d(fusion_channels, eps=norm_cfg['eps'], momentum=norm_cfg['momentum']),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_channels, output_channels),
            nn.BatchNorm1d(output_channels, eps=norm_cfg['eps'], momentum=norm_cfg['momentum']),
            nn.ReLU(inplace=True)
        )
        
        # 4. Scale info embedding (for compatibility)
        self.scale_embedding = nn.Parameter(torch.randn(1, 1), requires_grad=False)
        
    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with fixed multi-scale processing
        
        Args:
            voxels: (M, max_points, 4) voxel points
            num_points: (M,) number of points per voxel
            coors: (M, 4) voxel coordinates [batch_idx, z, y, x]
            
        Returns:
            features: (M, output_channels + 1) final features with scale info
            updated_coors: (M, 4) updated coordinates
        """
        M = voxels.shape[0]
        device = voxels.device
        
        # 1. Fixed scale assignment (no learning)
        # Use first point of each voxel for assignment
        representative_points = voxels[:, 0, :]  # (M, 4)
        scale_assignment = self.scale_assignment(representative_points)  # (M, num_scales)
        
        # 2. Process each scale
        scale_features = []
        for scale_idx in range(self.num_scales):
            # Get points assigned to this scale
            scale_weights = scale_assignment[:, scale_idx:scale_idx+1]  # (M, 1)
            
            # Weight the voxels by assignment
            weighted_voxels = voxels * scale_weights.unsqueeze(-1)  # (M, max_points, 4)
            
            # Process through scale-specific VFE
            scale_feat = self.scale_vfes[scale_idx](weighted_voxels, num_points)  # (M, feat_dim)
            scale_features.append(scale_feat)
            
        # 3. Fuse multi-scale features
        concatenated_features = torch.cat(scale_features, dim=1)  # (M, feat_dim * num_scales)
        fused_features = self.feature_fusion(concatenated_features)  # (M, output_channels)
        
        # 4. Add scale information (for compatibility with adaptive version)
        scale_info = torch.mean(scale_assignment, dim=1, keepdim=True)  # (M, 1)
        final_features = torch.cat([fused_features, scale_info], dim=1)  # (M, output_channels + 1)
        
        return final_features, coors
