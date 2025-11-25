"""
Complete Adaptive Octree VFE implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from .octree_node import OctreeNode
from .octree_builder import AdaptiveOctreeBuilder


@MODELS.register_module()
class AdaptiveOctreeVFE(BaseModule):
    """
    Adaptive Octree-based Voxel Feature Encoder
    
    Implements TRUE dynamic voxelization with learned splitting criteria
    
    This is the core of your PhD research: variable voxel sizes learned end-to-end
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: List[int] = [64, 128, 256],
        max_depth: int = 6,
        min_points_per_voxel: int = 5,
        max_points_per_voxel: int = 100,
        learnable_split: bool = True,
        split_temperature: float = 1.0,
        point_cloud_range: List[float] = [0, -40.0, -3.0, 70.4, 40.0, 1.0],
        init_cfg: Optional[dict] = None
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.max_depth = max_depth
        self.point_cloud_range = torch.tensor(point_cloud_range)
        
        print(f"🌳 Initializing AdaptiveOctreeVFE:")
        print(f"   Max depth: {max_depth} (finest voxel: ~{70.4/(2**max_depth):.3f}m)")
        print(f"   Learnable splitting: {learnable_split}")
        print(f"   Feature channels: {feat_channels}")
        
        # Point feature encoder
        layers = []
        prev_channels = in_channels
        for channels in feat_channels[:-1]:
            layers.extend([
                nn.Linear(prev_channels, channels),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True)
            ])
            prev_channels = channels
        
        layers.append(nn.Linear(prev_channels, feat_channels[-1]))
        self.point_encoder = nn.Sequential(*layers)
        
        # Octree builder
        self.octree_builder = AdaptiveOctreeBuilder(
            feat_channels=feat_channels[-1],
            max_depth=max_depth,
            min_points_per_voxel=min_points_per_voxel,
            max_points_per_voxel=max_points_per_voxel,
            split_temperature=split_temperature,
            learnable_split=learnable_split
        )
        
        # Voxel feature aggregator
        self.voxel_aggregator = nn.Sequential(
            nn.Linear(feat_channels[-1], feat_channels[-1] * 2),
            nn.LayerNorm(feat_channels[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels[-1] * 2, feat_channels[-1])
        )
    
    def forward(
        self,
        points: torch.Tensor,
        num_points: Optional[torch.Tensor] = None,
        coors: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive octree voxelization
        
        Args:
            points: (N, C) point features
            num_points: Ignored for octree (for compatibility)
            coors: Ignored for octree (for compatibility)
        
        Returns:
            Dictionary with adaptive voxel features:
                - voxel_features: (M, C) features for M adaptive voxels
                - voxel_centers: (M, 3) voxel center coordinates
                - voxel_sizes: (M, 1) voxel sizes (VARIABLE!)
                - num_voxels: int, number of voxels
                - octree_stats: dict with build statistics
        """
        device = points.device
        if self.point_cloud_range.device != device:
            self.point_cloud_range = self.point_cloud_range.to(device)
        
        # Encode point features
        point_features = self.point_encoder(points)
        
        # Build adaptive octree - THIS IS WHERE THE MAGIC HAPPENS
        root, build_stats = self.octree_builder.build(
            points,
            point_features,
            self.point_cloud_range
        )
        
        # Extract leaf voxels (these have VARIABLE sizes!)
        leaves = root.get_all_leaves()
        
        if not leaves:
            # Return empty tensors if no leaves
            return {
                'voxel_features': torch.empty(0, self.feat_channels[-1], device=device),
                'voxel_centers': torch.empty(0, 3, device=device),
                'voxel_sizes': torch.empty(0, 1, device=device),
                'num_voxels': 0,
                'octree_stats': build_stats
            }
        
        # Aggregate features for each leaf voxel
        voxel_features = []
        voxel_centers = []
        voxel_sizes = []
        
        for leaf in leaves:
            if leaf.features is not None:
                # Aggregate with learned network
                aggregated = self.voxel_aggregator(leaf.features.unsqueeze(0)).squeeze(0)
                voxel_features.append(aggregated)
                voxel_centers.append(leaf.center)
                voxel_sizes.append(torch.tensor([leaf.get_voxel_size()], device=device))
        
        if not voxel_features:
            return {
                'voxel_features': torch.empty(0, self.feat_channels[-1], device=device),
                'voxel_centers': torch.empty(0, 3, device=device),
                'voxel_sizes': torch.empty(0, 1, device=device),
                'num_voxels': 0,
                'octree_stats': build_stats
            }
        
        return {
            'voxel_features': torch.stack(voxel_features),
            'voxel_centers': torch.stack(voxel_centers),
            'voxel_sizes': torch.stack(voxel_sizes),  # VARIABLE sizes!
            'num_voxels': len(voxel_features),
            'octree_stats': build_stats
        }
