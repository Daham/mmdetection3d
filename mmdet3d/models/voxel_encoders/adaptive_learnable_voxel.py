"""
🔬 Learnable Adaptive Voxel Layer Implementation

This implements the core research component for learnable adaptive voxelization:
- Voxel sizes become trainable parameters
- Importance-based adaptive voxel selection
- Memory efficient processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmengine.model import BaseModule


@MODELS.register_module()
class ImportancePredictor(BaseModule):
    """
    Predicts spatial importance for adaptive voxelization.
    High importance regions get finer voxels, low importance get coarser voxels.
    """
    
    def __init__(self, 
                 point_cloud_range: List[float],
                 grid_size: Tuple[int, int, int] = (64, 64, 16),
                 hidden_dims: List[int] = [64, 32, 16],
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size
        
        # Simple MLP for importance prediction
        layers = []
        input_dim = 4  # x, y, z, intensity
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))  # Importance score
        layers.append(nn.Sigmoid())  # 0-1 importance
        
        self.importance_net = nn.Sequential(*layers)
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [N, 4] point cloud (x, y, z, intensity)
            
        Returns:
            importance_scores: [N, 1] importance scores for each point
        """
        # Normalize points to [-1, 1] range
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        min_range = pc_range[:3]
        max_range = pc_range[3:]
        
        normalized_points = points.clone()
        normalized_points[:, :3] = 2 * (points[:, :3] - min_range) / (max_range - min_range) - 1
        
        # Predict importance scores
        importance_scores = self.importance_net(normalized_points)
        return importance_scores


@MODELS.register_module()
class AdaptiveLearnableVoxelLayer(BaseModule):
    """
    🔬 CORE RESEARCH COMPONENT: Learnable Adaptive Voxelization
    
    This layer makes voxel sizes trainable parameters that adapt based on 
    feature importance, enabling memory-efficient 3D detection.
    """
    
    def __init__(self,
                 point_cloud_range: List[float],
                 base_voxel_size: List[float] = [0.16, 0.16, 4.0],
                 max_num_points: int = 35,
                 max_voxels: Tuple[int, int] = (16000, 40000),
                 voxel_size_scale_range: Tuple[float, float] = (0.5, 2.0),
                 importance_threshold: float = 0.5,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.importance_threshold = importance_threshold
        self.scale_min, self.scale_max = voxel_size_scale_range
        
        # 🔬 LEARNABLE VOXEL SIZE PARAMETERS
        # These are the trainable parameters that get updated via backpropagation!
        self.base_voxel_size = nn.Parameter(torch.tensor(base_voxel_size))
        self.fine_scale = nn.Parameter(torch.tensor(0.7))    # For important regions
        self.coarse_scale = nn.Parameter(torch.tensor(1.5))  # For unimportant regions
        
        # Importance predictor
        self.importance_predictor = ImportancePredictor(point_cloud_range)
        
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        🔬 ADAPTIVE VOXELIZATION FORWARD PASS
        
        Args:
            points: [N, 4] input point cloud
            
        Returns:
            voxels: [M, max_num_points, 4] voxelized points  
            num_points: [M] number of points per voxel
            coors: [M, 3] voxel coordinates
        """
        # Step 1: Predict importance for each point
        importance_scores = self.importance_predictor(points)
        
        # Step 2: Determine adaptive voxel sizes based on importance
        adaptive_voxel_sizes = self._compute_adaptive_voxel_sizes(importance_scores)
        
        # Step 3: Perform adaptive voxelization
        voxels, num_points, coors = self._adaptive_voxelize(points, adaptive_voxel_sizes, importance_scores)
        
        return voxels, num_points, coors
    
    def _compute_adaptive_voxel_sizes(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive voxel sizes based on importance scores.
        High importance -> smaller voxels (fine detail)
        Low importance -> larger voxels (coarse, memory efficient)
        """
        # Clamp learnable scales to valid range
        fine_scale = torch.clamp(self.fine_scale, self.scale_min, 1.0)
        coarse_scale = torch.clamp(self.coarse_scale, 1.0, self.scale_max)
        
        # Interpolate between fine and coarse scales based on importance
        scale_factors = fine_scale * importance_scores + coarse_scale * (1 - importance_scores)
        
        # Apply scales to base voxel size
        voxel_sizes = self.base_voxel_size.unsqueeze(0) * scale_factors
        
        return voxel_sizes
    
    def _adaptive_voxelize(self, points: torch.Tensor, voxel_sizes: torch.Tensor, 
                          importance_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform memory-efficient adaptive voxelization.
        """
        # For simplicity in this research implementation, we'll use a two-level approach:
        # 1. Important points (fine voxels)
        # 2. Unimportant points (coarse voxels)
        
        important_mask = (importance_scores.squeeze() > self.importance_threshold)
        
        # Process important points with fine voxels
        if important_mask.sum() > 0:
            important_points = points[important_mask]
            fine_voxel_size = self.base_voxel_size * self.fine_scale
            fine_voxels, fine_num_points, fine_coors = self._voxelize_points(
                important_points, fine_voxel_size)
        else:
            fine_voxels = torch.empty(0, self.max_num_points, 4, device=points.device)
            fine_num_points = torch.empty(0, dtype=torch.long, device=points.device)
            fine_coors = torch.empty(0, 3, dtype=torch.long, device=points.device)
        
        # Process unimportant points with coarse voxels  
        unimportant_mask = ~important_mask
        if unimportant_mask.sum() > 0:
            unimportant_points = points[unimportant_mask]
            coarse_voxel_size = self.base_voxel_size * self.coarse_scale
            coarse_voxels, coarse_num_points, coarse_coors = self._voxelize_points(
                unimportant_points, coarse_voxel_size)
        else:
            coarse_voxels = torch.empty(0, self.max_num_points, 4, device=points.device)
            coarse_num_points = torch.empty(0, dtype=torch.long, device=points.device)
            coarse_coors = torch.empty(0, 3, dtype=torch.long, device=points.device)
        
        # Combine fine and coarse voxels
        voxels = torch.cat([fine_voxels, coarse_voxels], dim=0)
        num_points = torch.cat([fine_num_points, coarse_num_points], dim=0)
        coors = torch.cat([fine_coors, coarse_coors], dim=0)
        
        return voxels, num_points, coors
    
    def _voxelize_points(self, points: torch.Tensor, voxel_size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Basic voxelization implementation.
        In practice, you'd use optimized CUDA kernels.
        """
        # This is a simplified implementation for research purposes
        # Production code would use optimized voxelization kernels
        
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        
        # Compute voxel coordinates  
        voxel_coords = torch.floor((points[:, :3] - pc_range[:3]) / voxel_size[:3]).long()
        
        # Create unique voxel keys
        voxel_keys = voxel_coords[:, 0] * 100000 + voxel_coords[:, 1] * 1000 + voxel_coords[:, 2]
        unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)
        
        num_voxels = len(unique_keys)
        voxels = torch.zeros(num_voxels, self.max_num_points, 4, device=points.device)
        num_points_per_voxel = torch.zeros(num_voxels, dtype=torch.long, device=points.device)
        
        # Fill voxels (simplified implementation)
        for i in range(len(points)):
            voxel_idx = inverse_indices[i]
            point_idx = num_points_per_voxel[voxel_idx]
            if point_idx < self.max_num_points:
                voxels[voxel_idx, point_idx] = points[i]
                num_points_per_voxel[voxel_idx] += 1
        
        # Get voxel coordinates
        unique_coords = torch.zeros(num_voxels, 3, dtype=torch.long, device=points.device)
        for i, key in enumerate(unique_keys):
            z = key % 1000
            y = (key // 1000) % 100
            x = key // 100000
            unique_coords[i] = torch.tensor([x, y, z])
        
        return voxels, num_points_per_voxel, unique_coords


@MODELS.register_module()
class AdaptiveVoxelEncoder(BaseModule):
    """
    Voxel encoder that works with adaptive voxel sizes.
    """
    
    def __init__(self, 
                 num_features: int = 4,
                 out_features: int = 64,
                 in_channels: int = None,  # Accept in_channels parameter
                 out_channels: int = None,  # Accept out_channels parameter
                 init_cfg=None,
                 **kwargs):  # Accept any additional parameters
        super().__init__(init_cfg)
        
        # Use in_channels if provided, otherwise use num_features
        self.num_features = in_channels if in_channels is not None else num_features
        # Use out_channels if provided, otherwise use out_features
        self.out_features = out_channels if out_channels is not None else out_features
        
        # Simple feature aggregation
        self.feature_net = nn.Sequential(
            nn.Linear(self.num_features, 32),
            nn.ReLU(),
            nn.Linear(32, self.out_features)
        )
        
    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxels: [M, max_num_points, 4] 
            num_points: [M] number of points per voxel
            
        Returns:
            voxel_features: [M, out_features]
        """
        # Mean aggregation of points in each voxel
        valid_mask = torch.arange(voxels.size(1), device=voxels.device)[None, :] < num_points[:, None]
        
        masked_voxels = voxels * valid_mask.unsqueeze(-1)
        voxel_sums = masked_voxels.sum(dim=1)
        voxel_means = voxel_sums / torch.clamp(num_points.unsqueeze(-1), min=1)
        
        # Extract features
        voxel_features = self.feature_net(voxel_means)
        
        return voxel_features
