# Adaptive-to-Regular Grid Bridge for Sparse Convolution
# This solves the problem of feeding adaptive voxels to sparse convolution

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from mmdet3d.registry import MODELS


@MODELS.register_module()
class AdaptiveToRegularBridge(nn.Module):
    """
    Bridge that converts adaptive voxelization to regular grid for sparse convolution.
    
    This solves the key challenge: sparse convolution needs regular grids,
    but adaptive voxelization creates irregular structures.
    
    Strategy:
    1. Perform adaptive voxelization with variable sizes
    2. Map adaptive voxels back to a regular base grid
    3. Handle conflicts when multiple adaptive voxels map to same grid cell
    4. Provide regular grid structure for sparse convolution
    """
    
    def __init__(self,
                 base_voxel_size: List[float] = [0.05, 0.05, 0.1],
                 point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                 min_voxel_size: List[float] = [0.025, 0.025, 0.05],
                 max_voxel_size: List[float] = [0.2, 0.2, 0.4],
                 adaptation_method: str = 'density',
                 grid_resolution: int = 32,
                 max_points_per_voxel: int = 32,
                 in_channels: int = 4,
                 feat_channels: List[int] = [64],
                 conflict_resolution: str = 'weighted_average',  # 'max', 'average', 'weighted_average'
                 regular_grid_size: List[int] = [41, 1600, 1408]):  # Output regular grid size
        super().__init__()
        
        self.base_voxel_size = torch.tensor(base_voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.min_voxel_size = torch.tensor(min_voxel_size)
        self.max_voxel_size = torch.tensor(max_voxel_size)
        self.adaptation_method = adaptation_method
        self.grid_resolution = grid_resolution
        self.max_points_per_voxel = max_points_per_voxel
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.conflict_resolution = conflict_resolution
        self.regular_grid_size = regular_grid_size
        
        # Adaptation network
        self.adaptation_net = self._build_adaptation_network()
        
        # Feature processing network
        self.feature_net = self._build_feature_network()
        
        # Conflict resolution network (for when multiple adaptive voxels map to same regular cell)
        if conflict_resolution == 'weighted_average':
            self.conflict_weights_net = nn.Sequential(
                nn.Linear(feat_channels[-1] + 3, 32),  # +3 for voxel size info
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def _build_adaptation_network(self):
        """Build network that predicts voxel size adaptations."""
        if self.adaptation_method == 'density':
            return nn.Sequential(
                nn.Linear(4, 64),  # density + position features
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # scale factors for x, y, z
                nn.Sigmoid()
            )
        else:
            return nn.Sequential(
                nn.Linear(3, 32),  # position only
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Sigmoid()
            )
    
    def _build_feature_network(self):
        """Build feature extraction network."""
        layers = []
        in_dim = self.in_channels + 3  # +3 for local voxel size info
        
        for out_dim in self.feat_channels:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            ])
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _adaptive_voxelize_and_map(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Perform adaptive voxelization and map to regular grid.
        
        Args:
            points: [N, C] input points
            
        Returns:
            regular_features: [M, feat_dim] features on regular grid
            regular_coords: [M, 4] regular grid coordinates
            num_points: [M] number of points per regular voxel
            adaptive_info: Dict with adaptation information
        """
        device = points.device
        pc_range = self.point_cloud_range.to(device)
        min_voxel = self.min_voxel_size.to(device)
        max_voxel = self.max_voxel_size.to(device)
        base_voxel = self.base_voxel_size.to(device)
        
        # Step 1: Create adaptation grid (same as before)
        adaptation_params = self._create_adaptation_grid(points)
        
        # Step 2: Adaptive voxelization with mapping to regular grid
        grid_size = (pc_range[3:6] - pc_range[0:3]) / self.grid_resolution
        grid_coords = ((points[:, :3] - pc_range[0:3]) / grid_size).long()
        grid_coords = torch.clamp(grid_coords, 0, self.grid_resolution - 1)
        
        # Regular grid for output (compatible with sparse convolution)
        regular_voxel_dict = {}
        adaptive_voxel_info = []  # Store info about adaptive voxels
        
        # Process each point
        for i, point in enumerate(points):
            # Get adaptation parameters for this point
            gx, gy, gz = grid_coords[i]
            local_scales = adaptation_params[gz, gy, gx]
            
            # Compute adaptive voxel size
            adaptive_voxel_size = min_voxel + (max_voxel - min_voxel) * local_scales
            
            # Map to regular grid coordinates using BASE voxel size
            regular_coords = ((point[:3] - pc_range[0:3]) / base_voxel).long()
            
            # Clamp to valid range
            regular_coords = torch.clamp(regular_coords, 
                                       torch.tensor([0, 0, 0], device=device),
                                       torch.tensor(self.regular_grid_size, device=device) - 1)
            
            # Create regular voxel key
            reg_key = (int(regular_coords[2]), int(regular_coords[1]), int(regular_coords[0]))  # z, y, x
            
            if reg_key not in regular_voxel_dict:
                regular_voxel_dict[reg_key] = {
                    'points': [],
                    'adaptive_sizes': [],
                    'adaptive_weights': [],
                    'regular_coords': regular_coords.clone()
                }
            
            # Add point if not full
            if len(regular_voxel_dict[reg_key]['points']) < self.max_points_per_voxel:
                # Compute weight based on how well adaptive size matches this regular cell
                size_match_score = 1.0 / (1.0 + torch.norm(adaptive_voxel_size - base_voxel))
                
                # Add point with size and weight info
                point_with_size = torch.cat([point, adaptive_voxel_size])
                regular_voxel_dict[reg_key]['points'].append(point_with_size)
                regular_voxel_dict[reg_key]['adaptive_sizes'].append(adaptive_voxel_size)
                regular_voxel_dict[reg_key]['adaptive_weights'].append(size_match_score)
        
        # Step 3: Convert to tensors and resolve conflicts
        num_voxels = len(regular_voxel_dict)
        regular_features = torch.zeros((num_voxels, self.feat_channels[-1]), device=device)
        regular_coords = torch.zeros((num_voxels, 4), dtype=torch.long, device=device)
        num_points = torch.zeros(num_voxels, dtype=torch.long, device=device)
        
        for idx, (reg_key, voxel_data) in enumerate(regular_voxel_dict.items()):
            points_in_voxel = len(voxel_data['points'])
            num_points[idx] = points_in_voxel
            
            if points_in_voxel > 0:
                # Set coordinates (batch_idx, z, y, x)
                z, y, x = reg_key
                regular_coords[idx] = torch.tensor([0, z, y, x], dtype=torch.long)
                
                # Process features through adaptive feature network
                voxel_points = torch.stack(voxel_data['points'])  # [n_pts, C+3]
                
                # Extract features for each point
                point_features = []
                for j, point_with_size in enumerate(voxel_points):
                    feat = self.feature_net(point_with_size.unsqueeze(0))
                    point_features.append(feat.squeeze(0))
                
                point_features = torch.stack(point_features)  # [n_pts, feat_dim]
                
                # Resolve conflicts using specified method
                if self.conflict_resolution == 'max':
                    final_feature, _ = torch.max(point_features, dim=0)
                elif self.conflict_resolution == 'average':
                    final_feature = torch.mean(point_features, dim=0)
                elif self.conflict_resolution == 'weighted_average':
                    # Use adaptive weights and learned conflict resolution
                    adaptive_weights = torch.stack(voxel_data['adaptive_weights']).to(device)
                    
                    # Learn additional weights based on features and sizes
                    weight_inputs = []
                    for j, point_with_size in enumerate(voxel_points):
                        feat_with_size = torch.cat([point_features[j], point_with_size[-3:]])
                        weight_inputs.append(feat_with_size)
                    
                    weight_inputs = torch.stack(weight_inputs)
                    learned_weights = self.conflict_weights_net(weight_inputs).squeeze(-1)
                    
                    # Combine adaptive and learned weights
                    combined_weights = adaptive_weights * learned_weights
                    combined_weights = combined_weights / (combined_weights.sum() + 1e-8)
                    
                    # Weighted average
                    final_feature = (point_features * combined_weights.unsqueeze(-1)).sum(dim=0)
                
                regular_features[idx] = final_feature
        
        # Prepare adaptive info
        adaptive_info = {
            'adaptation_params': adaptation_params,
            'num_adaptive_voxels': len(regular_voxel_dict),
            'avg_points_per_voxel': num_points.float().mean(),
            'conflict_resolution_method': self.conflict_resolution,
            'regular_grid_size': self.regular_grid_size
        }
        
        return regular_features, regular_coords, num_points, adaptive_info
    
    def _create_adaptation_grid(self, points: torch.Tensor) -> torch.Tensor:
        """Create adaptation grid (simplified version)."""
        device = points.device
        pc_range = self.point_cloud_range.to(device)
        grid_size = (pc_range[3:6] - pc_range[0:3]) / self.grid_resolution
        
        # Simple density-based adaptation
        grid_coords = ((points[:, :3] - pc_range[0:3]) / grid_size).long()
        grid_coords = torch.clamp(grid_coords, 0, self.grid_resolution - 1)
        
        # Count points in each grid cell
        density_grid = torch.zeros(
            (self.grid_resolution, self.grid_resolution, self.grid_resolution),
            device=device
        )
        
        for i in range(len(points)):
            x, y, z = grid_coords[i]
            density_grid[z, y, x] += 1
        
        # Normalize and create adaptation parameters
        density_grid = density_grid / (density_grid.max() + 1e-8)
        
        # Simple adaptation: high density -> small scale, low density -> large scale
        adaptation_params = 1.0 - density_grid.unsqueeze(-1).repeat(1, 1, 1, 3) * 0.8
        
        return adaptation_params
    
    def forward(self, features, num_points, coors):
        """
        Forward pass that bridges adaptive voxelization to regular grid.
        
        Args:
            features: [N, max_points, C] pre-voxelized features (ignored for true adaptation)
            num_points: [N] number of points per voxel (ignored)
            coors: [N, 4] voxel coordinates (ignored)
            
        Returns:
            regular_features: [M, feat_dim] features on regular grid
            regular_coords: [M, 4] regular grid coordinates (compatible with sparse conv)
            adaptive_info: Dict with adaptation information
        """
        # Reconstruct points from voxel features (approximation)
        # In practice, you'd pass original points directly
        reconstructed_points = []
        for i in range(features.shape[0]):
            n_pts = num_points[i]
            if n_pts > 0:
                voxel_points = features[i, :n_pts]
                reconstructed_points.append(voxel_points)
        
        if reconstructed_points:
            points = torch.cat(reconstructed_points, dim=0)
            
            # Apply adaptive voxelization with regular grid mapping
            regular_features, regular_coords, reg_num_points, adaptive_info = \
                self._adaptive_voxelize_and_map(points)
            
            return regular_features, regular_coords, adaptive_info
        else:
            # Fallback
            return features.new_zeros(0, self.feat_channels[-1]), \
                   coors.new_zeros(0, 4), \
                   {}
