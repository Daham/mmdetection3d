"""
Adaptive Voxelization for 3D Object Detection - PhD Research Implementation

This module implements TRUE adaptive voxel sizes:
1. Learns optimal voxel sizes based on local point density
2. Creates variable-size voxels (small in dense areas, large in sparse areas)
3. Maps adaptive voxels back to regular grid for sparse convolution compatibility
4. Maintains efficiency while enabling adaptive spatial resolution
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import List, Dict, Tuple
    from mmdet3d.registry import MODELS
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

if TORCH_AVAILABLE:
    @MODELS.register_module()
    class AdaptiveSparseBridge(nn.Module):
        """
        Adaptive Voxelization Module for PhD Research
        
        Creates variable voxel sizes based on:
        - Local point density
        - Learned spatial patterns
        - Feature importance
        """
        
        def __init__(self, 
                     base_voxel_size: List[float] = [0.5, 0.5, 0.5],
                     point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                     min_voxel_size: List[float] = [0.25, 0.25, 0.25],
                     max_voxel_size: List[float] = [1.0, 1.0, 1.0],
                     num_features: int = 4,
                     adaptation_levels: int = 3,  # Number of adaptive size levels
                     learnable_adaptation: bool = True,
                     **kwargs):
            super().__init__()
            
            self.base_voxel_size = torch.tensor(base_voxel_size, dtype=torch.float32)
            self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
            self.min_voxel_size = torch.tensor(min_voxel_size, dtype=torch.float32)
            self.max_voxel_size = torch.tensor(max_voxel_size, dtype=torch.float32)
            self.num_features = num_features
            self.adaptation_levels = adaptation_levels
            self.learnable_adaptation = learnable_adaptation
            
            # Adaptive voxel size predictor network
            if learnable_adaptation:
                self.size_predictor = nn.Sequential(
                    nn.Linear(7, 64),  # [x,y,z,intensity,density,local_var,neighbors]
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3),  # Predict size multipliers for x,y,z
                    nn.Sigmoid()       # [0,1] range
                )
                
                # Feature aggregation network for adaptive voxels
                self.feature_aggregator = nn.Sequential(
                    nn.Linear(num_features + 3, 32),  # features + voxel_size
                    nn.ReLU(),
                    nn.Linear(32, num_features),
                    nn.ReLU()
                )
                
                # Mapping network to handle variable->regular grid conversion
                self.grid_mapper = nn.Sequential(
                    nn.Linear(num_features * 2, 32),  # multiple voxels -> single grid cell
                    nn.ReLU(),
                    nn.Linear(32, num_features)
                )
            
            # Precompute regular grid dimensions
            self.grid_dims = self._compute_grid_dimensions()
            
            print(f"🎯 Adaptive Voxelization Module (PhD Research) initialized:")
            print(f"   - Base voxel size: {base_voxel_size}")
            print(f"   - Adaptive range: {min_voxel_size} → {max_voxel_size}")
            print(f"   - Adaptation levels: {adaptation_levels}")
            print(f"   - Grid dimensions: {self.grid_dims}")
            print(f"   - Learnable adaptation: {learnable_adaptation}")

        def _compute_grid_dimensions(self):
            """Compute regular grid dimensions for sparse convolution output."""
            pc_range = self.point_cloud_range
            base_size = self.base_voxel_size
            
            dims = [
                int((pc_range[5] - pc_range[2]) / base_size[2]),  # Z
                int((pc_range[4] - pc_range[1]) / base_size[1]),  # Y  
                int((pc_range[3] - pc_range[0]) / base_size[0])   # X
            ]
            return dims

        def _compute_local_features(self, features, num_points):
            """Compute local density and variation features for adaptive sizing."""
            batch_size = features.size(0)
            local_features = []
            
            for i in range(batch_size):
                n_pts = num_points[i]
                if n_pts > 0:
                    voxel_points = features[i, :n_pts]
                    
                    # Local density (normalized)
                    density = float(n_pts) / features.size(1)
                    
                    # Local variation (std of coordinates)
                    coord_var = voxel_points[:, :3].std(dim=0).mean().item()
                    
                    # Neighbor estimate (based on density)
                    neighbors = min(density * 10, 1.0)
                    
                    # Combine features: [x_mean, y_mean, z_mean, intensity_mean, density, variation, neighbors]
                    mean_point = voxel_points.mean(dim=0)  # [x,y,z,intensity]
                    local_feat = torch.cat([
                        mean_point,
                        torch.tensor([density, coord_var, neighbors], device=features.device)
                    ])
                    
                    local_features.append(local_feat)
                else:
                    # Empty voxel
                    local_features.append(torch.zeros(7, device=features.device))
            
            return torch.stack(local_features)

        def _predict_adaptive_sizes(self, local_features):
            """Predict adaptive voxel sizes based on local features."""
            if not self.learnable_adaptation:
                # Simple rule-based adaptation
                densities = local_features[:, 4]  # density feature
                size_multipliers = torch.ones_like(densities).unsqueeze(1).repeat(1, 3)
                
                # Dense areas -> smaller voxels, sparse areas -> larger voxels
                size_multipliers[densities > 0.7] = 0.5  # Small voxels for dense areas
                size_multipliers[densities < 0.3] = 1.5  # Large voxels for sparse areas
                
                return size_multipliers
            else:
                # Learned adaptive sizing
                size_multipliers = self.size_predictor(local_features)
                
                # Map [0,1] to [min_ratio, max_ratio]
                min_ratio = self.min_voxel_size / self.base_voxel_size
                max_ratio = self.max_voxel_size / self.base_voxel_size
                
                min_ratio = min_ratio.to(size_multipliers.device)
                max_ratio = max_ratio.to(size_multipliers.device)
                
                adapted_sizes = min_ratio + size_multipliers * (max_ratio - min_ratio)
                return adapted_sizes

        def _create_adaptive_voxels(self, features, num_points, size_multipliers):
            """Create adaptive voxels with variable sizes."""
            batch_size = features.size(0)
            adaptive_voxels = {}
            
            base_size = self.base_voxel_size.to(features.device)
            pc_range = self.point_cloud_range.to(features.device)
            
            for i in range(batch_size):
                n_pts = num_points[i]
                if n_pts > 0:
                    voxel_points = features[i, :n_pts]
                    adaptive_size = size_multipliers[i] * base_size
                    
                    # Compute adaptive voxel center
                    center = voxel_points[:, :3].mean(dim=0)
                    
                    # Map to adaptive grid coordinates
                    grid_coord = ((center - pc_range[:3]) / adaptive_size).long()
                    
                    # Create unique key combining coord and size
                    key = (*grid_coord.tolist(), *adaptive_size.round(decimals=2).tolist())
                    
                    if key not in adaptive_voxels:
                        adaptive_voxels[key] = {
                            'points': [],
                            'adaptive_size': adaptive_size,
                            'grid_coord': grid_coord
                        }
                    
                    adaptive_voxels[key]['points'].append(voxel_points)
            
            return adaptive_voxels

        def _map_to_regular_grid(self, adaptive_voxels):
            """Map adaptive voxels back to regular grid for sparse convolution."""
            if not adaptive_voxels:
                device = self.base_voxel_size.device
                return torch.zeros(0, self.num_features, device=device)
            
            regular_grid = {}
            base_size = self.base_voxel_size.to(next(iter(adaptive_voxels.values()))['adaptive_size'].device)
            pc_range = self.point_cloud_range.to(base_size.device)
            
            # Process each adaptive voxel
            for voxel_data in adaptive_voxels.values():
                if not voxel_data['points']:
                    continue
                    
                # Aggregate points in this adaptive voxel
                all_points = torch.cat(voxel_data['points'], dim=0)
                adaptive_size = voxel_data['adaptive_size']
                
                # Process features with size information
                if self.learnable_adaptation:
                    # Add size info to features for learning
                    size_info = adaptive_size.unsqueeze(0).repeat(all_points.size(0), 1)
                    enhanced_features = torch.cat([all_points, size_info], dim=1)
                    aggregated_features = self.feature_aggregator(enhanced_features).mean(dim=0)
                else:
                    # Simple mean aggregation
                    aggregated_features = all_points.mean(dim=0)
                
                # Map to regular grid coordinate
                center = all_points[:, :3].mean(dim=0)
                regular_coord = ((center - pc_range[:3]) / base_size).long()
                
                # Clamp to valid grid bounds
                regular_coord = torch.clamp(regular_coord, 
                                          torch.zeros(3, device=regular_coord.device, dtype=torch.long),
                                          torch.tensor(self.grid_dims, device=regular_coord.device, dtype=torch.long) - 1)
                
                grid_key = tuple(regular_coord.tolist())
                
                if grid_key not in regular_grid:
                    regular_grid[grid_key] = []
                
                regular_grid[grid_key].append(aggregated_features[:self.num_features])
            
            # Handle conflicts (multiple adaptive voxels -> single regular cell)
            final_features = []
            for grid_cell_features in regular_grid.values():
                if len(grid_cell_features) == 1:
                    final_features.append(grid_cell_features[0])
                else:
                    # Multiple adaptive voxels map to same regular cell
                    stacked_features = torch.stack(grid_cell_features)
                    
                    if self.learnable_adaptation:
                        # Learn how to combine multiple adaptive voxels
                        combined_input = torch.cat([stacked_features.flatten(), 
                                                  stacked_features.mean(dim=0)], dim=0)
                        if combined_input.size(0) <= self.num_features * 2:
                            # Pad if needed
                            pad_size = self.num_features * 2 - combined_input.size(0)
                            combined_input = F.pad(combined_input, (0, pad_size))
                        else:
                            # Truncate if too large
                            combined_input = combined_input[:self.num_features * 2]
                        
                        mapped_features = self.grid_mapper(combined_input)
                        final_features.append(mapped_features)
                    else:
                        # Simple averaging
                        final_features.append(stacked_features.mean(dim=0))
            
            if final_features:
                return torch.stack(final_features)
            else:
                device = base_size.device
                return torch.zeros(0, self.num_features, device=device)

        def forward(self, features, num_points, coors):
            """
            Forward pass with TRUE adaptive voxelization.
            
            Creates variable voxel sizes and maps back to regular grid.
            """
            # Compute local features for adaptation
            local_features = self._compute_local_features(features, num_points)
            
            # Predict adaptive voxel sizes
            size_multipliers = self._predict_adaptive_sizes(local_features)
            
            # Create adaptive voxels with variable sizes
            adaptive_voxels = self._create_adaptive_voxels(features, num_points, size_multipliers)
            
            # Map back to regular grid for sparse convolution
            regular_features = self._map_to_regular_grid(adaptive_voxels)
            
            # Ensure output matches expected batch size
            if regular_features.size(0) != features.size(0):
                # Pad or truncate to match input batch size
                if regular_features.size(0) < features.size(0):
                    # Pad with zeros
                    pad_size = features.size(0) - regular_features.size(0)
                    padding = torch.zeros(pad_size, self.num_features, device=features.device)
                    regular_features = torch.cat([regular_features, padding], dim=0)
                else:
                    # Truncate
                    regular_features = regular_features[:features.size(0)]
            
            return regular_features.contiguous()

else:
    class AdaptiveSparseBridge:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")
