"""
Adaptive Sparse Bridge - THE ONLY MODULE YOU NEED

This module:
1. Learns adaptive voxel sizes during training
2. Creates variable-size voxels based on local point density/features
3. Maps variable voxels back to regular grid for sparse convolution
4. Handles ALL the compatibility issues automatically

Usage:
- Replaces standard VFE
- Feeds directly to standard sparse convolution
- No additional setup required
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import List, Tuple, Dict, Optional
    from mmdet3d.registry import MODELS
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Import warning in adaptive_sparse_bridge: {e}")
    TORCH_AVAILABLE = False
    # Create dummy base class
    class nn:
        class Module:
            pass

if TORCH_AVAILABLE:
    @MODELS.register_module()
    class AdaptiveSparseBridge(nn.Module):
        """
        THE ONLY MODULE YOU NEED for adaptive voxelization + sparse convolution.
        
        Key Features:
        - Learns optimal voxel sizes during training
        - Creates adaptive voxels (dense areas = small voxels, sparse areas = large voxels)
        - Maps variable voxels to regular grid for sparse convolution compatibility
        - Handles conflicts automatically with learned weights
        """
        
        def __init__(self,
                     base_voxel_size: List[float] = [0.05, 0.05, 0.1],
                     point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                     min_voxel_size: List[float] = [0.025, 0.025, 0.05],    # Finest voxels
                     max_voxel_size: List[float] = [0.2, 0.2, 0.4],         # Coarsest voxels
                     adaptation_method: str = 'learned',                    # How to adapt sizes
                     max_points_per_voxel: int = 32,
                     in_channels: int = 4,
                     feat_channels: List[int] = [64],
                     learnable_adaptation: bool = True):                    # Enable learning
            super().__init__()
            
            self.base_voxel_size = torch.tensor(base_voxel_size)
            self.point_cloud_range = torch.tensor(point_cloud_range)
            self.min_voxel_size = torch.tensor(min_voxel_size)
            self.max_voxel_size = torch.tensor(max_voxel_size)
            self.adaptation_method = adaptation_method
            self.max_points_per_voxel = max_points_per_voxel
            self.in_channels = in_channels
            self.feat_channels = feat_channels
            self.learnable_adaptation = learnable_adaptation
            
            # Build the adaptive network that LEARNS voxel sizes
            self.adaptation_network = self._build_adaptation_network()
            
            # Build feature processing network
            self.feature_network = self._build_feature_network()
            
            # Build conflict resolution network (for mapping back to regular grid)
            self.conflict_resolver = self._build_conflict_resolver()
            
            # Regular grid size for sparse convolution output
            self.regular_grid_size = self._compute_regular_grid_size()
            
            print(f"🎯 AdaptiveSparseBridge initialized:")
            print(f"   - Voxel size range: {min_voxel_size} → {max_voxel_size}")
            print(f"   - Regular grid size: {self.regular_grid_size}")
            print(f"   - Learning: {learnable_adaptation}")
        
        def _build_adaptation_network(self):
            """Build network that LEARNS optimal voxel sizes."""
            if self.adaptation_method == 'learned':
                return nn.Sequential(
                    nn.Linear(self.in_channels + 3, 128),  # point features + position
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1), 
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3),  # Output: scale factors for x, y, z
                    nn.Sigmoid()       # [0, 1] → maps to [min_size, max_size]
                )
            else:
                # Simple density-based (non-learned)
                return nn.Sequential(
                    nn.Linear(4, 32),  # density + position
                    nn.ReLU(),
                    nn.Linear(32, 3),
                    nn.Sigmoid()
                )
        
        def _build_feature_network(self):
            """Build feature extraction network."""
            layers = []
            in_dim = self.in_channels + 3  # +3 for voxel size information
            
            for out_dim in self.feat_channels:
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),  # Use LayerNorm instead of BatchNorm1d
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                in_dim = out_dim
            
            return nn.Sequential(*layers)
        
        def _build_conflict_resolver(self):
            """Build network to resolve conflicts when mapping to regular grid."""
            return nn.Sequential(
                nn.Linear(self.feat_channels[-1] + 6, 32),  # features + size info + position
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),  # Weight for this voxel
                nn.Sigmoid()
            )
        
        def _compute_regular_grid_size(self):
            """Compute regular grid size for sparse convolution."""
            pc_range = self.point_cloud_range
            base_voxel = self.base_voxel_size
            
            grid_size = [
                int((pc_range[5] - pc_range[2]) / base_voxel[2]),  # Z
                int((pc_range[4] - pc_range[1]) / base_voxel[1]),  # Y
                int((pc_range[3] - pc_range[0]) / base_voxel[0])   # X
            ]
            return grid_size
        
        def _learn_adaptive_voxel_sizes(self, points: torch.Tensor) -> torch.Tensor:
            """
            LEARN optimal voxel sizes for each point.
            This is where the magic happens - the network learns what voxel size is best!
            """
            device = points.device
            
            # Normalize point positions for learning
            pc_range = self.point_cloud_range.to(device)
            normalized_pos = (points[:, :3] - pc_range[0:3]) / (pc_range[3:6] - pc_range[0:3])
            
            # Combine point features with normalized position
            adaptation_input = torch.cat([points, normalized_pos], dim=1)
            
            # Learn voxel size scales [0, 1]
            size_scales = self.adaptation_network(adaptation_input)
            
            # Map scales to actual voxel sizes
            min_voxel = self.min_voxel_size.to(device)
            max_voxel = self.max_voxel_size.to(device)
            adaptive_voxel_sizes = min_voxel + (max_voxel - min_voxel) * size_scales
            
            return adaptive_voxel_sizes
        
        def _create_adaptive_voxels(self, points: torch.Tensor, adaptive_sizes: torch.Tensor) -> Dict:
            """Create variable-size voxels."""
            device = points.device
            pc_range = self.point_cloud_range.to(device)
            
            # Create adaptive voxels
            adaptive_voxels = {}
            
            for i, (point, voxel_size) in enumerate(zip(points, adaptive_sizes)):
                # Compute adaptive voxel coordinate
                adaptive_coord = ((point[:3] - pc_range[0:3]) / voxel_size).long()
                
                # Create unique key including voxel size (to avoid mixing different sizes)
                voxel_key = (*adaptive_coord.tolist(), *voxel_size.round(decimals=3).tolist())
                
                if voxel_key not in adaptive_voxels:
                    adaptive_voxels[voxel_key] = {
                        'points': [],
                        'sizes': [],
                        'center': adaptive_coord,
                        'voxel_size': voxel_size
                    }
                
                # Add point if voxel not full
                if len(adaptive_voxels[voxel_key]['points']) < self.max_points_per_voxel:
                    adaptive_voxels[voxel_key]['points'].append(point)
                    adaptive_voxels[voxel_key]['sizes'].append(voxel_size)
            
            return adaptive_voxels
        
        def _map_to_regular_grid(self, adaptive_voxels: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            THE KEY STEP: Map variable adaptive voxels to regular grid for sparse convolution.
            This solves the compatibility problem!
            """
            if not adaptive_voxels:
                # Handle empty case
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.zeros(0, self.feat_channels[-1], device=device), \
                       torch.zeros(0, 4, dtype=torch.long, device=device)
            
            device = next(iter(adaptive_voxels.values()))['points'][0].device
            pc_range = self.point_cloud_range.to(device)
            base_voxel = self.base_voxel_size.to(device)
            
            # Regular grid for sparse convolution
            regular_grid = {}
            
            for voxel_data in adaptive_voxels.values():
                if not voxel_data['points']:
                    continue
                    
                # Process points in this adaptive voxel
                voxel_points = torch.stack(voxel_data['points'])
                voxel_sizes = torch.stack(voxel_data['sizes'])
                
                # Extract features for each point in batch
                voxel_points_with_sizes = []
                for j, (point, size) in enumerate(zip(voxel_points, voxel_sizes)):
                    # Add size information to point features
                    point_with_size = torch.cat([point, size])
                    voxel_points_with_sizes.append(point_with_size)
                
                # Process all points in this voxel as a batch
                if voxel_points_with_sizes:
                    batch_input = torch.stack(voxel_points_with_sizes)
                    point_features = self.feature_network(batch_input)
                else:
                    continue
                
                # Map each point to regular grid coordinate
                for point, feature, size in zip(voxel_points, point_features, voxel_sizes):
                    # Map to regular grid using BASE voxel size
                    regular_coord = ((point[:3] - pc_range[0:3]) / base_voxel).long()
                    
                    # Clamp to valid range
                    regular_coord = torch.clamp(regular_coord, 
                                              torch.zeros(3, device=device, dtype=torch.long),
                                              torch.tensor(self.regular_grid_size, device=device, dtype=torch.long) - 1)
                    
                    reg_key = tuple(regular_coord.tolist())
                    
                    if reg_key not in regular_grid:
                        regular_grid[reg_key] = {
                            'features': [],
                            'sizes': [],
                            'positions': []
                        }
                    
                    regular_grid[reg_key]['features'].append(feature)
                    regular_grid[reg_key]['sizes'].append(size)
                    regular_grid[reg_key]['positions'].append(point[:3])
            
            # Handle case where no valid voxels were created
            if not regular_grid:
                return torch.zeros(0, self.feat_channels[-1], device=device), \
                       torch.zeros(0, 4, dtype=torch.long, device=device)
            
            # Resolve conflicts using learned weights
            final_features = []
            final_coords = []
            
            for reg_key, grid_data in regular_grid.items():
                if not grid_data['features']:
                    continue
                    
                features = torch.stack(grid_data['features'])
                sizes = torch.stack(grid_data['sizes'])
                positions = torch.stack(grid_data['positions'])
                
                if len(features) == 1:
                    # No conflict
                    final_feature = features[0]
                else:
                    # CONFLICT RESOLUTION: Learn how to combine multiple adaptive voxels
                    # that map to the same regular grid cell
                    conflict_inputs = []
                    for feat, size, pos in zip(features, sizes, positions):
                        conflict_input = torch.cat([feat, size, pos])
                        conflict_inputs.append(conflict_input)
                    
                    conflict_inputs = torch.stack(conflict_inputs)
                    weights = self.conflict_resolver(conflict_inputs).squeeze(-1)
                    weights = weights / (weights.sum() + 1e-8)  # Normalize
                    
                    # Weighted combination
                    final_feature = (features * weights.unsqueeze(-1)).sum(dim=0)
                
                final_features.append(final_feature)
                final_coords.append(torch.tensor([0, reg_key[2], reg_key[1], reg_key[0]], dtype=torch.long, device=device))
            
            if final_features:
                final_features = torch.stack(final_features)
                final_coords = torch.stack(final_coords)
            else:
                final_features = torch.zeros(0, self.feat_channels[-1], device=device)
                final_coords = torch.zeros(0, 4, dtype=torch.long, device=device)
            
            return final_features, final_coords
        
        def forward(self, features, num_points, coors):
            """
            THE MAIN FORWARD PASS:
            1. Learn adaptive voxel sizes
            2. Create variable voxels  
            3. Map to regular grid for sparse convolution
            4. Return regular grid format
            """
            # Reconstruct points from voxel features (limitation of current interface)
            reconstructed_points = []
            for i in range(features.shape[0]):
                n_pts = num_points[i]
                if n_pts > 0:
                    voxel_points = features[i, :n_pts]
                    reconstructed_points.append(voxel_points)
            
            if not reconstructed_points:
                # No points to process
                return features.new_zeros(0, self.feat_channels[-1]), \
                       coors.new_zeros(0, 4), \
                       {'adaptive_info': 'no_points'}
            
            points = torch.cat(reconstructed_points, dim=0)
            
            # Step 1: LEARN adaptive voxel sizes
            adaptive_sizes = self._learn_adaptive_voxel_sizes(points)
            
            # Step 2: Create variable-size voxels
            adaptive_voxels = self._create_adaptive_voxels(points, adaptive_sizes)
            
            # Step 3: Map to regular grid (SOLVES sparse convolution compatibility)
            regular_features, regular_coords = self._map_to_regular_grid(adaptive_voxels)
            
            # Return format compatible with sparse convolution
            adaptive_info = {
                'num_adaptive_voxels': len(adaptive_voxels),
                'num_regular_voxels': len(regular_features),
                'size_range_used': {
                    'min': adaptive_sizes.min(dim=0)[0],
                    'max': adaptive_sizes.max(dim=0)[0],
                    'mean': adaptive_sizes.mean(dim=0)
                },
                'learning_enabled': self.learnable_adaptation
            }
            
            return regular_features, regular_coords, adaptive_info

else:
    # Fallback when torch is not available
    class AdaptiveSparseBridge:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AdaptiveSparseBridge")
