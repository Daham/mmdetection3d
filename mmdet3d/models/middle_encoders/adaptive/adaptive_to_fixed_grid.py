"""Middle encoder: Convert adaptive octree voxels to fixed grid for detection head.

This bridge component converts variable-sized voxels to a fixed sparse grid
that detection heads (RPN, etc.) expect.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from mmdet3d.registry import MODELS


@MODELS.register_module()
class AdaptiveToFixedGridEncoder(nn.Module):
    """Convert adaptive octree output to fixed grid for detection head.
    
    The octree produces variable-sized voxels, but detection heads expect
    fixed sparse grids. This encoder:
    1. Projects variable voxels to fixed grid cells
    2. Aggregates multiple adaptive voxels per grid cell
    3. Outputs sparse grid compatible with sparse convolutions
    
    Args:
        in_channels (int): Input feature dimension from octree backbone
        out_channels (int): Output feature dimension for detection head
        grid_size (list): Fixed grid size [X, Y, Z]
        voxel_size (list): Fixed voxel size [dx, dy, dz] in meters
        point_cloud_range (list): [xmin, ymin, zmin, xmax, ymax, zmax]
        aggregation (str): How to aggregate ('max', 'mean', 'attention')
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        grid_size: list = [480, 360, 32],
        voxel_size: list = [0.1, 0.1, 0.2],
        point_cloud_range: list = [0, -40, -3, 70.4, 40, 1],
        aggregation: str = 'attention'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.aggregation = aggregation
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention-based aggregation
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(out_channels + 1, 64),  # +1 for voxel size
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def adaptive_to_grid_coords(
        self, 
        voxel_coords: torch.Tensor,
        voxel_sizes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map adaptive voxel coordinates to fixed grid indices.
        
        Args:
            voxel_coords: [N, 3] - (x, y, z) in meters
            voxel_sizes: [N] - voxel size in meters
            
        Returns:
            grid_indices: [N, 3] - (ix, iy, iz) fixed grid indices
            weights: [N] - importance weights based on voxel size
        """
        # Convert world coordinates to grid indices
        pc_range = torch.tensor(
            self.point_cloud_range, 
            device=voxel_coords.device, 
            dtype=voxel_coords.dtype
        )
        
        voxel_size_tensor = torch.tensor(
            self.voxel_size,
            device=voxel_coords.device,
            dtype=voxel_coords.dtype
        )
        
        # Normalize to [0, grid_size]
        normalized = (voxel_coords - pc_range[:3]) / voxel_size_tensor
        grid_indices = torch.floor(normalized).long()
        
        # Clamp to valid range
        grid_size_tensor = torch.tensor(
            self.grid_size,
            device=voxel_coords.device
        )
        grid_indices = torch.clamp(grid_indices, 0, grid_size_tensor - 1)
        
        # Compute weights: smaller voxels (more precise) get higher weight
        weights = 1.0 / (voxel_sizes + 1e-6)
        weights = weights / weights.sum()  # Normalize
        
        return grid_indices, weights
    
    def aggregate_voxels_per_grid_cell(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_sizes: torch.Tensor,
        grid_indices: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate multiple adaptive voxels into each fixed grid cell.
        
        Args:
            voxel_features: [N, C] - features from octree backbone
            voxel_coords: [N, 3] - original coordinates
            voxel_sizes: [N] - voxel sizes
            grid_indices: [N, 3] - mapped fixed grid indices
            weights: [N] - aggregation weights
            
        Returns:
            fixed_features: [M, C] - aggregated features per grid cell
            fixed_coords: [M, 3] - unique grid cell coordinates
        """
        # Convert grid indices to unique identifiers
        grid_hash = (
            grid_indices[:, 0] * self.grid_size[1] * self.grid_size[2] +
            grid_indices[:, 1] * self.grid_size[2] +
            grid_indices[:, 2]
        )
        
        # Find unique grid cells
        unique_hashes, inverse_indices = torch.unique(
            grid_hash, return_inverse=True
        )
        
        num_unique = len(unique_hashes)
        
        # Aggregate based on strategy
        if self.aggregation == 'max':
            # Max pooling per grid cell
            fixed_features = torch.zeros(
                num_unique, self.out_channels,
                device=voxel_features.device,
                dtype=voxel_features.dtype
            )
            
            for i in range(num_unique):
                mask = inverse_indices == i
                if mask.sum() > 0:
                    fixed_features[i] = voxel_features[mask].max(dim=0)[0]
        
        elif self.aggregation == 'mean':
            # Weighted mean per grid cell
            fixed_features = torch.zeros(
                num_unique, self.out_channels,
                device=voxel_features.device,
                dtype=voxel_features.dtype
            )
            
            for i in range(num_unique):
                mask = inverse_indices == i
                if mask.sum() > 0:
                    w = weights[mask].unsqueeze(-1)
                    fixed_features[i] = (voxel_features[mask] * w).sum(dim=0)
        
        elif self.aggregation == 'attention':
            # Attention-based aggregation
            fixed_features = torch.zeros(
                num_unique, self.out_channels,
                device=voxel_features.device,
                dtype=voxel_features.dtype
            )
            
            for i in range(num_unique):
                mask = inverse_indices == i
                if mask.sum() > 0:
                    features = voxel_features[mask]  # [K, C]
                    sizes = voxel_sizes[mask].unsqueeze(-1)  # [K, 1]
                    
                    # Compute attention scores
                    attn_input = torch.cat([features, sizes], dim=-1)  # [K, C+1]
                    attn_scores = self.attention(attn_input)  # [K, 1]
                    attn_scores = attn_scores / (attn_scores.sum() + 1e-6)
                    
                    # Weighted sum
                    fixed_features[i] = (features * attn_scores).sum(dim=0)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Recover grid coordinates from hash
        fixed_coords = torch.zeros(
            num_unique, 3,
            device=voxel_features.device,
            dtype=torch.long
        )
        
        fixed_coords[:, 2] = unique_hashes % self.grid_size[2]
        temp = unique_hashes // self.grid_size[2]
        fixed_coords[:, 1] = temp % self.grid_size[1]
        fixed_coords[:, 0] = temp // self.grid_size[1]
        
        return fixed_features, fixed_coords
    
    def forward(self, voxel_dict: Dict) -> Dict:
        """Convert adaptive voxels to fixed sparse grid.
        
        Args:
            voxel_dict: Dictionary containing:
                - voxel_features: [N, C] - features from octree backbone
                - voxel_coords: [N, 3] - adaptive voxel coordinates
                - voxel_sizes: [N] - variable voxel sizes
                - batch_indices: [N] - batch index for each voxel
        
        Returns:
            Dict with:
                - voxel_features: [M, C] - fixed grid features
                - voxel_coords: [M, 4] - (batch_idx, z, y, x) for sparse conv
                - spatial_shape: [3] - fixed grid size
        """
        voxel_features = voxel_dict['voxel_features']  # [N, C_in]
        voxel_coords = voxel_dict['voxel_coords']      # [N, 3]
        voxel_sizes = voxel_dict['voxel_sizes']        # [N]
        batch_indices = voxel_dict['batch_indices']    # [N]
        
        batch_size = batch_indices.max().item() + 1
        
        # Transform features
        voxel_features = self.feature_transform(voxel_features)  # [N, C_out]
        
        # Process each batch separately
        all_fixed_features = []
        all_fixed_coords = []
        
        for batch_idx in range(batch_size):
            mask = batch_indices == batch_idx
            
            if mask.sum() == 0:
                continue
            
            batch_features = voxel_features[mask]
            batch_coords = voxel_coords[mask]
            batch_sizes = voxel_sizes[mask]
            
            # Map to fixed grid
            grid_indices, weights = self.adaptive_to_grid_coords(
                batch_coords, batch_sizes
            )
            
            # Aggregate voxels per grid cell
            fixed_features, fixed_coords = self.aggregate_voxels_per_grid_cell(
                batch_features, batch_coords, batch_sizes,
                grid_indices, weights
            )
            
            # Add batch index
            batch_column = torch.full(
                (len(fixed_coords), 1),
                batch_idx,
                device=fixed_coords.device,
                dtype=fixed_coords.dtype
            )
            fixed_coords = torch.cat([batch_column, fixed_coords], dim=-1)  # [M, 4]
            
            all_fixed_features.append(fixed_features)
            all_fixed_coords.append(fixed_coords)
        
        # Concatenate all batches
        final_features = torch.cat(all_fixed_features, dim=0)
        final_coords = torch.cat(all_fixed_coords, dim=0)
        
        return {
            'voxel_features': final_features,
            'voxel_coords': final_coords,
            'spatial_shape': self.grid_size,
            'batch_size': batch_size
        }
