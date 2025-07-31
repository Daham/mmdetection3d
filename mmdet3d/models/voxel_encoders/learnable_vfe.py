# mmdet3d/models/voxel_encoders/learnable_vfe.py

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

@MODELS.register_module()
class LearnableVFE(nn.Module):
    """Learnable Voxel Feature Encoder, replacing original PillarVFE."""
    def __init__(self,
                 in_channels,
                 feat_channels,
                 with_distance=False,
                 voxel_size=(0.5, 0.5, 0.5),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super().__init__()
        self.with_distance = with_distance
        # number of input features per point (xyz + intensity)
        in_dim = in_channels
        if with_distance:
            in_dim += 1
        # learnable scaling factors for different regions
        # Option: different scales for different Z-levels (height-based)
        self.num_height_regions = 4  # e.g., ground, vehicle, overhead
        self.height_scales = nn.Parameter(
            torch.ones(self.num_height_regions, dtype=torch.float), requires_grad=True)
        
        # Store height boundaries for region assignment
        self.register_buffer('height_boundaries', torch.tensor([-3.0, -1.0, 1.0, 3.0]))
        # project per-point features → hidden feat
        layers = []
        last_channels = in_dim
        for out_ch in feat_channels:
            layers.append(nn.Linear(last_channels, out_ch, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last_channels = out_ch
        self.point_fc = nn.Sequential(*layers)
        # final linear to get voxel-level feature
        self.voxel_fc = nn.Linear(last_channels, feat_channels[-1], bias=False)

        # save default sizes
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))

    def forward(self, features, num_points, coors):
        """
        Args:
            features (torch.Tensor): (sum(V), P, C) per-point features
            num_points (torch.Tensor): (sum(V),) number of points per voxel
            coors (torch.Tensor):  (sum(V), 4) voxel indices (batch, z,y,x)
        Returns:
            voxel_features (torch.Tensor): (sum(V), out_channels)
        """
        # Get current learned voxel size (using average for overall scaling effect)
        avg_scale = torch.exp(self.height_scales.mean())
        
        # 1. Compute centroid of each voxel
        points_mean = (features.sum(dim=1) /
                       num_points.type_as(features).view(-1, 1))
        
        # 2. Compute centroid features for each voxel (simplified)
        # Each voxel already has its own centroid computed above
        f_centroid = points_mean.unsqueeze(1).repeat(1, features.size(1), 1)
                
        # 3. Local deviation
        f_dev = features - f_centroid
        # 4. Region-adaptive feature enhancement (simplified approach)
        if self.with_distance:
            # Since voxelization already happened with fixed grid, we can't change actual voxel sizes
            # Instead, we learn region-specific feature weightings and distance calculations
            pc_range = self.pc_range.view(2, 3)
            coords = coors[:, 1:].float()
            
            # Convert voxel coordinates to world Z coordinates  
            world_z = coords[:, 0] * self.voxel_size[2] + self.voxel_size[2] / 2 + pc_range[0, 2]
            
            # Assign each voxel to a height region
            region_ids = torch.zeros(world_z.size(0), dtype=torch.long, device=world_z.device)
            for i in range(len(self.height_boundaries) - 1):
                boundary_low = self.height_boundaries[i]
                boundary_high = self.height_boundaries[i + 1]
                mask = (world_z >= boundary_low) & (world_z < boundary_high)
                region_ids[mask] = i
            
            # Handle edge case: assign highest region to values above last boundary
            mask_high = world_z >= self.height_boundaries[-1]
            region_ids[mask_high] = len(self.height_boundaries) - 2
            
            # Get region-specific feature weights (not actual voxel size scaling)
            region_weights = torch.exp(self.height_scales[region_ids])  # (num_voxels,)
            
            # Compute distance features with fixed voxel size
            points_xyz = coords * self.voxel_size + self.voxel_size / 2 + pc_range[0]
            center = coords * self.voxel_size + self.voxel_size / 2 + pc_range[0]
            dist = torch.norm(points_xyz - center, dim=1, keepdim=True)
            
            # Apply region-specific weighting to distance features
            dist = dist * region_weights.unsqueeze(1)
            dist = dist.unsqueeze(1).repeat(1, features.size(1), 1)
            features = torch.cat([features, dist], dim=-1)

        # 5. Per-point MLP
        pts_feats = self.point_fc(features.view(-1, features.size(-1)))
        pts_feats = pts_feats.view(features.size(0), -1, pts_feats.size(-1))
        # 6. Aggregate by max pooling
        voxel_feats, _ = torch.max(pts_feats, dim=1)
        # 7. Final linear
        voxel_feats = self.voxel_fc(voxel_feats)
        
        # 8. Apply region-specific weighting to final features if distance was computed
        if self.with_distance:
            # Use the region_weights computed earlier
            pc_range = self.pc_range.view(2, 3)
            coords = coors[:, 1:].float()
            world_z = coords[:, 0] * self.voxel_size[2] + self.voxel_size[2] / 2 + pc_range[0, 2]
            
            region_ids = torch.zeros(world_z.size(0), dtype=torch.long, device=world_z.device)
            for i in range(len(self.height_boundaries) - 1):
                boundary_low = self.height_boundaries[i]
                boundary_high = self.height_boundaries[i + 1]
                mask = (world_z >= boundary_low) & (world_z < boundary_high)
                region_ids[mask] = i
            
            mask_high = world_z >= self.height_boundaries[-1]
            region_ids[mask_high] = len(self.height_boundaries) - 2
            
            final_region_weights = torch.exp(self.height_scales[region_ids])
            voxel_feats = voxel_feats * final_region_weights.unsqueeze(1)
        
        return voxel_feats
