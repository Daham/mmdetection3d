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
        # learnable scaling factor for voxel size
        self.scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float), requires_grad=True)
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
        # 1. Compute centroid of each voxel
        points_mean = (features.sum(dim=1) /
                       num_points.type_as(features).view(-1, 1))
        # 2. Expand to per-point
        f_centroid = points_mean[coors[:, 0], :]  # batch gathering
        f_centroid = f_centroid.unsqueeze(1).repeat(1, features.size(1), 1)
        # 3. Local deviation
        f_dev = features - f_centroid
        # 4. Optionally add distance to center
        if self.with_distance:
            # compute point coords from coors, scaled by learnable factor
            voxel_size = self.voxel_size * torch.exp(self.scale)
            pc_range = self.pc_range.view(2, 3)
            coords = coors[:, 1:].float()
            points_xyz = coords * voxel_size + voxel_size / 2 + pc_range[0]
            center = coords * voxel_size + voxel_size / 2 + pc_range[0]
            dist = torch.norm(points_xyz - center, dim=1, keepdim=True)
            dist = dist.unsqueeze(1).repeat(1, features.size(1), 1)
            features = torch.cat([features, dist], dim=-1)

        # 5. Per-point MLP
        pts_feats = self.point_fc(features.view(-1, features.size(-1)))
        pts_feats = pts_feats.view(features.size(0), -1, pts_feats.size(-1))
        # 6. Aggregate by max pooling
        voxel_feats, _ = torch.max(pts_feats, dim=1)
        # 7. Final linear
        voxel_feats = self.voxel_fc(voxel_feats)
        return voxel_feats
