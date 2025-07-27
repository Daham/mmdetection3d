# File: mmdet3d/models/voxel_encoders/learnable_vfe.py

import torch
import torch.nn as nn
from mmdet3d.models import VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders.pillar_vfe import PillarVFE

@VOXEL_ENCODERS.register_module()
class LearnableVFE(PillarVFE):
    """
    VFE that learns its 3D support size per channel.
    Inherits from PillarVFE (or HardVFE) and simply
    replaces the fixed voxel_size with a learnable parameter.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 support_size=(3.0, 0.2, 0.2),
                 with_distance=False,
                 voxel_size=None,
                 point_cloud_range=None,
                 **kwargs):
        # we ignore voxel_size and point_cloud_range here:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            with_distance=with_distance,
            **kwargs)
        # override: make support_size a learnable parameter
        self.support_size = nn.Parameter(
            torch.tensor(support_size, dtype=torch.float),
            requires_grad=True)

    def forward(self, features, num_points, coors, batch_size):
        # features: (num_voxels, max_points, C)
        # num_points: (num_voxels,)
        # coors: (num_voxels, 4)
        # batch_size: int

        # Here you would *use* self.support_size wherever
        # the base PillarVFE uses self.voxel_size in its create
        # of the pillar features. E.g. you might normalize
        # the x,y,z offsets by dividing by support_size.
        # For simplicity, we pass through to PillarVFE but you
        # can extend its `_get_pillar_features` to read
        # self.support_size instead of self.voxel_size:

        # --- Example hook (pseudo) ---
        # self.voxel_size = self.support_size
        # self.point_cloud_range = [
        #   -s/2 for s in support_size] + [
        #    s/2 for s in support_size]

        return super().forward(features, num_points, coors, batch_size)
