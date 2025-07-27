import torch
from torch import nn
from mmengine.runner import auto_fp16  # Changed from mmcv.runner
from mmdet3d.registry import MODELS
from mmdet3d.models import build_backbone, build_neck, build_head
from mmdet3d.models.builder import build_voxel_encoder

@MODELS.register_module()
class AdaptiveVoxelSECOND(nn.Module):
    """
    SECOND-based 3D detector with per-location learnable voxel support size.
    """
    def __init__(self,
                 reader=None,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        # 1) Voxel encoder uses LearnableVFE instead of HardVFE
        vfe_cfg = reader.copy()
        vfe_cfg['type'] = 'LearnableVFE'
        vfe_cfg['support_size'] = reader['voxel_size']
        self.voxel_encoder = build_voxel_encoder(vfe_cfg)

        # 2) The rest is identical to standard SECOND
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck else None
        self.bbox_head = build_head(bbox_head)

        self.fp16_enabled = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @auto_fp16(apply_to=('points', ))
    def extract_feat(self, points, img_metas=None):
        voxels, coords, num_points = self.voxel_encoder(points)
        # Continue with the rest of your implementation...
        pass

