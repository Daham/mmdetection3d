from mmcv.runner import auto_fp16
from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from mmdet3d.models.voxel_encoders.learnable_vfe import LearnableVFE

@DETECTORS.register_module()
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
        x = self.backbone(voxels, coords, num_points)
        if self.neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(points, img_metas)
        losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes_3d, gt_labels_3d,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, rescale=False):
        x = self.extract_feat(points, img_metas)
        return self.bbox_head.simple_test(x, img_metas, rescale=rescale)

    def aug_test(self, points, img_metas, rescale=False):
        return self.simple_test(points, img_metas, rescale)
