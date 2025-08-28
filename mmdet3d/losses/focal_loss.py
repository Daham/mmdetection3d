from mmdet.models.losses import FocalLoss as MMDetFocalLoss
from mmdet.registry import MODELS

@MODELS.register_module()
class FocalLoss(MMDetFocalLoss):
    """Wrapper around MMDet's FocalLoss so it can be used in MMDetection3D configs."""
    pass
