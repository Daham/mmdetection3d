# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector

from typing import List, Optional
from mmdet3d.structures import Det3DDataSample

# Type alias for optional data samples list
OptSampleList = Optional[List[Det3DDataSample]]


@MODELS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

    # def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
    #     """Extract features from points."""
    #     voxel_dict = batch_inputs_dict['voxels']
    #     voxel_features = self.voxel_encoder(voxel_dict['voxels'],
    #                                         voxel_dict['num_points'],
    #                                         voxel_dict['coors'])
    #     batch_size = voxel_dict['coors'][-1, 0].item() + 1
    #     x = self.middle_encoder(voxel_features, voxel_dict['coors'],
    #                             batch_size)
    #     x = self.backbone(x)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
# In file: /home/cse/mmdetection_project/mmdetection3d/mmdet3d/models/detectors/voxelnet.py

# In file: /home/cse/mmdetection_project/mmdetection3d/mmdet3d/models/detectors/voxelnet.py


    # In file: /home/cse/mmdetection_project/mmdetection3d/mmdet3d/models/detectors/voxelnet.py


    # def extract_feat(self, batch_inputs_dict: dict,
    #                  batch_data_samples: List['Det3DDataSample']) -> Tensor:
    #     """Extract features from points.
    
    #     Args:
    #         batch_inputs_dict (dict): The batch of inputs, containing voxel info.
    #         batch_data_samples (List[Det3DDataSample]): The batch of data samples,
    #             containing metadata.
    
    #     Returns:
    #         torch.Tensor: The B x C x H x W features after middle encoder.
    #     """
    #     voxel_dict = batch_inputs_dict['voxels']
    
    #     voxel_features, updated_coors = self.voxel_encoder(
    #         voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors'])
    
    #     # --- THIS IS THE FINAL FIX ---
    #     # The batch size is the length of the batch_data_samples list,
    #     # which is now correctly passed into this function.
    #     batch_size = len(batch_data_samples)
    
    #     x = self.middle_encoder(voxel_features, updated_coors, batch_size)
    
    #     x = self.backbone(x)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: OptSampleList = None) -> tuple:
        """Extract features from either voxels or raw points (for adaptive voxelization).
    
        This method supports both standard HardVFE (which returns only voxel_features)
        and custom encoders like AdaptiveVFE (which may return a tuple of outputs).
        The code checks the output type to handle both cases.
        """
        # 🔬 PHD RESEARCH: Handle adaptive voxelization from raw points
        if 'voxels' in batch_inputs_dict:
            # Standard voxelization pipeline
            voxel_dict = batch_inputs_dict['voxels']
            voxel_encoder_out = self.voxel_encoder(
                voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors']
            )
            
            # Handle different encoder output formats
            if isinstance(voxel_encoder_out, tuple):
                voxel_features, updated_coors = voxel_encoder_out[:2]
            else:
                voxel_features = voxel_encoder_out
                updated_coors = voxel_dict['coors']
                
            batch_size = voxel_dict['coors'][-1, 0].item() + 1
            
        elif 'points' in batch_inputs_dict:
            # 🔬 ADAPTIVE VOXELIZATION: Direct from raw points
            points = batch_inputs_dict['points']
            
            # Assume single batch for simplicity (can be extended)
            if isinstance(points, list):
                points = points[0]  # Take first batch
            batch_size = 1
            
            # Call adaptive voxel encoder with raw points
            voxel_encoder_out = self.voxel_encoder(points)
            
            if isinstance(voxel_encoder_out, tuple):
                voxel_features, updated_coors = voxel_encoder_out[:2]
            else:
                raise ValueError(f"Adaptive voxel encoder must return (features, coords), got {type(voxel_encoder_out)}")
        else:
            raise KeyError("Neither 'voxels' nor 'points' found in batch_inputs_dict")
        
        # Continue with middle encoder and backbone
        x = self.middle_encoder(voxel_features, updated_coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x