# mmdet3d/models/detectors/adaptive_voxelnet.py

import torch
from mmdet3d.registry import MODELS
from .voxelnet import VoxelNet

@MODELS.register_module()
class AdaptiveVoxelNet(VoxelNet):
    """
    VoxelNet variant that supports learnable voxel sizes.
    
    This detector properly handles the enhanced output from LearnableVFE
    including scaled coordinates and dynamic sparse shapes.
    """
    
    def extract_feat(self, batch_inputs_dict):
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = voxel_dict['voxels']
        num_points = voxel_dict['num_points']
        coors = voxel_dict['coors']
        batch_size = coors[-1, 0] + 1

        # Enhanced VFE forward pass
        vfe_output = self.voxel_encoder(voxel_features, num_points, coors)
        
        # Handle different return formats from LearnableVFE
        if isinstance(vfe_output, tuple) and len(vfe_output) == 3:
            # LearnableVFE returns (features, scaled_coors, scale_factor)
            voxel_features, scaled_coors, scale_factor = vfe_output
            use_adaptive = True
        else:
            # Standard VFE returns only features
            voxel_features = vfe_output
            scaled_coors = coors
            scale_factor = None
            use_adaptive = False

        # Middle encoder processing
        if hasattr(self.middle_encoder, 'forward') and use_adaptive:
            # Use adaptive sparse encoder if available
            if 'scale_factor' in self.middle_encoder.forward.__code__.co_varnames:
                x = self.middle_encoder(voxel_features, scaled_coors, 
                                      batch_size, scale_factor)
            else:
                x = self.middle_encoder(voxel_features, scaled_coors, batch_size)
        else:
            # Standard middle encoder
            x = self.middle_encoder(voxel_features, coors, batch_size)

        # Backbone processing
        if self.with_neck:
            x = self.backbone(x)
            x = self.neck(x)
        else:
            x = self.backbone(x)
        
        return x
    
    def loss(self, batch_inputs_dict, batch_data_samples):
        """Forward function for training."""
        x = self.extract_feat(batch_inputs_dict)
        losses = self.bbox_head.loss(x, batch_data_samples)
        
        # Add regularization loss for scale parameter if using LearnableVFE
        if hasattr(self.voxel_encoder, 'scale'):
            scale_reg_loss = 0.001 * torch.pow(self.voxel_encoder.scale, 2)
            losses['scale_reg_loss'] = scale_reg_loss
            
        return losses
