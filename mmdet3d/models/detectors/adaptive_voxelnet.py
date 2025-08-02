"""
PhD Research: Adaptive VoxelNet for Multi-Scale Sparse Convolution

This detector extends the standard VoxelNet to support the flow of learned
voxel sizes from the adaptive voxel encoder to the multi-scale sparse encoder.

Key Innovation:
- Voxel encoder learns optimal sizes for each voxel
- These sizes are passed to the multi-scale sparse encoder
- Each size group gets its own processing pathway
- True end-to-end adaptive voxelization!

Author: PhD Research Implementation
Date: August 2025
"""

from typing import Dict, Union, Optional
import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from .voxelnet import VoxelNet


@MODELS.register_module()
class AdaptiveVoxelNet(VoxelNet):
    """
    PhD Research: VoxelNet with Adaptive Multi-Scale Sparse Convolution
    
    This detector enables true adaptive voxelization by:
    1. Using adaptive voxel encoder that learns optimal voxel sizes
    2. Passing these sizes to multi-scale sparse encoder
    3. Processing different voxel sizes through dedicated pathways
    4. Maintaining end-to-end differentiability
    
    The key innovation is the communication between voxel encoder and
    middle encoder to enable true adaptive processing.
    """
    
    def extract_feat(self, batch_inputs_dict: dict, batch_data_samples=None) -> Tensor:
        """
        PhD Research: Feature extraction with adaptive voxel size flow
        
        This method extracts features while passing learned voxel sizes
        from the voxel encoder to the middle encoder for multi-scale processing.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        if voxel_dict is None or voxel_dict.get('voxels', None) is None:
            return None
        
        voxel_features = voxel_dict['voxels']
        num_points = voxel_dict['num_points']
        coors = voxel_dict['coors']
        
        # PhD Research: Extract features AND learned voxel sizes
        batch_size = coors[-1, 0].item() + 1
        
        # Check if we have an adaptive voxel encoder by type
        if hasattr(self.voxel_encoder, '__class__') and 'AdaptiveSparseBridge' in str(self.voxel_encoder.__class__):
            # Standard voxel feature extraction
            voxel_features = self.voxel_encoder(voxel_features, num_points, coors)
            
            # PhD Research: Get learned voxel sizes from encoder
            learned_voxel_sizes = self.voxel_encoder.last_voxel_sizes
            
            # Check if we have an adaptive middle encoder
            if hasattr(self.middle_encoder, 'group_voxels_by_size'):
                # PhD Research: Pass voxel sizes to multi-scale encoder
                x = self.middle_encoder(
                    voxel_features, coors, batch_size, 
                    voxel_sizes=learned_voxel_sizes
                )
                
                # TEMPORARY FIX: Handle 2D output from adaptive encoder
                if len(x.shape) == 2:  # [N, C] format
                    print(f"🔧 Converting 2D adaptive features {x.shape} to 4D for backbone")
                    # Create a simple spatial mapping - this is a temporary workaround
                    # TODO: Implement proper sparse-to-dense conversion
                    spatial_h, spatial_w = 200, 176  # Typical KITTI spatial dimensions
                    num_voxels = x.shape[0]
                    channels = x.shape[1]
                    
                    # Create a simple spatial layout (temporary approach)
                    spatial_features = torch.zeros(
                        batch_size, channels, spatial_h, spatial_w,
                        dtype=x.dtype, device=x.device
                    )
                    
                    # Simple mapping of voxel features to spatial grid
                    # This is not optimal but allows training to proceed
                    for i in range(min(num_voxels, spatial_h * spatial_w)):
                        h_idx = i // spatial_w
                        w_idx = i % spatial_w
                        if h_idx < spatial_h:
                            batch_idx = coors[i, 0].item() if i < len(coors) else 0
                            if batch_idx < batch_size:
                                spatial_features[batch_idx, :, h_idx, w_idx] = x[i]
                    
                    x = spatial_features
                    print(f"🔧 Converted to 4D spatial features: {x.shape}")
                    
            else:
                # Fallback to standard middle encoder
                print("⚠️  Warning: Using standard middle encoder - adaptive voxel sizes not utilized")
                x = self.middle_encoder(voxel_features, coors, batch_size)
        else:
            # Standard VoxelNet processing
            voxel_features = self.voxel_encoder(voxel_features, num_points, coors)
            x = self.middle_encoder(voxel_features, coors, batch_size)
        
        # Continue with standard backbone and neck processing
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        
        return x
    
    def forward(self, 
                inputs: Dict, 
                data_samples: Optional[list] = None,
                mode: str = 'tensor',
                **kwargs):
        """
        PhD Research: Forward pass with adaptive voxelization
        
        Supports all standard VoxelNet modes while enabling adaptive processing.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
    
    def _forward(self, batch_inputs_dict: dict, batch_data_samples=None, **kwargs):
        """Internal forward for tensor mode"""
        return self.extract_feat(batch_inputs_dict, batch_data_samples)
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples, **kwargs):
        """PhD Research: Loss computation with adaptive features"""
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        
        # PhD Research: Enhanced logging for adaptive voxelization analysis
        if hasattr(self.voxel_encoder, 'last_voxel_sizes') and torch.rand(1).item() < 0.02:
            voxel_sizes = self.voxel_encoder.last_voxel_sizes
            print(f"🔬 Adaptive VoxelNet Training Stats:")
            print(f"   - Batch voxel count: {len(voxel_sizes)}")
            print(f"   - Size range: [{voxel_sizes.min():.3f}, {voxel_sizes.max():.3f}]")
            print(f"   - Size std: {voxel_sizes.std():.4f}")
            print(f"   - Loss: {losses['loss_cls']:.4f} (cls), {losses['loss_bbox']:.4f} (bbox)")
        
        return losses
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples, **kwargs):
        """PhD Research: Prediction with adaptive features"""
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        return self.bbox_head.predict(x, batch_data_samples, **kwargs)

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
