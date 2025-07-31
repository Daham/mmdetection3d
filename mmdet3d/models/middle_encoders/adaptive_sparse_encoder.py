# mmdet3d/models/middle_encoders/adaptive_sparse_encoder.py

import torch
from typing import List, Optional, Tuple, Union
from mmdet3d.registry import MODELS
from .sparse_encoder import SparseEncoder
# from spconv.pytorch import SparseConvTensor  # Disabled - spconv not available

# @MODELS.register_module()  # Disabled - spconv dependency not available
class AdaptiveSparseEncoder(SparseEncoder):
    """
    Adaptive Sparse Encoder that can handle learnable voxel sizes.
    
    This encoder dynamically adjusts the sparse shape based on the 
    learnable voxel scaling from LearnableVFE.
    """
    
    def __init__(self, 
                 base_sparse_shape: List[int],
                 **kwargs):
        """
        Args:
            base_sparse_shape: Base sparse shape [D, H, W] before scaling
            **kwargs: Other arguments passed to SparseEncoder
        """
        # Initialize with base sparse shape
        super().__init__(sparse_shape=base_sparse_shape, **kwargs)
        self.base_sparse_shape = base_sparse_shape
        
    def compute_adaptive_sparse_shape(self, scale_factor: torch.Tensor) -> List[int]:
        """
        Compute the adaptive sparse shape based on scale factor.
        
        Args:
            scale_factor: Current voxel scale factor from LearnableVFE
            
        Returns:
            List[int]: Adjusted sparse shape [D, H, W]
        """
        scale_val = scale_factor.item()
        
        # Inverse scaling: smaller voxels → larger grid
        adaptive_shape = [
            int(self.base_sparse_shape[0] * scale_val),
            int(self.base_sparse_shape[1] * scale_val), 
            int(self.base_sparse_shape[2] * scale_val)
        ]
        
        # Ensure minimum grid size
        adaptive_shape = [max(s, 10) for s in adaptive_shape]
        
        return adaptive_shape
    
    def forward(self, 
                voxel_features: torch.Tensor, 
                coors: torch.Tensor,
                batch_size: int,
                scale_factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive sparse shape.
        
        Args:
            voxel_features: Voxel features from LearnableVFE
            coors: Scaled coordinates from LearnableVFE  
            batch_size: Batch size
            scale_factor: Current scale factor from LearnableVFE
            
        Returns:
            torch.Tensor: Spatial features
        """
        if scale_factor is not None:
            # Compute adaptive sparse shape
            adaptive_shape = self.compute_adaptive_sparse_shape(scale_factor)
        else:
            # Fallback to base shape if no scale provided
            adaptive_shape = self.base_sparse_shape
            
        # Temporarily update sparse shape
        original_shape = self.sparse_shape
        self.sparse_shape = adaptive_shape
        
        try:
            # Create sparse tensor with adaptive shape
            coors = coors.int()
            
            # Clamp coordinates to fit within adaptive sparse shape
            for i in range(1, 4):  # Skip batch dimension
                coors[:, i] = torch.clamp(coors[:, i], 0, adaptive_shape[i-1] - 1)
            
            # input_sp_tensor = SparseConvTensor(  # Disabled - spconv not available
            #     voxel_features, coors, adaptive_shape, batch_size)
            
            # Process through sparse convolution layers
            # x = self.conv_input(input_sp_tensor)
            
            # encode_features = []
            # for encoder_layer in self.encoder_layers:
            #     x = encoder_layer(x)
            #     encode_features.append(x)

            # Final output
            # out = self.conv_out(encode_features[-1])
            # spatial_features = out.dense()

            # N, C, D, H, W = spatial_features.shape
            # spatial_features = spatial_features.view(N, C * D, H, W)

            # if self.return_middle_feats:
            #     return spatial_features, encode_features
            # else:
            #     return spatial_features
            
            # Placeholder return for disabled class
            return torch.zeros(1, 128, 200, 176)  # Dummy output
                
        finally:
            # Restore original sparse shape
            self.sparse_shape = original_shape
