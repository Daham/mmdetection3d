# mmdet3d/models/middle_encoders/adaptive_sparse_encoder_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.registry import MODELS
from .sparse_encoder import SparseEncoder

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


@MODELS.register_module()
class AdaptiveSparseEncoderV3(SparseEncoder):
    """
    Advanced Sparse Encoder that supports truly adaptive voxelization.
    
    This encoder extends SparseEncoder to handle:
    1. Dynamic voxel sizes during training
    2. Content-aware voxel processing
    3. Multi-scale adaptive features
    4. Efficient sparse convolutions with adaptive information
    
    The key insight is to use the sparse convolution framework while
    incorporating adaptive voxel size information as additional features
    and attention mechanisms.
    """
    
    def __init__(self, 
                 adaptive_processing=True,
                 adaptive_attention=True,
                 multi_scale_fusion=True,
                 content_aware_weighting=True,
                 adaptive_feature_dim=32,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.adaptive_processing = adaptive_processing
        self.adaptive_attention = adaptive_attention
        self.multi_scale_fusion = multi_scale_fusion
        self.content_aware_weighting = content_aware_weighting
        self.adaptive_feature_dim = adaptive_feature_dim
        
        if self.adaptive_processing:
            self._build_adaptive_layers()
    
    def _build_adaptive_layers(self):
        """Build adaptive processing layers."""
        
        # Adaptive size encoder - converts adaptive sizes to features
        self.size_encoder = nn.Sequential(
            nn.Linear(3, self.adaptive_feature_dim),  # 3D voxel sizes
            nn.LayerNorm(self.adaptive_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.adaptive_feature_dim, self.adaptive_feature_dim),
            nn.LayerNorm(self.adaptive_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Content-aware feature processor
        if self.content_aware_weighting:
            self.content_processor = nn.Sequential(
                nn.Linear(self.in_channels + self.adaptive_feature_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.in_channels),
                nn.Sigmoid()  # Attention weights
            )
        
        # Adaptive attention mechanism
        if self.adaptive_attention:
            self.adaptive_attention_layer = AdaptiveAttentionModule(
                feature_dim=self.in_channels,
                adaptive_dim=self.adaptive_feature_dim,
                hidden_dim=128
            )
        
        # Multi-scale fusion for adaptive features
        if self.multi_scale_fusion:
            self.multi_scale_fusion_layer = MultiScaleFusionModule(
                feature_dim=self.in_channels,
                scales=[1.0, 0.5, 2.0]  # Different scale factors
            )
    
    def forward(self, voxel_features: Tensor, coors: Tensor, 
                batch_size: int, adaptive_info: Optional[Dict] = None) -> Union[Tensor, Tuple[Tensor, list]]:
        """
        Forward pass with adaptive voxel processing.
        
        Args:
            voxel_features (Tensor): [N, C] voxel features
            coors (Tensor): [N, 4] coordinates (batch_idx, z, y, x)  
            batch_size (int): Batch size
            adaptive_info (Dict, optional): Adaptive voxelization information
                - adaptive_sizes: [N, 3] or [3] adaptive voxel sizes
                - density_info: [N] voxel density information
                - content_features: [N, K] content-based features
        
        Returns:
            Tensor or Tuple: Spatial features and optionally middle features
        """
        
        if adaptive_info is not None and self.adaptive_processing:
            voxel_features = self._process_adaptive_features(
                voxel_features, coors, adaptive_info
            )
        
        # Use parent class forward for sparse convolution processing
        return super().forward(voxel_features, coors, batch_size)
    
    def _process_adaptive_features(self, voxel_features: Tensor, coors: Tensor, 
                                 adaptive_info: Dict) -> Tensor:
        """Process voxel features with adaptive information."""
        
        adaptive_sizes = adaptive_info.get('adaptive_sizes', None)
        density_info = adaptive_info.get('density_info', None)
        
        if adaptive_sizes is None:
            return voxel_features
        
        # Encode adaptive size information
        if adaptive_sizes.dim() == 1:  # Global adaptive sizes [3]
            adaptive_sizes = adaptive_sizes.unsqueeze(0).expand(len(voxel_features), -1)
        
        # Normalize adaptive sizes (important for stability)
        normalized_sizes = torch.clamp(adaptive_sizes, min=0.1, max=5.0)
        size_features = self.size_encoder(normalized_sizes)
        
        # Content-aware weighting
        if self.content_aware_weighting:
            combined_features = torch.cat([voxel_features, size_features], dim=1)
            content_weights = self.content_processor(combined_features)
            voxel_features = voxel_features * content_weights
        
        # Adaptive attention
        if self.adaptive_attention:
            voxel_features = self.adaptive_attention_layer(
                voxel_features, size_features, density_info
            )
        
        # Multi-scale fusion
        if self.multi_scale_fusion:
            voxel_features = self.multi_scale_fusion_layer(
                voxel_features, normalized_sizes
            )
        
        return voxel_features


class AdaptiveAttentionModule(nn.Module):
    """Attention module that incorporates adaptive voxel size information."""
    
    def __init__(self, feature_dim: int, adaptive_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.adaptive_dim = adaptive_dim
        
        # Attention computation
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim + adaptive_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: Tensor, size_features: Tensor, 
                density_info: Optional[Tensor] = None) -> Tensor:
        """Apply adaptive attention."""
        
        # Compute attention weights
        attention_input = torch.cat([features, size_features], dim=1)
        attention_weights = self.attention_net(attention_input)
        
        # Transform features
        transformed_features = self.feature_transform(features)
        
        # Apply attention
        attended_features = transformed_features * attention_weights
        
        # Residual connection
        output = features + attended_features
        
        return output


class MultiScaleFusionModule(nn.Module):
    """Multi-scale fusion module for adaptive voxel features."""
    
    def __init__(self, feature_dim: int, scales: List[float] = [0.5, 1.0, 2.0]):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.scales = scales
        self.num_scales = len(scales)
        
        # Scale-specific processing
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * self.num_scales, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Scale selection based on adaptive sizes
        self.scale_selector = nn.Sequential(
            nn.Linear(3, 32),  # 3D voxel sizes
            nn.ReLU(),
            nn.Linear(32, self.num_scales),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features: Tensor, adaptive_sizes: Tensor) -> Tensor:
        """Apply multi-scale processing based on adaptive sizes."""
        
        # Process features at different scales
        scale_features = []
        for i, processor in enumerate(self.scale_processors):
            scale_features.append(processor(features))
        
        # Combine scale features
        combined_features = torch.cat(scale_features, dim=1)
        fused_features = self.fusion(combined_features)
        
        # Scale-aware weighting
        scale_weights = self.scale_selector(adaptive_sizes)
        
        # Weighted combination of original and fused features
        weighted_scales = sum(
            weight.unsqueeze(1) * scale_feat 
            for weight, scale_feat in zip(scale_weights.unbind(1), scale_features)
        )
        
        # Residual connection
        output = features + weighted_scales
        
        return output


@MODELS.register_module() 
class AdaptiveSparseEncoderV3Simple(SparseEncoder):
    """
    Simplified version focusing on core adaptive functionality.
    """
    
    def __init__(self, adaptive_channel_boost=64, **kwargs):
        super().__init__(**kwargs)
        
        self.adaptive_channel_boost = adaptive_channel_boost
        
        # Simple adaptive size processor
        self.size_processor = nn.Sequential(
            nn.Linear(3, self.adaptive_channel_boost),
            nn.ReLU(),
            nn.Linear(self.adaptive_channel_boost, self.in_channels),
            nn.Tanh()  # Bounded adjustment
        )
    
    def forward(self, voxel_features: Tensor, coors: Tensor, 
                batch_size: int, adaptive_info: Optional[Dict] = None) -> Union[Tensor, Tuple[Tensor, list]]:
        """Simple adaptive processing."""
        
        if adaptive_info is not None and 'adaptive_sizes' in adaptive_info:
            adaptive_sizes = adaptive_info['adaptive_sizes']
            
            if adaptive_sizes.dim() == 1:
                adaptive_sizes = adaptive_sizes.unsqueeze(0).expand(len(voxel_features), -1)
            
            # Normalize and process adaptive sizes
            normalized_sizes = torch.clamp(adaptive_sizes, min=0.1, max=3.0)
            size_adjustments = self.size_processor(normalized_sizes)
            
            # Apply adaptive adjustment
            voxel_features = voxel_features + size_adjustments * 0.1  # Small adjustment
        
        return super().forward(voxel_features, coors, batch_size)
