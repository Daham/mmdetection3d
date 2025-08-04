"""
Enhanced ScaleNet with Continuous Voxel Size Prediction + Soft Interpolation
============================================================================

This is a backward-compatible enhancement to the existing ScaleNet that adds:
1. Continuous voxel size prediction (instead of discrete selection)
2. Soft interpolation between neighboring discrete scales
3. Smooth scale transitions for better feature quality

Features:
- 🔄 Backward Compatible: Existing discrete mode still works
- 🎯 Continuous Prediction: Predicts any voxel size in range
- 🌊 Soft Interpolation: Smooth transitions between scales
- ⚡ Configurable: Can switch between discrete/continuous modes

Author: PhD Research Implementation - Continuous Scale Enhancement
Date: August 4, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from mmdet3d.registry import MODELS


@MODELS.register_module()
class EnhancedContinuousScaleNet(nn.Module):
    """
    Enhanced ScaleNet with continuous voxel size prediction and soft interpolation.
    
    Key Features:
    - Predicts continuous voxel sizes instead of discrete selection
    - Soft interpolation between neighboring discrete scales
    - Backward compatible with existing discrete mode
    - Smooth scale transitions for better feature quality
    """
    
    def __init__(self,
                 in_channels: int = 4,  # x, y, z, intensity
                 hidden_dims: List[int] = [64, 32],
                 num_scales: int = 10,  # Number of discrete scales for interpolation
                 
                 # 🚀 NEW: Continuous prediction parameters
                 continuous_mode: bool = True,  # Enable continuous prediction
                 min_voxel_size: float = 0.01,  # Minimum voxel size (1cm)
                 max_voxel_size: float = 1.0,   # Maximum voxel size (1m)
                 interpolation_neighbors: int = 3,  # Number of neighbors for interpolation
                 
                 # Legacy parameters for backward compatibility
                 temperature: float = 5.0,
                 dropout_rate: float = 0.05):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.num_scales = min(max(num_scales, 3), 10)  # At least 3 scales for interpolation
        self.dropout_rate = dropout_rate
        
        # 🚀 NEW: Continuous prediction parameters
        self.continuous_mode = continuous_mode
        self.min_voxel_size = min_voxel_size
        self.max_voxel_size = max_voxel_size
        self.interpolation_neighbors = min(interpolation_neighbors, num_scales)
        
        # Legacy discrete mode parameters
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_decay = nn.Parameter(torch.tensor(0.9995))
        self.min_temperature = 0.5
        self.iteration_count = 0
        
        # Generate discrete scales for interpolation
        self._generate_discrete_scales()
        
        # Build the enhanced network
        self._build_enhanced_network()
        
        print(f\"🚀 EnhancedContinuousScaleNet: {'Continuous' if continuous_mode else 'Discrete'} mode\")
        print(f\"📏 Scale range: {min_voxel_size:.3f}m - {max_voxel_size:.3f}m\")
        print(f\"🎯 Discrete scales: {[f'{s:.3f}m' for s in self.discrete_scales.tolist()]}\")
        
    def _generate_discrete_scales(self):
        \"\"\"Generate discrete scales for interpolation base.\"\"\"
        # Logarithmic distribution for optimal coverage
        log_min = torch.log(torch.tensor(self.min_voxel_size))
        log_max = torch.log(torch.tensor(self.max_voxel_size))
        log_scales = torch.linspace(log_min, log_max, self.num_scales)
        scales = torch.exp(log_scales)
        
        # Register as buffer for proper device handling
        self.register_buffer('discrete_scales', scales)
        
    def _build_enhanced_network(self):
        \"\"\"Build the enhanced network with both discrete and continuous heads.\"\"\"
        # Enhanced spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8)
        )
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = self.in_channels + 8  # 4 + 8 spatial features
        
        for hidden_dim in self.hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
            
        self.shared_encoder = nn.Sequential(*shared_layers)
        
        if self.continuous_mode:
            # 🚀 NEW: Continuous voxel size prediction head
            self.continuous_head = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05),
                nn.Linear(prev_dim // 2, 1),
                nn.Sigmoid()  # Normalize to [0, 1] range
            )
            
            # Additional confidence prediction for interpolation weighting
            self.confidence_head = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(prev_dim // 2, 1),
                nn.Sigmoid()  # Confidence in [0, 1]
            )
        else:
            # Legacy discrete prediction head
            self.discrete_head = nn.Linear(prev_dim, self.num_scales)
            
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        \"\"\"Initialize network weights for stable training.\"\"\"
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Special initialization for continuous head
        if self.continuous_mode and hasattr(self, 'continuous_head'):
            # Initialize to predict mid-range values initially
            final_layer = self.continuous_head[-2]  # Before sigmoid
            nn.init.zeros_(final_layer.bias)  # Bias = 0 → sigmoid output ≈ 0.5
            
    def forward(self, points: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        \"\"\"
        Enhanced forward pass with continuous or discrete prediction.
        
        Args:
            points: (N, 4) - x, y, z, intensity
            training: whether in training mode
            
        Returns:
            scale_assignment: (N, num_scales) - interpolation weights OR discrete assignment
            predicted_scales: (N,) - continuous voxel sizes for each point
            extra_info: Dict with additional information (confidence, neighbors, etc.)
        \"\"\"
        device = points.device
        N = points.shape[0]
        
        # Enhanced spatial encoding
        spatial_features = self.spatial_encoder(points[:, :3])
        normalized_points = F.normalize(points, dim=1)
        enhanced_features = torch.cat([normalized_points, spatial_features], dim=1)
        
        # Shared feature extraction
        shared_features = self.shared_encoder(enhanced_features)  # (N, hidden_dim)
        
        if self.continuous_mode:
            return self._forward_continuous(shared_features, training)
        else:
            return self._forward_discrete(shared_features, training)
            
    def _forward_continuous(self, features: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        \"\"\"Continuous voxel size prediction with soft interpolation.\"\"\"
        N = features.shape[0]
        device = features.device
        
        # Predict continuous voxel sizes
        size_ratios = self.continuous_head(features).squeeze(-1)  # (N,) in [0, 1]
        predicted_scales = self.min_voxel_size + size_ratios * (self.max_voxel_size - self.min_voxel_size)
        
        # Predict confidence for adaptive interpolation
        confidence = self.confidence_head(features).squeeze(-1)  # (N,) in [0, 1]
        
        # 🌊 SOFT INTERPOLATION: Find neighboring discrete scales and compute weights
        scale_assignment = torch.zeros(N, self.num_scales, device=device)
        neighbor_info = []
        
        for i in range(N):
            target_size = predicted_scales[i].item()
            conf = confidence[i].item()
            
            # Find nearest discrete scales
            distances = torch.abs(self.discrete_scales - target_size)
            _, nearest_indices = torch.topk(distances, self.interpolation_neighbors, largest=False)
            
            # Compute inverse distance weights
            nearest_scales = self.discrete_scales[nearest_indices]
            nearest_distances = distances[nearest_indices]
            
            # Avoid division by zero for exact matches
            nearest_distances = torch.clamp(nearest_distances, min=1e-6)
            
            # Inverse distance weighting with confidence modulation
            weights = 1.0 / nearest_distances
            
            # Confidence-based sharpening: high confidence → sharper weights
            if conf > 0.5:
                sharpening_factor = 2.0 * conf  # Range [1.0, 2.0]
                weights = weights ** sharpening_factor
                
            # Normalize weights
            weights = weights / weights.sum()
            
            # Assign weights to scale_assignment
            scale_assignment[i, nearest_indices] = weights
            
            # Store neighbor information
            neighbor_info.append({
                'target_size': target_size,
                'confidence': conf,
                'neighbors': nearest_indices.tolist(),
                'neighbor_scales': nearest_scales.tolist(),
                'weights': weights.tolist()
            })
        
        # Extra information for analysis
        extra_info = {
            'mode': 'continuous',
            'confidence': confidence,
            'size_ratios': size_ratios,
            'neighbor_info': neighbor_info,
            'interpolation_neighbors': self.interpolation_neighbors
        }
        
        return scale_assignment, predicted_scales, extra_info
        
    def _forward_discrete(self, features: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        \"\"\"Legacy discrete scale selection for backward compatibility.\"\"\"
        # Legacy Gumbel-Softmax implementation
        scale_logits = self.discrete_head(features)
        
        # Temperature scheduling
        if training:
            self.iteration_count += 1
            current_temp = max(
                self.temperature * (self.temperature_decay ** (self.iteration_count // 100)),
                self.min_temperature
            )
            self.temperature.data.fill_(current_temp)
        else:
            current_temp = self.temperature.item()
            
        # Discrete assignment
        if training:
            scale_assignment = F.gumbel_softmax(scale_logits, tau=current_temp, hard=False, dim=1)
        else:
            scale_assignment = F.one_hot(torch.argmax(scale_logits, dim=1), num_classes=self.num_scales).float()
            
        # Compute predicted scales
        predicted_scales = torch.sum(scale_assignment * self.discrete_scales.unsqueeze(0), dim=1)
        
        extra_info = {
            'mode': 'discrete',
            'temperature': current_temp,
            'scale_logits': scale_logits
        }
        
        return scale_assignment, predicted_scales, extra_info
    
    def get_scale_info(self) -> Dict:
        \"\"\"Return comprehensive scale configuration information.\"\"\"
        return {
            'mode': 'continuous' if self.continuous_mode else 'discrete',
            'num_discrete_scales': self.num_scales,
            'discrete_scales': self.discrete_scales.tolist(),
            'continuous_range': f\"{self.min_voxel_size:.3f}m - {self.max_voxel_size:.3f}m\",
            'interpolation_neighbors': self.interpolation_neighbors,
            'scale_ratio': self.max_voxel_size / self.min_voxel_size
        }


@MODELS.register_module()
class EnhancedMultiScaleFeatureInterpolator(nn.Module):
    \"\"\"
    Enhanced feature interpolator that performs soft interpolation between
    discrete scale features based on continuous scale predictions.
    \"\"\"
    
    def __init__(self,
                 scale_channels: List[int] = [64] * 10,
                 fusion_channels: int = 128,
                 output_channels: int = 64,
                 interpolation_mode: str = 'weighted_sum'):  # 'weighted_sum' or 'attention'
        super().__init__()
        
        self.scale_channels = scale_channels
        self.num_scales = len(scale_channels)
        self.interpolation_mode = interpolation_mode
        total_channels = sum(scale_channels)
        
        if interpolation_mode == 'attention':
            # Attention-based interpolation
            self.attention_net = nn.Sequential(
                nn.Linear(total_channels, fusion_channels),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_channels, self.num_scales),
                nn.Softmax(dim=-1)
            )
        
        # Feature fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(total_channels, fusion_channels),
            nn.LayerNorm(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(fusion_channels, fusion_channels // 2),
            nn.LayerNorm(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_channels // 2, output_channels)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(total_channels, output_channels) if total_channels != output_channels else nn.Identity()
        self.output_channels = output_channels
        
    def forward(self, 
                multi_scale_features: List[torch.Tensor], 
                interpolation_weights: torch.Tensor,
                extra_info: Dict = None) -> torch.Tensor:
        \"\"\"
        Perform soft interpolation between multi-scale features.
        
        Args:
            multi_scale_features: List of features from each discrete scale
            interpolation_weights: (N, num_scales) - soft weights for interpolation
            extra_info: Additional information from enhanced scale net
            
        Returns:
            Interpolated and fused features
        \"\"\"
        device = interpolation_weights.device
        
        # Handle empty features
        if not multi_scale_features or all(f.numel() == 0 for f in multi_scale_features):
            return torch.zeros(1, self.output_channels, device=device)
        
        # Ensure all features have the same batch size (use global pooling if needed)
        processed_features = []
        max_batch_size = 1
        
        for scale_id, features in enumerate(multi_scale_features):
            if features.numel() > 0 and features.shape[0] > 0:
                # Global average pooling to get consistent size
                if len(features.shape) > 2:
                    features = features.mean(dim=1)  # Average over spatial/voxel dimension
                
                # Ensure single feature per scale
                scale_summary = features.mean(dim=0, keepdim=True)  # (1, channels)
                processed_features.append(scale_summary)
                max_batch_size = max(max_batch_size, features.shape[0])
            else:
                # Empty scale - use small random features
                expected_channels = self.scale_channels[scale_id]
                empty_features = torch.randn(1, expected_channels, device=device) * 0.01
                processed_features.append(empty_features)
        
        # Concatenate all scale features
        concatenated_features = torch.cat(processed_features, dim=-1)  # (1, total_channels)
        
        # Expand to match interpolation weights batch size
        batch_size = interpolation_weights.shape[0]
        if concatenated_features.shape[0] != batch_size:
            concatenated_features = concatenated_features.expand(batch_size, -1)
        
        # 🌊 SOFT INTERPOLATION: Apply interpolation weights
        if self.interpolation_mode == 'attention':
            # Learn additional attention weights
            attention_weights = self.attention_net(concatenated_features)  # (N, num_scales)
            # Combine with interpolation weights
            combined_weights = interpolation_weights * attention_weights
            combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            combined_weights = interpolation_weights
        
        # Weight the concatenated features by scale importance
        # Split concatenated features back to individual scales
        feature_splits = torch.split(concatenated_features, self.scale_channels, dim=-1)
        
        # Apply soft weighting to each scale's features
        weighted_features = []
        for i, scale_features in enumerate(feature_splits):
            weight = combined_weights[:, i:i+1]  # (N, 1)
            weighted = scale_features * weight  # (N, channels)
            weighted_features.append(weighted)
        
        # Concatenate weighted features
        final_concatenated = torch.cat(weighted_features, dim=-1)  # (N, total_channels)
        
        # Final fusion
        main_features = self.fusion_net(final_concatenated)
        skip_features = self.skip_connection(final_concatenated)
        fused_features = main_features + skip_features
        
        return fused_features


# Demonstration of how to integrate with existing code
def create_enhanced_continuous_vfe_config():
    \"\"\"
    Example configuration for enhanced continuous VFE.
    This shows how to integrate the continuous scale prediction
    with the existing ImportanceGuidedMultiScaleVFE.
    \"\"\"
    
    config = dict(
        type='ImportanceGuidedMultiScaleVFE',
        
        # 🚀 NEW: Enhanced continuous scale configuration
        enhanced_scale_net=dict(
            type='EnhancedContinuousScaleNet',
            continuous_mode=True,  # Enable continuous prediction
            min_voxel_size=0.01,   # 1cm minimum
            max_voxel_size=1.0,    # 1m maximum
            interpolation_neighbors=3,  # Use 3 nearest scales for interpolation
            num_scales=10,         # Number of discrete scales for interpolation
        ),
        
        # Enhanced feature interpolation
        enhanced_fusion=dict(
            type='EnhancedMultiScaleFeatureInterpolator',
            interpolation_mode='weighted_sum',  # or 'attention'
            fusion_channels=128,
            output_channels=64
        ),
        
        # Standard parameters (unchanged)
        vfe_channels=[32, 64],
        max_num_points=5,
        max_voxels=(12000, 30000),
        point_cloud_range=[0, -40, -3, 70.4, 40, 1]
    )
    
    return config


# Register the enhanced modules
__all__ = [
    'EnhancedContinuousScaleNet',
    'EnhancedMultiScaleFeatureInterpolator', 
    'create_enhanced_continuous_vfe_config'
]
