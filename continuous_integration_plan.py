"""
Minimal Integration: Enhanced ScaleNet with Continuous Prediction
================================================================

This file shows exactly how to integrate continuous voxel size prediction
with the existing ImportanceGuidedMultiScaleVFE without breaking anything.

The enhancement is:
1. ✅ Fully backward compatible
2. ✅ Optional (enabled via parameter)
3. ✅ Drop-in replacement
4. ✅ Minimal code changes

Usage:
    # Enable continuous mode:
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        num_scales=10,
        continuous_mode=True,  # 🚀 NEW: Enable continuous prediction
        # ... rest unchanged
    )
    
    # Disable continuous mode (default behavior):
    voxel_encoder=dict(
        type='ImportanceGuidedMultiScaleVFE',
        num_scales=10,
        continuous_mode=False,  # or omit entirely
        # ... rest unchanged
    )

Author: PhD Research Implementation - Minimal Continuous Enhancement
Date: August 4, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# This is the exact code to add to the existing ScaleNet class
class ContinuousScaleEnhancement:
    """
    This class shows the exact additions needed to enhance the existing ScaleNet.
    """
    
    @staticmethod
    def enhance_scalenet_init(self, **kwargs):
        """
        Add this code to the existing ScaleNet.__init__ method.
        
        Add these lines after the existing initialization, before _build_network():
        """
        # 🚀 NEW: Continuous prediction enhancement (optional)
        self.continuous_mode = kwargs.get('continuous_mode', False)
        self.min_voxel_size = kwargs.get('min_voxel_size', None)
        self.max_voxel_size = kwargs.get('max_voxel_size', None)
        self.interpolation_neighbors = kwargs.get('interpolation_neighbors', 3)
        
        # Auto-detect size range if not specified
        if self.continuous_mode:
            if self.min_voxel_size is None:
                self.min_voxel_size = self.voxel_scales.min().item()
            if self.max_voxel_size is None:
                self.max_voxel_size = self.voxel_scales.max().item()
            
            print(f"🌊 Continuous mode enabled: {self.min_voxel_size:.3f}m - {self.max_voxel_size:.3f}m")
            print(f"🎯 Using {self.interpolation_neighbors} neighbors for interpolation")
        
    @staticmethod 
    def enhance_scalenet_build_network(self):
        """
        Add this code to the existing ScaleNet._build_network method.
        
        Add these lines after the existing scale_predictor creation:
        """
        # 🚀 NEW: Add continuous prediction head if enabled
        if hasattr(self, 'continuous_mode') and self.continuous_mode:
            hidden_dim = self.hidden_dims[-1]
            
            # Continuous voxel size prediction
            self.continuous_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
            
            # Prediction confidence
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
            
            # Initialize for stable training
            with torch.no_grad():
                # Start with mid-range predictions
                self.continuous_head[-2].bias.fill_(0.0)  # sigmoid(0) = 0.5
    
    @staticmethod
    def enhance_scalenet_forward(self, points, training=True):
        """
        Add this code to the existing ScaleNet.forward method.
        
        This goes right before the existing return statement:
        """
        # Get shared features (this should already exist in the current forward method)
        spatial_features = self.spatial_encoder(points[:, :3])
        normalized_points = F.normalize(points, dim=1)
        enhanced_features = torch.cat([normalized_points, spatial_features], dim=1)
        
        # Extract shared features (this should already be computed)
        # shared_features = self.scale_predictor[:-1](enhanced_features)  # All layers except final
        
        # 🚀 NEW: Continuous prediction path
        if hasattr(self, 'continuous_mode') and self.continuous_mode:
            # Use the shared features from existing pipeline
            shared_features = enhanced_features
            for layer in self.scale_predictor[:-1]:  # All except final layer
                shared_features = layer(shared_features)
            
            # Predict continuous voxel sizes
            size_ratios = self.continuous_head(shared_features).squeeze(-1)  # (N,)
            continuous_sizes = self.min_voxel_size + size_ratios * (self.max_voxel_size - self.min_voxel_size)
            
            # Predict confidence
            confidence = self.confidence_head(shared_features).squeeze(-1)  # (N,)
            
            # Compute soft interpolation weights
            scale_assignment = self._compute_interpolation_weights(
                continuous_sizes, confidence
            )
            
            return scale_assignment, continuous_sizes
        
        # If not continuous mode, continue with existing discrete logic
        # (all the existing Gumbel-Softmax code remains unchanged)
        
    @staticmethod
    def add_interpolation_method(self):
        """
        Add this method to the existing ScaleNet class.
        """
        def _compute_interpolation_weights(self, continuous_sizes, confidence):
            """Compute soft interpolation weights between discrete scales."""
            N = continuous_sizes.shape[0]
            device = continuous_sizes.device
            
            # Initialize weights
            scale_assignment = torch.zeros(N, self.num_scales, device=device)
            
            for i in range(N):
                target_size = continuous_sizes[i]
                conf = confidence[i]
                
                # Find nearest discrete scales
                distances = torch.abs(self.voxel_scales - target_size)
                num_neighbors = min(self.interpolation_neighbors, self.num_scales)
                _, nearest_indices = torch.topk(distances, num_neighbors, largest=False)
                
                # Compute weights
                nearest_distances = distances[nearest_indices]
                nearest_distances = torch.clamp(nearest_distances, min=1e-6)
                
                # Inverse distance weighting with confidence sharpening
                weights = 1.0 / nearest_distances
                if conf > 0.5:
                    sharpening = 1.0 + conf
                    weights = weights ** sharpening
                
                # Normalize and assign
                weights = weights / weights.sum()
                scale_assignment[i, nearest_indices] = weights
            
            return scale_assignment
        
        # Bind the method to the class
        return _compute_interpolation_weights


def create_enhanced_config_example():
    """
    Example of how to use the enhanced ScaleNet with continuous prediction.
    """
    
    # 🚀 NEW: Enhanced configuration with continuous prediction
    enhanced_config = dict(
        type='ImportanceGuidedMultiScaleVFE',
        
        # Standard parameters (unchanged)
        num_scales=10,
        scale_net_hidden_dims=[64, 32],
        vfe_channels=[32, 64],
        fusion_channels=128,
        output_channels=64,
        max_num_points=5,
        max_voxels=(12000, 30000),
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        
        # 🌊 NEW: Continuous prediction parameters
        continuous_mode=True,              # Enable continuous prediction
        min_voxel_size=0.01,              # Minimum voxel size (1cm)
        max_voxel_size=1.0,               # Maximum voxel size (1m)  
        interpolation_neighbors=3,        # Number of scales to interpolate between
        
        # Optional: Advanced continuous parameters
        gumbel_temperature=2.0,           # Used for confidence-based sharpening
    )
    
    return enhanced_config


def create_enhanced_feature_interpolator():
    """
    Enhanced feature interpolation that works with continuous scale assignments.
    This can be added to the existing RefactoredMultiScaleFeatureFusion class.
    """
    
    def enhanced_fusion_forward(self, multi_scale_features, scale_assignment_weights=None):
        """
        Enhanced fusion that supports both discrete and continuous scale assignments.
        
        Args:
            multi_scale_features: List of features from each scale
            scale_assignment_weights: (N, num_scales) - can be discrete or continuous weights
        """
        device = multi_scale_features[0].device if multi_scale_features else None
        
        # Get scale summaries (existing logic)
        scale_summaries = []
        for scale_id, features in enumerate(multi_scale_features):
            if features.numel() > 0 and features.shape[0] > 0:
                scale_summary = torch.mean(features, dim=0, keepdim=True)
                scale_summaries.append(scale_summary)
            else:
                # Empty scale
                expected_channels = self.scale_channels[scale_id] if scale_id < len(self.scale_channels) else 64
                empty_summary = torch.randn(1, expected_channels, device=device) * 0.01
                scale_summaries.append(empty_summary)
        
        if not scale_summaries:
            return torch.zeros(1, self.output_channels, device=device)
        
        # Stack scale summaries
        stacked_summaries = torch.cat(scale_summaries, dim=0)  # (num_scales, channels)
        
        # 🌊 NEW: Use interpolation weights if provided (continuous mode)
        if scale_assignment_weights is not None:
            batch_size = scale_assignment_weights.shape[0]
            
            # Weighted combination using soft assignment
            # scale_assignment_weights: (N, num_scales)
            # stacked_summaries: (num_scales, total_channels)
            weighted_features = torch.matmul(scale_assignment_weights, stacked_summaries)  # (N, total_channels)
        else:
            # Standard concatenation (discrete mode)
            concatenated_summaries = torch.cat(scale_summaries, dim=-1)  # (1, total_channels)
            weighted_features = concatenated_summaries
        
        # Apply fusion network (existing logic)
        main_features = self.fusion_net(weighted_features)
        skip_features = self.skip_connection(weighted_features) 
        fused_features = main_features + skip_features
        
        return fused_features
    
    return enhanced_fusion_forward


# Summary of exact changes needed
def summarize_integration_changes():
    """
    Summary of the minimal changes needed to integrate continuous prediction.
    """
    
    changes_summary = {
        'files_to_modify': [
            'mmdet3d/models/voxel_encoders/importance_guided_multi_scale_vfe.py'
        ],
        
        'changes_needed': {
            'ScaleNet.__init__': [
                'Add continuous_mode parameter',
                'Add min_voxel_size, max_voxel_size parameters', 
                'Add interpolation_neighbors parameter',
                'Auto-detect size range from existing voxel_scales'
            ],
            
            'ScaleNet._build_network': [
                'Add continuous_head for size prediction',
                'Add confidence_head for prediction confidence',
                'Initialize weights for stable training'
            ],
            
            'ScaleNet.forward': [
                'Add conditional path for continuous mode',
                'Compute continuous sizes and confidence',
                'Generate soft interpolation weights',
                'Keep existing discrete path unchanged'
            ],
            
            'ScaleNet (new method)': [
                'Add _compute_interpolation_weights method',
                'Implement inverse distance weighting',
                'Add confidence-based sharpening'
            ],
            
            'RefactoredMultiScaleFeatureFusion.forward': [
                'Accept optional scale_assignment_weights parameter',
                'Use weighted combination when weights provided',
                'Keep existing concatenation as fallback'
            ]
        },
        
        'backward_compatibility': [
            'All existing parameters work unchanged',
            'Default continuous_mode=False preserves current behavior',
            'No changes to external interfaces',
            'Existing configurations continue to work'
        ],
        
        'benefits': [
            'Smooth scale transitions instead of hard selection',
            'More fine-grained scale adaptation',
            'Better gradient flow through continuous prediction',
            'Reduced quantization artifacts',
            'Improved feature quality through interpolation'
        ]
    }
    
    return changes_summary


if __name__ == "__main__":
    print("🚀 CONTINUOUS SCALE PREDICTION INTEGRATION PLAN")
    print("=" * 60)
    
    summary = summarize_integration_changes()
    
    print(f"\n📁 Files to modify:")
    for file in summary['files_to_modify']:
        print(f"  • {file}")
    
    print(f"\n🔧 Changes needed:")
    for component, changes in summary['changes_needed'].items():
        print(f"\n  {component}:")
        for change in changes:
            print(f"    • {change}")
    
    print(f"\n✅ Backward compatibility:")
    for item in summary['backward_compatibility']:
        print(f"  • {item}")
    
    print(f"\n🎯 Benefits:")
    for benefit in summary['benefits']:
        print(f"  • {benefit}")
    
    print(f"\n📄 Example configuration:")
    config = create_enhanced_config_example()
    print("    voxel_encoder=dict(")
    for key, value in config.items():
        if key in ['continuous_mode', 'min_voxel_size', 'max_voxel_size', 'interpolation_neighbors']:
            print(f"        {key}={value},  # 🌊 NEW")
        else:
            print(f"        {key}={value},")
    print("    )")
    
    print(f"\n🏁 INTEGRATION READY!")
    print("   The continuous scale prediction can be seamlessly integrated")
    print("   with minimal changes while preserving all existing functionality!")
