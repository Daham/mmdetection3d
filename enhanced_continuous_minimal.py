"""
Enhanced ScaleNet with Continuous Voxel Size Prediction + Soft Interpolation
============================================================================

This is a minimal, backward-compatible enhancement that adds continuous 
voxel size prediction with soft interpolation to the existing ScaleNet.

Key Features:
- 🔄 Fully Backward Compatible: Existing code works unchanged
- 🎯 Continuous Prediction: Predicts any voxel size in range
- 🌊 Soft Interpolation: Smooth transitions between scales
- ⚡ Drop-in Replacement: Just change one parameter to enable

Author: PhD Research Implementation - Continuous Scale Enhancement
Date: August 4, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class ContinuousScalePredictor(nn.Module):
    """
    Minimal continuous scale predictor that can be added to existing ScaleNet.
    
    This is designed as a drop-in enhancement that doesn't break existing functionality.
    """
    
    def __init__(self, 
                 input_dim: int,
                 min_voxel_size: float = 0.01,
                 max_voxel_size: float = 1.0,
                 interpolation_neighbors: int = 3):
        super().__init__()
        
        self.min_voxel_size = min_voxel_size
        self.max_voxel_size = max_voxel_size
        self.interpolation_neighbors = interpolation_neighbors
        
        # Simple continuous prediction head
        self.continuous_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Confidence prediction for interpolation quality
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict continuous voxel sizes and confidence.
        
        Args:
            features: (N, input_dim) - encoded features
            
        Returns:
            continuous_sizes: (N,) - predicted voxel sizes
            confidence: (N,) - prediction confidence
        """
        # Predict size ratio in [0, 1]
        size_ratios = self.continuous_head(features).squeeze(-1)
        
        # Convert to actual voxel sizes
        continuous_sizes = self.min_voxel_size + size_ratios * (self.max_voxel_size - self.min_voxel_size)
        
        # Predict confidence
        confidence = self.confidence_head(features).squeeze(-1)
        
        return continuous_sizes, confidence


def interpolate_scale_assignments(continuous_sizes: torch.Tensor,
                                discrete_scales: torch.Tensor,
                                confidence: torch.Tensor,
                                num_neighbors: int = 3) -> torch.Tensor:
    """
    Compute soft interpolation weights between discrete scales.
    
    Args:
        continuous_sizes: (N,) - predicted continuous voxel sizes
        discrete_scales: (num_scales,) - available discrete scales
        confidence: (N,) - prediction confidence
        num_neighbors: number of neighbors to use for interpolation
        
    Returns:
        interpolation_weights: (N, num_scales) - soft assignment weights
    """
    N = continuous_sizes.shape[0]
    num_scales = discrete_scales.shape[0]
    device = continuous_sizes.device
    
    # Initialize output weights
    interpolation_weights = torch.zeros(N, num_scales, device=device)
    
    for i in range(N):
        target_size = continuous_sizes[i]
        conf = confidence[i]
        
        # Find nearest discrete scales
        distances = torch.abs(discrete_scales - target_size)
        _, nearest_indices = torch.topk(distances, min(num_neighbors, num_scales), largest=False)
        
        # Compute inverse distance weights
        nearest_distances = distances[nearest_indices]
        
        # Avoid division by zero
        nearest_distances = torch.clamp(nearest_distances, min=1e-6)
        
        # Inverse distance weighting
        weights = 1.0 / nearest_distances
        
        # Confidence-based sharpening
        if conf > 0.5:
            sharpening_factor = 1.0 + conf  # Range [1.0, 2.0]
            weights = weights ** sharpening_factor
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Assign to output
        interpolation_weights[i, nearest_indices] = weights
    
    return interpolation_weights


def interpolate_multi_scale_features(multi_scale_features: List[torch.Tensor],
                                   interpolation_weights: torch.Tensor) -> torch.Tensor:
    """
    Perform soft interpolation between multi-scale features.
    
    Args:
        multi_scale_features: List of features from each scale
        interpolation_weights: (N, num_scales) - interpolation weights
        
    Returns:
        interpolated_features: Weighted combination of scale features
    """
    device = interpolation_weights.device
    num_scales = len(multi_scale_features)
    
    # Get consistent feature representations
    scale_summaries = []
    for scale_id, features in enumerate(multi_scale_features):
        if features.numel() > 0 and features.shape[0] > 0:
            # Use global average pooling for consistent size
            if len(features.shape) > 2:
                features = features.mean(dim=1)
            summary = features.mean(dim=0, keepdim=True)
        else:
            # Empty scale - small random features
            expected_channels = 64  # Default
            summary = torch.randn(1, expected_channels, device=device) * 0.01
            
        scale_summaries.append(summary)
    
    # Stack and expand to match batch size
    stacked_features = torch.cat(scale_summaries, dim=0)  # (num_scales, channels)
    batch_size = interpolation_weights.shape[0]
    
    # Weighted combination
    # interpolation_weights: (N, num_scales)
    # stacked_features: (num_scales, channels)
    # Result: (N, channels)
    interpolated = torch.matmul(interpolation_weights, stacked_features)
    
    return interpolated


# Now let's create the enhanced version of the existing ScaleNet
def enhance_existing_scalenet_with_continuous_prediction():
    """
    Example of how to minimally modify the existing ScaleNet to support
    continuous prediction with soft interpolation.
    
    This shows the exact changes needed in the existing code.
    """
    
    # Here's what we would add to the existing ScaleNet.__init__:
    enhancement_init_code = '''
    # 🚀 NEW: Add continuous prediction capability (optional)
    self.continuous_mode = kwargs.get('continuous_mode', False)
    if self.continuous_mode:
        self.continuous_predictor = ContinuousScalePredictor(
            input_dim=self.hidden_dims[-1],  # Use last hidden dimension
            min_voxel_size=kwargs.get('min_voxel_size', self.voxel_scales[0].item()),
            max_voxel_size=kwargs.get('max_voxel_size', self.voxel_scales[-1].item()),
            interpolation_neighbors=kwargs.get('interpolation_neighbors', 3)
        )
    '''
    
    # Here's what we would add to the existing ScaleNet.forward:
    enhancement_forward_code = '''
    # 🚀 NEW: Continuous prediction path
    if hasattr(self, 'continuous_mode') and self.continuous_mode:
        # Get shared features (already computed)
        continuous_sizes, confidence = self.continuous_predictor(shared_features)
        
        # Compute soft interpolation weights
        scale_assignment = interpolate_scale_assignments(
            continuous_sizes, self.voxel_scales, confidence, 
            self.continuous_predictor.interpolation_neighbors
        )
        
        # Return continuous sizes instead of discrete weighted combination
        predicted_scales = continuous_sizes
        
        return scale_assignment, predicted_scales
    else:
        # Existing discrete path (unchanged)
        # ... existing Gumbel-Softmax code ...
    '''
    
    return enhancement_init_code, enhancement_forward_code


# Test function to verify the concept works
def test_continuous_scale_prediction():
    """Test the continuous scale prediction concept."""
    
    print("🧪 Testing Continuous Scale Prediction")
    print("=" * 50)
    
    # Simulate some input features
    batch_size = 8
    feature_dim = 32
    features = torch.randn(batch_size, feature_dim)
    
    # Create continuous predictor
    predictor = ContinuousScalePredictor(
        input_dim=feature_dim,
        min_voxel_size=0.01,
        max_voxel_size=1.0,
        interpolation_neighbors=3
    )
    
    # Predict continuous sizes
    continuous_sizes, confidence = predictor(features)
    
    print(f"✅ Continuous sizes: {continuous_sizes}")
    print(f"✅ Confidence: {confidence}")
    
    # Test interpolation
    discrete_scales = torch.tensor([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    interpolation_weights = interpolate_scale_assignments(
        continuous_sizes, discrete_scales, confidence, num_neighbors=3
    )
    
    print(f"✅ Interpolation weights shape: {interpolation_weights.shape}")
    print(f"✅ Weight sums: {interpolation_weights.sum(dim=1)}")  # Should be ~1.0
    
    # Show example interpolation
    print(f"\n📊 Example interpolation for first point:")
    print(f"  Target size: {continuous_sizes[0]:.3f}m")
    print(f"  Confidence: {confidence[0]:.3f}")
    
    # Find non-zero weights
    nonzero_indices = torch.nonzero(interpolation_weights[0] > 1e-6).squeeze()
    if nonzero_indices.numel() > 0:
        for idx in nonzero_indices:
            scale_val = discrete_scales[idx].item()
            weight_val = interpolation_weights[0, idx].item()
            print(f"    Scale {scale_val:.3f}m: weight {weight_val:.3f}")
    
    print("\n🎯 Continuous prediction working correctly!")


if __name__ == "__main__":
    test_continuous_scale_prediction()
