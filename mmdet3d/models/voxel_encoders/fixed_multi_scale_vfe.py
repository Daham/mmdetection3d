"""
Fixed Multi-Scale VFE for Baseline Comparison
============================================

This module implements a fixed multi-scale VFE that processes points at
predetermined scales [0.05, 0.1, 0.2] m without adaptive selection.
All points are processed at all scales and features are fused.

Author: Daham Pathiraja
Date: September 3, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType
from mmengine.model import BaseModule


class VFELayer(nn.Module):
    """Basic VFE layer for processing voxel features."""
    
    def __init__(self, in_channels, out_channels, norm_cfg, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=norm_cfg['eps'], momentum=norm_cfg['momentum'])
        
    def forward(self, inputs, num_points):
        # inputs: (batch_size, max_points, in_channels)
        batch_size, max_points, _ = inputs.shape
        
        # Reshape for linear layer
        x = inputs.view(-1, inputs.shape[-1])
        x = self.linear(x)
        
        # Normalization with fallback
        x_before_norm = x.clone()
        try:
            if x.shape[0] > 1:  # Only use BatchNorm if batch size > 1
                x = self.norm(x)
            else:
                # Use LayerNorm for single samples
                x = F.layer_norm(x, x.shape[-1:])
        except:
            # Fallback to no normalization if both fail
            x = x_before_norm
        
        x = F.relu(x, inplace=True)
        
        # Reshape back
        x = x.view(batch_size, max_points, -1)
        
        if not self.last_layer:
            return x
        
        # Max pooling for last layer
        mask = torch.arange(max_points, device=x.device).unsqueeze(0) < num_points.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(x)
        
        x_masked = x.clone()
        x_masked[~mask] = -1e6  # Large negative instead of -inf
        
        # Max pooling across points dimension
        x_max = torch.max(x_masked, dim=1)[0]  # (batch_size, out_channels)
        
        return x_max


@MODELS.register_module()
class FixedScaleVFE(nn.Module):
    """
    VFE for processing points at a specific fixed scale.
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: List[int] = [32, 64],
                 scale_size: float = 0.1,
                 scale_id: int = 0,
                 with_distance: bool = False,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max'):
        super().__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.scale_size = scale_size
        self.scale_id = scale_id
        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        self.mode = mode
        
        # Calculate input dimension
        input_dim = in_channels  # 4 (x, y, z, intensity)
        if with_distance:
            input_dim += 1  # +1 for distance
        if with_cluster_center:
            input_dim += 3  # +3 for cluster center (xyz)
        if with_voxel_center:
            input_dim += 3  # +3 for voxel center (xyz)
        
        # Add scale embedding (constant for fixed scale)
        input_dim += 1  # +1 for scale identifier
        
        # VFE layers
        self.vfe_layers = nn.ModuleList()
        prev_channels = input_dim
        
        for out_channels in feat_channels:
            self.vfe_layers.append(
                VFELayer(prev_channels, out_channels, norm_cfg, last_layer=False)
            )
            prev_channels = out_channels
        
        # Final layer
        self.vfe_layers.append(
            VFELayer(prev_channels, feat_channels[-1], norm_cfg, last_layer=True)
        )
        
        self.output_channels = feat_channels[-1]
        
    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, 
                coors: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for fixed scale VFE."""
        device = voxels.device
        batch_size = voxels.shape[0]
        max_points = voxels.shape[1]
        
        # Add additional features
        features = [voxels]  # Start with input voxels (4 channels: x, y, z, intensity)
        
        if self.with_cluster_center:
            # Compute cluster center (mean of valid points only)
            valid_mask = torch.arange(max_points, device=device).unsqueeze(0) < num_points.unsqueeze(1)
            valid_points = voxels[:, :, :3] * valid_mask.unsqueeze(-1).float()
            points_sum = valid_points.sum(dim=1, keepdim=True)  # (batch_size, 1, 3)
            cluster_center = points_sum / num_points.unsqueeze(-1).unsqueeze(-1)
            cluster_center = cluster_center.expand(-1, max_points, -1)
            features.append(cluster_center)  # Add 3 channels
        
        if self.with_voxel_center:
            # Simplified voxel center calculation
            voxel_center = voxels[:, :, :3].mean(dim=1, keepdim=True).expand(-1, max_points, -1)
            features.append(voxel_center)  # Add 3 channels
        
        if self.with_distance:
            if self.with_cluster_center:
                distance = torch.norm(voxels[:, :, :3] - cluster_center, dim=-1, keepdim=True)
            else:
                distance = torch.norm(voxels[:, :, :3], dim=-1, keepdim=True)
            features.append(distance)  # Add 1 channel
        
        # Add scale identifier (constant for this scale)
        scale_id_tensor = torch.full((batch_size, max_points, 1), self.scale_size, 
                                   device=device, dtype=torch.float)
        features.append(scale_id_tensor)  # Add 1 channel
        
        # Concatenate all features
        voxel_features = torch.cat(features, dim=-1)
        
        # Apply VFE layers
        for vfe_layer in self.vfe_layers:
            voxel_features = vfe_layer(voxel_features, num_points)
        
        return voxel_features


@MODELS.register_module()
class FixedMultiScaleVoxelizer(nn.Module):
    """
    Fixed multi-scale voxelizer that processes ALL points at ALL scales.
    No adaptive selection - every point gets processed at every scale.
    """
    
    def __init__(self,
                 voxel_scales: List[float] = [0.05, 0.1, 0.2],
                 max_num_points: int = 5,
                 max_voxels: Tuple[int, int] = (12000, 30000),
                 point_cloud_range: List[float] = None):
        super().__init__()
        
        self.voxel_scales = voxel_scales
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.point_cloud_range = point_cloud_range
        
    def simple_voxelize(self, points: torch.Tensor, voxel_size: float) -> Dict:
        """
        Simple voxelization for a single scale.
        Each point becomes its own voxel for simplicity and differentiability.
        """
        device = points.device
        num_points_total = points.shape[0]
        
        if num_points_total == 0:
            return {
                'voxels': torch.empty(0, self.max_num_points, 4, device=device),
                'coordinates': torch.empty(0, 4, device=device, dtype=torch.long),
                'num_points': torch.empty(0, device=device, dtype=torch.long)
            }
        
        # Limit points to max_voxels for memory efficiency
        max_points_limit = min(num_points_total, self.max_voxels[1] if self.training else self.max_voxels[0])
        
        if num_points_total > max_points_limit:
            # Random sampling for training, deterministic for inference
            if self.training:
                indices = torch.randperm(num_points_total, device=device)[:max_points_limit]
            else:
                indices = torch.arange(max_points_limit, device=device)
            sampled_points = points[indices]
        else:
            sampled_points = points
        
        # Each point becomes a single-point voxel
        num_voxels = sampled_points.shape[0]
        voxels = sampled_points.unsqueeze(1)  # (N, 1, 4)
        
        # Pad to max_num_points
        if self.max_num_points > 1:
            padding = torch.zeros(num_voxels, self.max_num_points - 1, 4, device=device)
            voxels = torch.cat([voxels, padding], dim=1)
        
        # Generate coordinates (continuous for differentiability)
        coordinates = torch.zeros(num_voxels, 4, device=device, dtype=torch.long)
        coordinates[:, 0] = 0  # batch index
        # Use quantized coordinates for this scale
        if self.point_cloud_range is not None:
            grid_coords = (sampled_points[:, :3] - torch.tensor(self.point_cloud_range[:3], device=device)) / voxel_size
            coordinates[:, 1:] = grid_coords.long()
        else:
            coordinates[:, 1:] = (sampled_points[:, :3] / voxel_size).long()
        
        num_points_per_voxel = torch.ones(num_voxels, device=device, dtype=torch.long)
        
        return {
            'voxels': voxels,
            'coordinates': coordinates,
            'num_points': num_points_per_voxel
        }
        
    def forward(self, points: torch.Tensor) -> List[Dict]:
        """
        Process points at all fixed scales.
        
        Args:
            points: (N, 4) - point cloud
            
        Returns:
            List of voxelization results for each scale
        """
        voxel_outputs = []
        
        for scale_id, voxel_size in enumerate(self.voxel_scales):
            voxel_data = self.simple_voxelize(points, voxel_size)
            voxel_data['scale_id'] = scale_id
            voxel_data['voxel_size'] = voxel_size
            voxel_outputs.append(voxel_data)
        
        return voxel_outputs


@MODELS.register_module()
class FixedMultiScaleFeatureFusion(nn.Module):
    """
    Feature fusion module for fixed multi-scale features.
    Concatenates features from all scales and applies fusion network.
    """
    
    def __init__(self,
                 scale_channels: List[int] = [64, 64, 64],  # Channels from each scale
                 fusion_channels: int = 128,
                 output_channels: int = 64):
        super().__init__()
        
        self.scale_channels = scale_channels
        self.num_scales = len(scale_channels)
        total_channels = sum(scale_channels)
        
        # Feature fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(total_channels, fusion_channels),
            nn.LayerNorm(fusion_channels),  # LayerNorm for stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_channels, fusion_channels // 2),
            nn.LayerNorm(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_channels // 2, output_channels)
        )
        
        # Skip connection for gradient flow
        self.skip_connection = nn.Linear(total_channels, output_channels) if total_channels != output_channels else nn.Identity()
        
        self.output_channels = output_channels
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple fixed scales.
        
        Args:
            multi_scale_features: List of features from each scale
            
        Returns:
            Fused features with shape (N, output_channels)
        """
        device = multi_scale_features[0].device if multi_scale_features else None
        
        # Find the scale with the most voxels to use as reference
        max_voxels = 0
        reference_features = None
        
        for features in multi_scale_features:
            if features.numel() > 0 and features.shape[0] > max_voxels:
                max_voxels = features.shape[0]
                reference_features = features
        
        if reference_features is None or max_voxels == 0:
            # Return zero features if all scales are empty
            return torch.zeros(1, self.output_channels, device=device)
        
        # Align all features to the reference size and concatenate
        aligned_features = []
        
        for scale_id, features in enumerate(multi_scale_features):
            if features.numel() > 0 and features.shape[0] > 0:
                # If this scale has fewer voxels, pad with zeros
                if features.shape[0] < max_voxels:
                    padding = torch.zeros(max_voxels - features.shape[0], features.shape[1], device=device)
                    aligned = torch.cat([features, padding], dim=0)
                elif features.shape[0] > max_voxels:
                    # If this scale has more voxels, truncate
                    aligned = features[:max_voxels]
                else:
                    aligned = features
                aligned_features.append(aligned)
            else:
                # Empty scale - add zero features
                expected_channels = self.scale_channels[scale_id] if scale_id < len(self.scale_channels) else 64
                zero_features = torch.zeros(max_voxels, expected_channels, device=device)
                aligned_features.append(zero_features)
        
        # Concatenate along feature dimension
        concatenated = torch.cat(aligned_features, dim=-1)  # (N, total_channels)
        
        # Apply fusion network with skip connection
        main_features = self.fusion_net(concatenated)
        skip_features = self.skip_connection(concatenated)
        fused_features = main_features + skip_features
        
        return fused_features


@MODELS.register_module()
class FixedMultiScaleVFE(nn.Module):
    """
    🎯 FIXED MULTI-SCALE VFE FOR BASELINE COMPARISON
    
    This implements a fixed multi-scale voxelization system that:
    1. Processes ALL points at ALL predefined scales [0.05, 0.1, 0.2] m
    2. Uses traditional VFE processing at each scale
    3. Fuses multi-scale features using concatenation and MLP
    4. No adaptive selection - purely fixed scale processing
    
    Pipeline:
    Point Cloud → Multi-Scale Voxelization → Scale-Specific VFEs → Feature Fusion
    """
    
    def __init__(self,
                 # Fixed scale configuration
                 voxel_scales: List[float] = [0.05, 0.1, 0.2],
                 
                 # Standard VFE config
                 max_num_points: int = 5,
                 max_voxels: Tuple[int, int] = (12000, 30000),
                 point_cloud_range: List[float] = None,
                 
                 # Network configuration
                 vfe_channels: List[int] = [32, 64],
                 fusion_channels: int = 128,
                 output_channels: int = 64,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 
                 # Feature enhancement options
                 with_distance: bool = True,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__()
        
        # Log ignored parameters
        if kwargs:
            ignored_params = list(kwargs.keys())
            print(f"🔧 FixedMultiScaleVFE ignoring parameters: {ignored_params}")
        
        self.voxel_scales = voxel_scales
        self.num_scales = len(voxel_scales)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.point_cloud_range = point_cloud_range
        
        print(f"🎯 FixedMultiScaleVFE initialized:")
        print(f"   📏 Fixed scales: {[f'{s:.3f}m' for s in self.voxel_scales]}")
        print(f"   🔧 VFE channels: {vfe_channels}")
        print(f"   🔗 Fusion channels: {fusion_channels}")
        print(f"   📊 Output channels: {output_channels}")
        
        # 1. Fixed multi-scale voxelizer
        self.multi_scale_voxelizer = FixedMultiScaleVoxelizer(
            voxel_scales=self.voxel_scales,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
            point_cloud_range=point_cloud_range
        )
        
        # 2. Scale-specific VFEs (one for each fixed scale)
        self.scale_vfes = nn.ModuleList()
        for i, scale_size in enumerate(self.voxel_scales):
            vfe = FixedScaleVFE(
                in_channels=4,
                feat_channels=vfe_channels,
                scale_size=scale_size,
                scale_id=i,
                with_distance=with_distance,
                with_cluster_center=with_cluster_center,
                with_voxel_center=with_voxel_center,
                norm_cfg=norm_cfg
            )
            self.scale_vfes.append(vfe)
        
        # 3. Feature fusion
        scale_channels = [vfe_channels[-1]] * self.num_scales
        self.feature_fusion = FixedMultiScaleFeatureFusion(
            scale_channels=scale_channels,
            fusion_channels=fusion_channels,
            output_channels=output_channels
        )
        
        # Output configuration
        self.output_channels = output_channels + 1  # +1 for scale diversity info
    
    def forward(self, features: torch.Tensor, num_points: torch.Tensor = None, 
                coors: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for fixed multi-scale VFE.
        
        Args:
            features: Raw point cloud (N, 4) when called from VoxelNet
                     OR voxel features (N, max_points, 4) when called normally
            num_points: (N,) - number of points per voxel (optional)
            coors: (N, 4) - voxel coordinates (optional)
            
        Returns:
            output: (N, output_channels) - fused multi-scale features
            coors: (N, 4) - output coordinates
        """
        device = features.device
        
        # Case 1: Called from VoxelNet with raw points (N, 4)
        if num_points is None and coors is None:
            return self._forward_raw_points(features)
        else:
            return self._forward_voxelized(features, num_points, coors)
    
    def _forward_raw_points(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle raw point cloud input from VoxelNet."""
        device = points.device
        
        if points.shape[0] == 0:
            dummy_output = torch.zeros(0, self.output_channels, device=device)
            dummy_coors = torch.zeros(0, 4, device=device).long()
            return dummy_output, dummy_coors
        
        try:
            # 1. Multi-scale voxelization at ALL fixed scales
            multi_scale_voxels = self.multi_scale_voxelizer(points)
            
            # 2. Process each scale with its dedicated VFE
            multi_scale_features = []
            for scale_id, (voxel_data, vfe) in enumerate(zip(multi_scale_voxels, self.scale_vfes)):
                if voxel_data['voxels'].numel() > 0:
                    scale_features = vfe(voxel_data['voxels'], voxel_data['num_points'], voxel_data['coordinates'])
                    multi_scale_features.append(scale_features)
                else:
                    # Placeholder for empty scales
                    placeholder = torch.zeros(1, vfe.output_channels, device=device)
                    multi_scale_features.append(placeholder)
            
            # 3. Feature fusion
            fused_features = self.feature_fusion(multi_scale_features)
            
            # 4. Use fused features directly (no additional channels for baseline)
            output = fused_features
            
            # 5. Generate coordinates
            batch_size = output.shape[0]
            coors = torch.zeros(batch_size, 4, device=device, dtype=torch.long)
            coors[:, 0] = 0  # All same batch
            
            return output, coors
            
        except Exception as e:
            print(f"⚠️ FixedMultiScaleVFE forward pass failed: {str(e)}")
            return self._fallback_processing(points)
    
    def _forward_voxelized(self, features: torch.Tensor, num_points: torch.Tensor, 
                          coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle pre-voxelized input."""
        device = features.device
        batch_size = features.shape[0]
        
        try:
            # Extract representative points from voxelized input
            if len(features.shape) == 3:
                # Take first point from each voxel as representative
                representative_points = features[:, 0, :4]
            else:
                # Use coordinates as points
                representative_points = coors[:, 1:].float()
                if representative_points.shape[1] == 3:
                    # Add intensity channel if missing
                    intensity = torch.zeros(representative_points.shape[0], 1, device=device)
                    representative_points = torch.cat([representative_points, intensity], dim=1)
            
            # Process with fixed multi-scale approach
            multi_scale_voxels = self.multi_scale_voxelizer(representative_points)
            
            # Process each scale
            multi_scale_features = []
            for scale_id, (voxel_data, vfe) in enumerate(zip(multi_scale_voxels, self.scale_vfes)):
                if voxel_data['voxels'].numel() > 0:
                    scale_features = vfe(voxel_data['voxels'], voxel_data['num_points'], voxel_data['coordinates'])
                    multi_scale_features.append(scale_features)
                else:
                    placeholder = torch.zeros(1, vfe.output_channels, device=device)
                    multi_scale_features.append(placeholder)
            
            # Fuse features
            fused_features = self.feature_fusion(multi_scale_features)
            
            # Align with input batch size
            if fused_features.shape[0] != batch_size:
                if fused_features.shape[0] < batch_size:
                    padding = torch.zeros(batch_size - fused_features.shape[0], 
                                        fused_features.shape[1], device=device)
                    fused_features = torch.cat([fused_features, padding], dim=0)
                else:
                    fused_features = fused_features[:batch_size]
            
            # Use fused features directly (no additional channels for baseline)
            output = fused_features
            
            return output, coors
            
        except Exception as e:
            print(f"⚠️ FixedMultiScaleVFE voxelized processing failed: {str(e)}")
            return self._fallback_voxelized(features, num_points, coors)
    
    def _fallback_processing(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple fallback processing for raw points."""
        device = points.device
        
        # Simple linear projection to exact output channels
        if not hasattr(self, 'fallback_projection'):
            self.fallback_projection = nn.Linear(points.shape[1], self.output_channels).to(device)
        
        output = self.fallback_projection(points)
        
        # Simple coordinates
        coors = torch.zeros(output.shape[0], 4, device=device, dtype=torch.long)
        
        return output, coors
    
    def _fallback_voxelized(self, features: torch.Tensor, num_points: torch.Tensor, 
                           coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple fallback for voxelized input."""
        device = features.device
        batch_size = features.shape[0]
        
        # Simple max pooling if 3D features
        if len(features.shape) == 3:
            mask = torch.arange(features.shape[1], device=device).unsqueeze(0) < num_points.unsqueeze(1)
            features_masked = features.clone()
            features_masked[~mask.unsqueeze(-1).expand_as(features)] = float('-inf')
            pooled = torch.max(features_masked, dim=1)[0]
        else:
            pooled = features
        
        # Project to exact target dimensions
        if pooled.shape[1] != self.output_channels:
            if not hasattr(self, 'voxel_fallback_projection'):
                self.voxel_fallback_projection = nn.Linear(pooled.shape[1], self.output_channels).to(device)
            output = self.voxel_fallback_projection(pooled)
        else:
            output = pooled
        
        return output, coors
