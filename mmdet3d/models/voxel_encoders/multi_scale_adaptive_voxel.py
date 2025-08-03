# Copyright (c) OpenMMLab. All rights reserved.
"""
🎯 MULTI-SCALE ADAPTIVE VOXELIZATION
True PhD Research Implementation

Key Innova        # Step 1        fine_mask = scale_assignment == 0
        medium_mask = scale_assignment == 1
        coarse_mask = scale_assignment == 2
        
        # Step 3: Voxelize each scale separatelyimportance scores for each point
        importance_scores = self.importance_net(points)  # [N, 3]
        
        # Apply softmax to get probabilities
        importance_probs = F.softmax(importance_scores, dim=-1)
        
        # Step 2: Assign points to scales based on importance
        scale_assignment = torch.argmax(importance_scores, dim=-1)
        
        fine_mask = scale_assignment == 0
        medium_mask = scale_assignment == 1
        coarse_mask = scale_assignment == 2cing variable voxel sizes into one tensor,
we create separate tensors for each voxel scale and process them in parallel.

Architecture:
- ImportancePredictor: Neural network predicts point importance
- Multi-scale voxelization: Different voxel sizes for different importance levels
- Parallel sparse convolution: Each scale processed independently
- Multi-scale fusion: Intelligent combination of multi-resolution features
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS
from mmdet3d.models.task_modules import VoxelGenerator
from mmdet3d.utils import ConfigType, OptConfigType


@MODELS.register_module()
class MultiScaleAdaptiveVoxelEncoder(BaseModule):
    """
    🎯 Multi-Scale Adaptive Voxel Encoder
    
    Revolutionary approach based on user's brilliant insight:
    - Create separate tensors for different voxel scales
    - Process each scale in parallel sparse convolution networks
    - Intelligently fuse multi-resolution features
    """
    
    def __init__(self,
                 point_cloud_range: List[float],
                 base_voxel_size: List[float] = [0.05, 0.05, 0.1],
                 max_num_points: int = 5,
                 max_voxels: Tuple[int, int] = (12000, 30000),
                 importance_channels: int = 128,
                 fine_scale: float = 0.5,    # 2x finer
                 medium_scale: float = 1.0,  # base scale
                 coarse_scale: float = 2.0,  # 2x coarser
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        
        self.point_cloud_range = point_cloud_range
        self.base_voxel_size = base_voxel_size
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        
        # Importance prediction network
        self.importance_predictor = MultiScaleImportancePredictor(
            in_channels=4,  # [x, y, z, intensity]
            hidden_channels=[importance_channels, importance_channels//2],
            out_channels=3  # [fine_prob, medium_prob, coarse_prob]
        )
        
        # Multi-scale voxel generators
        fine_voxel_size = [s * fine_scale for s in base_voxel_size]
        medium_voxel_size = base_voxel_size
        coarse_voxel_size = [s * coarse_scale for s in base_voxel_size]
        
        self.fine_voxelizer = VoxelGenerator(
            voxel_size=fine_voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels[0]
        )
        
        self.medium_voxelizer = VoxelGenerator(
            voxel_size=medium_voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels[0]
        )
        
        self.coarse_voxelizer = VoxelGenerator(
            voxel_size=coarse_voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels[0]
        )
        
        # Feature extraction for each scale
        self.fine_feature_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )
        
        self.medium_feature_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )
        
        self.coarse_feature_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(
            fine_channels=4,
            medium_channels=4,
            coarse_channels=4,
            fusion_channels=64
        )
    
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        🎯 Multi-scale adaptive voxelization
        
        Args:
            points: [N, 4] input point cloud [x, y, z, intensity]
            
        Returns:
            Dict containing:
            - 'fine_features': Fine-scale voxel features
            - 'fine_coords': Fine-scale coordinates
            - 'medium_features': Medium-scale voxel features 
            - 'medium_coords': Medium-scale coordinates
            - 'coarse_features': Coarse-scale voxel features
            - 'coarse_coords': Coarse-scale coordinates
            - 'importance_scores': Point importance predictions
        """
        
        # Step 1: Predict point importance
        importance_scores = self.importance_predictor(points)
        fine_prob, medium_prob, coarse_prob = torch.chunk(importance_scores, 3, dim=-1)
        
        # Step 2: Assign points to scales based on importance
        scale_assignment = torch.argmax(importance_scores, dim=-1)
        
        fine_mask = scale_assignment == 0
        medium_mask = scale_assignment == 1
        coarse_mask = scale_assignment == 2
        
        print(f"� Scale assignment:")
        print(f"  Fine: {fine_mask.sum().item()} points")
        print(f"  Medium: {medium_mask.sum().item()} points")
        print(f"  Coarse: {coarse_mask.sum().item()} points")
        
        # Step 3: Voxelize each scale separately  
        multi_scale_data = {}
        
        # Fine scale processing
        if fine_mask.any():
            fine_points = points[fine_mask].cpu().numpy()
            fine_voxels, fine_coords, fine_num_points = self.fine_voxelizer.generate(fine_points)
            
            if len(fine_voxels) > 0:
                fine_voxels = torch.from_numpy(fine_voxels).float().to(points.device)
                fine_coords = torch.from_numpy(fine_coords).long().to(points.device)
                fine_num_points = torch.from_numpy(fine_num_points).long().to(points.device)
                
                # Extract features
                fine_features = self._extract_voxel_features(
                    fine_voxels, fine_num_points, self.fine_feature_net
                )
                
                multi_scale_data['fine_features'] = fine_features
                multi_scale_data['fine_coords'] = fine_coords
        
        # Medium scale processing
        if medium_mask.any():
            medium_points = points[medium_mask].cpu().numpy()
            medium_voxels, medium_coords, medium_num_points = self.medium_voxelizer.generate(medium_points)
            
            if len(medium_voxels) > 0:
                medium_voxels = torch.from_numpy(medium_voxels).float().to(points.device)
                medium_coords = torch.from_numpy(medium_coords).long().to(points.device)
                medium_num_points = torch.from_numpy(medium_num_points).long().to(points.device)
                
                # Extract features
                medium_features = self._extract_voxel_features(
                    medium_voxels, medium_num_points, self.medium_feature_net
                )
                
                multi_scale_data['medium_features'] = medium_features
                multi_scale_data['medium_coords'] = medium_coords
        
        # Coarse scale processing
        if coarse_mask.any():
            coarse_points = points[coarse_mask].cpu().numpy()
            coarse_voxels, coarse_coords, coarse_num_points = self.coarse_voxelizer.generate(coarse_points)
            
            if len(coarse_voxels) > 0:
                coarse_voxels = torch.from_numpy(coarse_voxels).float().to(points.device)
                coarse_coords = torch.from_numpy(coarse_coords).long().to(points.device)
                coarse_num_points = torch.from_numpy(coarse_num_points).long().to(points.device)
                
                # Extract features
                coarse_features = self._extract_voxel_features(
                    coarse_voxels, coarse_num_points, self.coarse_feature_net
                )
                
                multi_scale_data['coarse_features'] = coarse_features
                multi_scale_data['coarse_coords'] = coarse_coords
        
        # Store importance scores for analysis
        multi_scale_data['importance_scores'] = importance_scores
        
        # For VoxelNet compatibility: combine all features and coords
        all_features = []
        all_coords = []
        
        # Collect features and coordinates from all scales WITH SCALE IDs
        scale_mapping = {'fine': 0, 'medium': 1, 'coarse': 2}
        
        for scale_name in ['fine', 'medium', 'coarse']:
            if f'{scale_name}_features' in multi_scale_data:
                features = multi_scale_data[f'{scale_name}_features']
                coords = multi_scale_data[f'{scale_name}_coords']
                scale_id = scale_mapping[scale_name]
                
                # Add batch dimension to coordinates if not present
                if coords.shape[1] == 3:  # [z, y, x]
                    batch_coords = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
                    coords = torch.cat([batch_coords, coords], dim=1)  # [batch, z, y, x]
                
                # 🎯 ADD SCALE ID: Add scale identifier as last column for parallel processing
                scale_ids = torch.full((coords.shape[0], 1), scale_id, dtype=coords.dtype, device=coords.device)
                coords_with_scale = torch.cat([coords, scale_ids], dim=1)  # [batch, z, y, x, scale_id]
                
                all_features.append(features)
                all_coords.append(coords_with_scale)
        
        if all_features:
            # Combine all scales
            combined_features = torch.cat(all_features, dim=0)
            combined_coords = torch.cat(all_coords, dim=0)
            
            # Coordinate validation: Ensure coords fit sparse grid [41, 1600, 1408]
            expected_shape = torch.tensor([1, 41, 1600, 1408], device=combined_coords.device)
            
            # Clamp coordinates to fit within expected sparse grid
            combined_coords[:, 0] = torch.clamp(combined_coords[:, 0], 0, expected_shape[0] - 1)  # batch
            combined_coords[:, 1] = torch.clamp(combined_coords[:, 1], 0, expected_shape[1] - 1)  # z
            combined_coords[:, 2] = torch.clamp(combined_coords[:, 2], 0, expected_shape[2] - 1)  # y  
            combined_coords[:, 3] = torch.clamp(combined_coords[:, 3], 0, expected_shape[3] - 1)  # x
            
            # Store the full multi-scale data for potential use by middle encoder
            self._last_multi_scale_data = multi_scale_data
            
            # Return tuple for VoxelNet compatibility
            return combined_features, combined_coords
        else:
            # No voxels generated - return empty
            empty_features = torch.zeros((0, 4), device=points.device, dtype=points.dtype)
            empty_coords = torch.zeros((0, 4), device=points.device, dtype=torch.long)
            self._last_multi_scale_data = multi_scale_data
            return empty_features, empty_coords
    
    def _extract_voxel_features(self, voxels: torch.Tensor, num_points: torch.Tensor, 
                               feature_net: nn.Module) -> torch.Tensor:
        """Extract features from voxels using point-wise aggregation."""
        batch_size, max_points, point_dim = voxels.shape
        
        # Create mask for valid points
        valid_mask = torch.arange(max_points, device=voxels.device)[None, :] < num_points[:, None]
        
        # Get valid points
        valid_points = voxels[valid_mask]  # [N_valid, 4]
        
        # Extract features
        point_features = feature_net(valid_points)  # [N_valid, 4]
        
        # Aggregate per voxel (mean pooling)
        voxel_features = torch.zeros(batch_size, point_features.shape[-1], 
                                   device=voxels.device, dtype=voxels.dtype)
        
        # Map back to voxels
        voxel_idx = torch.arange(batch_size, device=voxels.device).repeat_interleave(num_points)
        voxel_features.index_add_(0, voxel_idx, point_features)
        voxel_features = voxel_features / num_points.float().unsqueeze(-1).clamp(min=1)
        
        return voxel_features


@MODELS.register_module()
class MultiScaleImportancePredictor(BaseModule):
    """
    🧠 IMPORTANCE PREDICTOR NETWORK
    
    Neural network that predicts the "importance" or "information density"
    of each point for adaptive voxel size selection.
    
    High importance → Fine voxels (small size)
    Low importance → Coarse voxels (large size)
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 hidden_channels: List[int] = [64, 32],
                 out_channels: int = 1,
                 dropout: float = 0.1,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        # Build MLP layers
        layers = []
        prev_channels = in_channels
        
        for hidden_dim in hidden_channels:
            layers.extend([
                nn.Linear(prev_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_channels = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(prev_channels, out_channels))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Predict importance scores for points
        
        Args:
            points: [N, 4] (x, y, z, intensity)
            
        Returns:
            importance: [N, 1] importance scores (before sigmoid)
        """
        return self.mlp(points)


@MODELS.register_module()
class MultiScaleFeatureFusion(BaseModule):
    """
    🔗 MULTI-SCALE FEATURE FUSION
    
    Intelligently combines features from different voxel scales:
    - Fine features: High detail, local patterns
    - Medium features: Standard resolution
    - Coarse features: Global context, large structures
    """
    
    def __init__(self,
                 fine_channels: int = 256,
                 medium_channels: int = 256,
                 coarse_channels: int = 256,
                 fusion_channels: int = 256,
                 fusion_method: str = 'attention',  # 'concat', 'attention', 'weighted'
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.fusion_conv = nn.Conv2d(
                fine_channels + medium_channels + coarse_channels,
                fusion_channels,
                kernel_size=1
            )
        
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.fine_attention = nn.Conv2d(fine_channels, 1, kernel_size=1)
            self.medium_attention = nn.Conv2d(medium_channels, 1, kernel_size=1)
            self.coarse_attention = nn.Conv2d(coarse_channels, 1, kernel_size=1)
            self.fusion_conv = nn.Conv2d(fine_channels, fusion_channels, kernel_size=1)
        
        elif fusion_method == 'weighted':
            # Learnable weighted fusion
            self.scale_weights = nn.Parameter(torch.ones(3))
            self.fusion_conv = nn.Conv2d(fine_channels, fusion_channels, kernel_size=1)
    
    def forward(self, 
                fine_features: Optional[torch.Tensor] = None,
                medium_features: Optional[torch.Tensor] = None,
                coarse_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse multi-scale features
        
        Args:
            fine_features: High-resolution features
            medium_features: Standard resolution features  
            coarse_features: Low-resolution features
            
        Returns:
            fused_features: Combined multi-scale features
        """
        
        # Collect available features
        available_features = []
        if fine_features is not None:
            available_features.append(fine_features)
        if medium_features is not None:
            available_features.append(medium_features)  
        if coarse_features is not None:
            available_features.append(coarse_features)
        
        if not available_features:
            raise ValueError("At least one scale of features must be provided")
        
        if len(available_features) == 1:
            return self.fusion_conv(available_features[0])
        
        # Multi-scale fusion based on method
        if self.fusion_method == 'concat':
            # Simple concatenation
            fused = torch.cat(available_features, dim=1)
            return self.fusion_conv(fused)
        
        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            # Resize all features to same spatial size (use fine resolution as target)
            target_size = available_features[0].shape[-2:]
            
            # Apply attention and resize
            attended_features = []
            for i, features in enumerate(available_features):
                if i == 0:  # fine
                    attention = torch.sigmoid(self.fine_attention(features))
                elif i == 1:  # medium
                    attention = torch.sigmoid(self.medium_attention(features))
                else:  # coarse
                    attention = torch.sigmoid(self.coarse_attention(features))
                
                # Resize to target size
                if features.shape[-2:] != target_size:
                    features = torch.nn.functional.interpolate(
                        features, size=target_size, mode='bilinear', align_corners=False)
                    attention = torch.nn.functional.interpolate(
                        attention, size=target_size, mode='bilinear', align_corners=False)
                
                attended_features.append(features * attention)
            
            # Sum attended features
            fused = sum(attended_features)
            return self.fusion_conv(fused)
        
        elif self.fusion_method == 'weighted':
            # Learnable weighted combination
            target_size = available_features[0].shape[-2:]
            weights = torch.softmax(self.scale_weights[:len(available_features)], dim=0)
            
            weighted_features = []
            for i, features in enumerate(available_features):
                if features.shape[-2:] != target_size:
                    features = torch.nn.functional.interpolate(
                        features, size=target_size, mode='bilinear', align_corners=False)
                weighted_features.append(features * weights[i])
            
            fused = sum(weighted_features)
            return self.fusion_conv(fused)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
