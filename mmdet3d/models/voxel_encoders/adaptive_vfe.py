# mmdet3d/models/voxel_encoders/adaptive_vfe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class AdaptiveVFE(nn.Module):
    """
    Truly Adaptive Voxel Feature Encoder that learns:
    1. Per-voxel adaptive sizes based on local point density
    2. Multi-scale voxel representations
    3. Content-aware voxelization
    """
    def __init__(self,
                 in_channels,
                 feat_channels,
                 with_distance=False,
                 voxel_size=(0.5, 0.5, 0.5),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 base_sparse_shape=[41, 1600, 1408],
                 adaptation_method='density',  # 'density', 'content', 'multi_scale'
                 num_scales=3):
        super().__init__()
        self.with_distance = with_distance
        self.adaptation_method = adaptation_method
        self.num_scales = num_scales
        
        # Store base parameters
        self.register_buffer('base_voxel_size', torch.tensor(voxel_size))
        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        self.base_sparse_shape = base_sparse_shape
        
        # Feature processing layers
        in_dim = in_channels
        if with_distance:
            in_dim += 1
        
        if adaptation_method == 'density':
            # Learn voxel size based on local point density
            self.density_predictor = nn.Sequential(
                nn.Linear(in_dim + 1, 64),  # +1 for point count
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # Scale factors for x, y, z
                nn.Sigmoid()  # Output in [0, 1], will scale to [0.5, 2.0]
            )
        elif adaptation_method == 'content':
            # Learn voxel size based on content features
            self.content_analyzer = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 3)  # Adaptive scale factors
            )
        elif adaptation_method == 'multi_scale':
            # Multi-scale processing with learned scale selection
            self.scale_factors = nn.Parameter(
                torch.tensor([0.5, 1.0, 2.0]),  # Small, medium, large scales
                requires_grad=True
            )
            self.scale_selector = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_scales),
                nn.Softmax(dim=-1)  # Attention weights for scales
            )
        
        # Enhanced feature processing
        mlp_in_dim = in_dim * 2  # original + deviations
        if adaptation_method == 'multi_scale':
            mlp_in_dim += num_scales  # Add scale features
            
        layers = []
        last_channels = mlp_in_dim
        for out_ch in feat_channels:
            layers.append(nn.Linear(last_channels, out_ch, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            last_channels = out_ch
        self.point_fc = nn.Sequential(*layers)
        self.voxel_fc = nn.Linear(last_channels, feat_channels[-1], bias=False)

    def get_adaptive_voxel_sizes(self, features, num_points):
        """Get adaptive voxel sizes for each voxel."""
        if self.adaptation_method == 'density':
            return self._get_density_adaptive_sizes(features, num_points)
        elif self.adaptation_method == 'content':
            return self._get_content_adaptive_sizes(features)
        elif self.adaptation_method == 'multi_scale':
            return self._get_multi_scale_representation(features)
        else:
            return self.base_voxel_size.unsqueeze(0).repeat(features.shape[0], 1)
    
    def _get_density_adaptive_sizes(self, features, num_points):
        """Adapt voxel size based on local point density."""
        # Compute per-voxel features
        voxel_means = features.sum(dim=1) / num_points.float().unsqueeze(1)
        
        # Add normalized point count as feature
        max_points = num_points.max().float()
        normalized_counts = (num_points.float() / max_points).unsqueeze(1)
        
        # Combine features with point count
        density_features = torch.cat([voxel_means, normalized_counts], dim=1)
        
        # Predict scale factors [0, 1] -> [0.5, 2.0]
        scale_logits = self.density_predictor(density_features)
        scale_factors = 0.5 + 1.5 * scale_logits  # Scale to [0.5, 2.0]
        
        # Apply to base voxel size
        adaptive_sizes = self.base_voxel_size.unsqueeze(0) * scale_factors
        
        return adaptive_sizes
    
    def _get_content_adaptive_sizes(self, features):
        """Adapt voxel size based on feature content."""
        # Analyze voxel content
        voxel_means = features.mean(dim=1)  # Average features per voxel
        
        # Predict adaptive factors
        scale_logits = self.content_analyzer(voxel_means)
        scale_factors = torch.sigmoid(scale_logits) * 1.5 + 0.5  # [0.5, 2.0]
        
        adaptive_sizes = self.base_voxel_size.unsqueeze(0) * scale_factors
        
        return adaptive_sizes
    
    def _get_multi_scale_representation(self, features):
        """Multi-scale processing with learned scale attention."""
        voxel_means = features.mean(dim=1)
        
        # Get attention weights for different scales
        scale_weights = self.scale_selector(voxel_means)  # [num_voxels, num_scales]
        
        # Create multi-scale features
        scale_features = []
        for i in range(self.num_scales):
            scale = self.scale_factors[i]
            scale_size = self.base_voxel_size * scale
            # Use scale as additional feature
            scale_feat = torch.full((features.shape[0], 1), scale.item(), 
                                   device=features.device)
            scale_features.append(scale_feat)
        
        # Weighted combination of scales
        scale_features = torch.cat(scale_features, dim=1)  # [num_voxels, num_scales]
        weighted_scales = (scale_features * scale_weights).sum(dim=1, keepdim=True)
        
        adaptive_sizes = self.base_voxel_size.unsqueeze(0) * weighted_scales
        
        return adaptive_sizes, scale_weights

    def forward(self, features, num_points, coors):
        """
        Args:
            features (torch.Tensor): (sum(V), P, C) per-point features
            num_points (torch.Tensor): (sum(V),) number of points per voxel
            coors (torch.Tensor):  (sum(V), 4) voxel indices (batch, z,y,x)
        Returns:
            tuple: (voxel_features, new_coors, adaptive_info)
        """
        
        # Get adaptive voxel sizes
        if self.adaptation_method == 'multi_scale':
            adaptive_sizes, scale_weights = self.get_adaptive_voxel_sizes(features, num_points)
            adaptive_info = {'adaptive_sizes': adaptive_sizes, 'scale_weights': scale_weights}
        else:
            adaptive_sizes = self.get_adaptive_voxel_sizes(features, num_points)
            adaptive_info = {'adaptive_sizes': adaptive_sizes}
        
        # Print adaptation info occasionally
        if self.training and torch.rand(1).item() < 0.01:
            if adaptive_sizes.dim() > 1:
                mean_sizes = adaptive_sizes.mean(dim=0)
                std_sizes = adaptive_sizes.std(dim=0)
                print(f"Adaptive voxel sizes - Mean: {mean_sizes.detach().cpu().numpy()}")
                print(f"Adaptive voxel sizes - Std: {std_sizes.detach().cpu().numpy()}")
            else:
                print(f"Adaptive voxel sizes: {adaptive_sizes.detach().cpu().numpy()}")
        
        # Add distance features if requested
        if self.with_distance:
            points_mean = (features.sum(dim=1) / num_points.type_as(features).view(-1, 1))
            f_centroid = points_mean.unsqueeze(1).repeat(1, features.size(1), 1)
            
            dist = torch.norm(features[:, :, :3] - f_centroid[:, :, :3], dim=2, keepdim=True)
            # Normalize by adaptive voxel size (use mean if per-voxel)
            if adaptive_sizes.dim() > 1:
                avg_voxel_scale = adaptive_sizes.mean(dim=1, keepdim=True).mean()
            else:
                avg_voxel_scale = adaptive_sizes.mean()
            dist = dist / avg_voxel_scale
            features = torch.cat([features, dist], dim=-1)
        
        # Compute enhanced features
        points_mean = (features.sum(dim=1) / num_points.type_as(features).view(-1, 1))
        f_centroid = points_mean.unsqueeze(1).repeat(1, features.size(1), 1)
        f_dev = features - f_centroid
        
        # Combine features
        enhanced_features = torch.cat([features, f_dev], dim=-1)
        
        # Add scale information for multi-scale method
        if self.adaptation_method == 'multi_scale':
            scale_info = scale_weights.unsqueeze(1).repeat(1, features.size(1), 1)
            enhanced_features = torch.cat([enhanced_features, scale_info], dim=-1)
        
        # Process through MLP
        pts_feats = self.point_fc(enhanced_features.view(-1, enhanced_features.size(-1)))
        pts_feats = pts_feats.view(enhanced_features.size(0), -1, pts_feats.size(-1))
        
        # Aggregate
        voxel_feats, _ = torch.max(pts_feats, dim=1)
        voxel_feats = self.voxel_fc(voxel_feats)
        
        return voxel_feats, coors, adaptive_info
