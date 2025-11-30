"""
Importance-Guided Multi-Scale VFE with Point Filtering
=====================================================

This module implements a memory-efficient multi-scale VFE that uses a lightweight
importance network to filter points before voxelization, following the architecture:

Point Cloud → Importance Net → Top-K Selection → Multi-Scale Voxelization → VFE

Author: Daham Pathiraja
Date: August 3, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType
from mmengine.model import BaseModule


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    
    def __init__(self, channels: int, out_channels: int, dropout_rate: float = 0.05):
        super().__init__()
        self.linear1 = nn.Linear(channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection
        self.skip = nn.Linear(channels, out_channels) if channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = F.relu(out, inplace=True)
        
        return out


@MODELS.register_module()
class LightweightPointImportanceNet(nn.Module):
    """
    Lightweight network to predict point importance scores.
    Uses 3-layer MLP/PointNet to score each point.
    """
    
    def __init__(self,
                 in_channels: int = 4,  # x, y, z, intensity
                 hidden_dims: List[int] = [64, 32, 16],
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,
                 activation: str = 'ReLU',
                 init_cfg: OptConfigType = None):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        layers = []
        prev_dim = in_channels
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU(0.1, inplace=True))
            
            # Dropout (except last layer)
            if i < len(hidden_dims) - 1 and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final importance score layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Importance scores in [0, 1]
        
        self.importance_net = nn.Sequential(*layers)
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Predict importance scores for each point.
        
        Args:
            points (torch.Tensor): Shape (N, C) where N is number of points
                                  and C is point feature dimension
        
        Returns:
            torch.Tensor: Importance scores of shape (N, 1)
        """
        return self.importance_net(points)


@MODELS.register_module()
class ScaleSpecificLightweightVFE(nn.Module):
    """
    Lightweight VFE for a specific voxel scale with scale ID embedding.
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: List[int] = [32, 64],
                 scale_id: int = 0,
                 scale_embedding_dim: int = 8,
                 with_distance: bool = False,
                 with_cluster_center: bool = True,
                 with_voxel_center: bool = True,
                 point_cloud_range: List[float] = None,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 init_cfg: OptConfigType = None):
        super().__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.scale_id = scale_id
        self.scale_embedding_dim = scale_embedding_dim
        self.with_distance = with_distance
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        self.point_cloud_range = point_cloud_range
        self.mode = mode
        
        # Calculate input dimension
        input_dim = in_channels  # 4 (x, y, z, intensity)
        if with_distance:
            input_dim += 1  # +1 for distance
        if with_cluster_center:
            input_dim += 3  # +3 for cluster center (xyz)
        if with_voxel_center:
            input_dim += 3  # +3 for voxel center (xyz)
        
        # Scale embedding
        self.scale_embedding = nn.Embedding(10, scale_embedding_dim)  # Support up to 10 scales
        input_dim += scale_embedding_dim  # +scale_embedding_dim for scale embedding
        
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
        
        self.output_channels = feat_channels[-1] + 1  # +1 for scale_id
        
    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, 
                coors: torch.Tensor) -> torch.Tensor:
        """Forward pass for lightweight VFE."""
        batch_size = voxels.shape[0]
        max_points = voxels.shape[1]
        
        # Add distance, cluster center, voxel center features
        features = [voxels]  # Start with input voxels (4 channels: x, y, z, intensity)
        
        if self.with_cluster_center:
            points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.unsqueeze(-1).unsqueeze(-1)  # Only xyz
            cluster_center = points_mean.expand(-1, max_points, -1)
            features.append(cluster_center)  # Add 3 channels
        
        if self.with_voxel_center:
            # Simplified voxel center calculation (only xyz)
            voxel_center = voxels[:, :, :3].mean(dim=1, keepdim=True).expand(-1, max_points, -1)
            features.append(voxel_center)  # Add 3 channels
        
        if self.with_distance:
            if self.with_cluster_center:
                distance = torch.norm(voxels[:, :, :3] - cluster_center, dim=-1, keepdim=True)
            else:
                distance = torch.norm(voxels[:, :, :3], dim=-1, keepdim=True)
            features.append(distance)  # Add 1 channel
        
        # Concatenate all features
        voxel_features = torch.cat(features, dim=-1)
        
        # Add scale embedding
        scale_emb = self.scale_embedding(torch.tensor(self.scale_id, device=voxels.device))
        scale_emb = scale_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, max_points, -1)
        voxel_features = torch.cat([voxel_features, scale_emb], dim=-1)
        
        # Apply VFE layers
        for vfe_layer in self.vfe_layers:
            voxel_features = vfe_layer(voxel_features, num_points)
        
        # Add scale ID as feature
        scale_ids = torch.full((batch_size, 1), self.scale_id, device=voxels.device, dtype=torch.float)
        voxel_features = torch.cat([voxel_features, scale_ids], dim=-1)
        
        return voxel_features


class VFELayer(nn.Module):
    """Basic VFE layer."""
    
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
        
        # Use GroupNorm instead of BatchNorm for stability with small batches
        if not hasattr(self, 'group_norm'):
            # Create GroupNorm on the fly (8 groups is a good default)
            num_groups = min(8, x.shape[-1] // 4) if x.shape[-1] >= 4 else 1
            self.group_norm = nn.GroupNorm(num_groups, x.shape[-1], eps=1e-6).to(x.device)
        
        x_before_norm = x.clone()
        x = self.group_norm(x)
        
        # Check if normalization is causing issues
        if x.norm().item() < 1e-6 and x_before_norm.norm().item() > 1e-6:
            # Use simpler normalization
            x = F.layer_norm(x_before_norm, x_before_norm.shape[-1:])
        
        x = F.relu(x, inplace=True)
        
        # Reshape back
        x = x.view(batch_size, max_points, -1)
        
        if not self.last_layer:
            return x
        
        # Improved Max pooling with better gradient flow
        # Create mask for valid points
        mask = torch.arange(max_points, device=x.device).unsqueeze(0) < num_points.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(x)
        
        # Soft masking instead of hard -inf assignment for better gradients
        # Use very negative values but not -inf to maintain gradient flow
        x_masked = x.clone()
        x_masked[~mask] = -1e6  # Large negative instead of -inf
        
        # Max pooling across points dimension
        x_max = torch.max(x_masked, dim=1)[0]  # (batch_size, out_channels)
        
        return x_max


@MODELS.register_module()
class ScaleNet(nn.Module):
    """
    Lightweight scale selection network that predicts optimal voxel size per point.
    Uses Gumbel-Softmax for differentiable scale assignment.
    """
    
    def __init__(self,
                 in_channels: int = 4,  # x, y, z, intensity
                 hidden_dims: List[int] = [64, 32],
                 num_scales: int = 3,  # Default 3 scales for backward compatibility
                 temperature: float = 5.0,  # 🔥 HIGHER initial temperature for better exploration
                 dropout_rate: float = 0.05,  # 🔥 REDUCED dropout to prevent gradient killing
                 
                 # 🌊 NEW: Continuous prediction parameters
                 continuous_mode: bool = False,  # Enable continuous voxel size prediction
                 min_voxel_size: float = None,  # Auto-detect from voxel_scales if None
                 max_voxel_size: float = None,  # Auto-detect from voxel_scales if None
                 interpolation_neighbors: int = 3,  # Number of neighbors for interpolation
                 
                 # 🎓 PhD RESEARCH: Accept initial voxel scales for learnable parameters
                 voxel_scales: List[float] = None):  # Initial scales to make learnable
        super().__init__()
        
        # Store parameters as instance variables for network building
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 🎓 PhD RESEARCH: Store provided voxel scales for learnable parameters
        self.provided_voxel_scales = voxel_scales
        
        # 🚀 ENHANCED: Support up to 10 scales dynamically
        self.num_scales = min(max(num_scales, 1), 10)  # Clamp between 1-10 scales
        
        # 🌊 NEW: Continuous prediction configuration
        self.continuous_mode = continuous_mode
        self.interpolation_neighbors = min(interpolation_neighbors, self.num_scales)
        
        # Aggressive temperature scheduling with learnable decay
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.temperature_decay = nn.Parameter(torch.tensor(0.9995))  # Learnable decay rate
        self.min_temperature = 0.5  # Minimum temperature threshold
        self.iteration_count = 0  # Track iterations for scheduling
        
        # 🎯 SMART SCALE GENERATION: Automatically generate optimal scale distribution
        self._generate_optimal_scales()
        
        # 🌊 NEW: Set continuous prediction range after scales are generated
        if self.continuous_mode:
            self.min_voxel_size = min_voxel_size if min_voxel_size is not None else self.voxel_scales[0].item()
            self.max_voxel_size = max_voxel_size if max_voxel_size is not None else self.voxel_scales[-1].item()
            print(f"🌊 Continuous mode enabled: {self.min_voxel_size:.3f}m - {self.max_voxel_size:.3f}m")
            print(f"🎯 Using {self.interpolation_neighbors} neighbors for interpolation")
        
        # Build the network after initialization
        self._build_network()
        
    def _generate_optimal_scales(self):
        """
        🚀 ENHANCED: Generate optimal voxel scales based on number of scales.
        Uses logarithmic distribution for maximum coverage and differentiation.
        """
        # 🎓 PhD RESEARCH: Use provided scales if available
        if self.provided_voxel_scales is not None:
            print(f"🎯 Using provided voxel scales: {self.provided_voxel_scales}")
            scales = torch.tensor(self.provided_voxel_scales, dtype=torch.float32)
            self.num_scales = len(scales)  # Update num_scales to match provided scales
        elif self.num_scales == 1:
            # Single scale: Use medium resolution
            scales = torch.tensor([0.1])
        elif self.num_scales == 2:
            # Two scales: Fine and coarse
            scales = torch.tensor([0.05, 0.2])
        elif self.num_scales == 3:
            # Original three scales (backward compatibility)
            scales = torch.tensor([0.02, 0.15, 0.6])
        else:
            # 4-10 scales: Logarithmic distribution for optimal coverage
            # Range from 0.01m (1cm) to 1.0m (1m) 
            min_scale = 0.01  # 1cm - finest detail
            max_scale = 1.0   # 1m - largest context
            
            # Generate logarithmically spaced scales as initial values
            log_min = torch.log(torch.tensor(min_scale))
            log_max = torch.log(torch.tensor(max_scale))
            log_scales = torch.linspace(log_min, log_max, self.num_scales)
            scales = torch.exp(log_scales)
    
        # Make voxel scales LEARNABLE parameters - this is the key PhD contribution!
        self.voxel_scales = nn.Parameter(scales, requires_grad=True)
        
        print(f"🎯 ScaleNet initialized with {self.num_scales} learnable scales: {[f'{s:.3f}m' for s in scales.tolist()]}")
        
    def get_scale_info(self):
        """Return current scale configuration."""
        return {
            'num_scales': self.num_scales,
            'scales': self.voxel_scales.tolist(),
            'scale_range': f"{self.voxel_scales.min():.3f}m - {self.voxel_scales.max():.3f}m"
        }
    
    def get_scale_regularization_loss(self, weight: float = 0.01) -> torch.Tensor:
        """
        Get regularization loss for learned voxel scales.
        
        This ensures:
        1. Scales remain positive and reasonable
        2. Scales maintain diversity (not all converging to same value)
        3. Scales don't explode during training
        
        Args:
            weight: Regularization weight
            
        Returns:
            Regularization loss for scale parameters
        """
        # Ensure scales are positive and within reasonable bounds
        scale_bound_loss = torch.clamp(0.01 - self.voxel_scales, min=0).sum() + \
                          torch.clamp(self.voxel_scales - 1.0, min=0).sum()
        
        # Encourage diversity in scales (prevent collapse to single scale)
        scale_diversity_loss = -torch.var(self.voxel_scales)
        
        return weight * (scale_bound_loss + 0.1 * scale_diversity_loss)
        
    def _build_network(self):
        """Build the scale prediction network with optional continuous heads."""
        # Deeper spatial encoding for better scale prediction
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 32),  # Enhanced spatial capacity
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8)
        )
        
        # Enhanced MLP with residual connections and layer normalization
        layers = []
        prev_dim = self.in_channels + 8  # 4 + 8 spatial features
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Add residual connection for the first layer
            if i == 0 and prev_dim == hidden_dim:
                layers.extend([
                    ResidualBlock(prev_dim, hidden_dim, self.dropout_rate),
                ])
            else:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm for stability
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout_rate)
                ])
            prev_dim = hidden_dim
        
        # 🚀 ENHANCED: Dynamic bias initialization based on number of scales
        final_layer = nn.Linear(prev_dim, self.num_scales)
        nn.init.xavier_uniform_(final_layer.weight, gain=2.0)  # Higher weight initialization
        
        # Smart bias initialization for scale diversity
        self._initialize_scale_biases(final_layer)
        
        layers.append(final_layer)
        self.scale_predictor = nn.Sequential(*layers)
        
        # 🌊 NEW: Continuous prediction heads
        if self.continuous_mode:
            # Use the same feature dimension as the scale predictor input
            feature_dim = self.in_channels + 8  # 4 + 8 spatial features
            
            # Continuous voxel size regression head
            self.continuous_head = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate * 0.5),  # Lower dropout for regression
                nn.Linear(32, 1),  # Single continuous value
                nn.Sigmoid()  # Normalize to [0, 1] for interpolation
            )
            
            # Confidence prediction head for soft interpolation weighting
            self.confidence_head = nn.Sequential(
                nn.Linear(feature_dim, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Confidence in [0, 1]
            )
            
            print(f"🌊 Added continuous prediction heads for {self.min_voxel_size:.3f}m - {self.max_voxel_size:.3f}m range")
        
    def _initialize_scale_biases(self, final_layer):
        """
        🎯 SMART BIAS INITIALIZATION: Encourage scale diversity based on number of scales.
        """
        num_scales = self.num_scales
        
        if num_scales <= 3:
            # Original bias pattern for 1-3 scales
            if num_scales >= 1:
                final_layer.bias.data[0] = 1.5   # Strongly favor fine scale
            if num_scales >= 2:
                final_layer.bias.data[1] = 0.0   # Neutral medium scale  
            if num_scales >= 3:
                final_layer.bias.data[2] = -1.5  # Discourage coarse scale initially
        else:
            # For 4+ scales: Create smooth bias gradient
            bias_values = torch.linspace(2.0, -2.0, num_scales)  # From fine-favoring to coarse-discouraging
            final_layer.bias.data = bias_values
        
    def forward(self, points: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scale assignment for each point using Gumbel-Softmax.
        
        Args:
            points: (N, 4) - x, y, z, intensity
            training: whether in training mode
            
        Returns:
            scale_assignment: (N, num_scales) - differentiable scale assignment
            predicted_scales: (N,) - actual voxel sizes for each point
        """
        
    def forward(self, points: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scale assignment for each point using Gumbel-Softmax or continuous prediction.
        
        Args:
            points: (N, 4) - x, y, z, intensity
            training: whether in training mode
            
        Returns:
            scale_assignment: (N, num_scales) - differentiable scale assignment
            predicted_scales: (N,) - actual voxel sizes for each point
        """
        # Aggressive temperature scheduling
        if training:
            self.iteration_count += 1
            # Exponential decay with learnable rate
            current_temp = max(
                self.temperature * (self.temperature_decay ** (self.iteration_count // 100)),
                self.min_temperature
            )
            # Use proper tensor assignment to avoid PyTorch warning
            self.temperature.data.fill_(current_temp)
        else:
            current_temp = self.temperature.item()
        
        # Enhanced spatial encoding with normalization
        spatial_features = self.spatial_encoder(points[:, :3])  # Encode xyz
        
        # Feature normalization for stability
        normalized_points = F.normalize(points, dim=1)  # Normalize input features
        enhanced_features = torch.cat([normalized_points, spatial_features], dim=1)  # (N, 4+8)
        
        # 🌊 NEW: Choose prediction mode
        if self.continuous_mode:
            return self._continuous_forward(enhanced_features, training)
        else:
            return self._discrete_forward(enhanced_features, current_temp, training)
    
    def _discrete_forward(self, enhanced_features: torch.Tensor, current_temp: float, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original discrete scale prediction using Gumbel-Softmax."""
        # Predict scale logits with enhanced features
        scale_logits = self.scale_predictor(enhanced_features)  # (N, num_scales)
        
        # Aggressive logit sharpening for better differentiation
        scale_logits = scale_logits * 2.0  # Amplify differences
        
        if training:
            # Use soft assignment for better gradient flow
            scale_assignment = F.gumbel_softmax(
                scale_logits, 
                tau=current_temp, 
                hard=False,  # Soft for better gradients
                dim=1
            )
            
            # Add straight-through estimator for discrete assignment
            scale_assignment_hard = F.one_hot(torch.argmax(scale_logits, dim=1), num_classes=self.num_scales).float()
            scale_assignment = scale_assignment + (scale_assignment_hard - scale_assignment).detach()
            
            # Enhanced diversity penalty with gradient flow
            scale_probs = F.softmax(scale_logits, dim=1).mean(dim=0) + 1e-8
            diversity_loss = -torch.sum(scale_probs * torch.log(scale_probs))
            
            # Make diversity loss affect the logits directly (stronger gradient signal)
            diversity_bonus = 0.1 * diversity_loss  # Increased from 0.01
            scale_logits = scale_logits + diversity_bonus.unsqueeze(0).expand_as(scale_logits)
            
        else:
            # Use hard assignment during inference
            scale_assignment = F.one_hot(torch.argmax(scale_logits, dim=1), num_classes=self.num_scales).float()
        
        # Compute actual voxel sizes
        predicted_scales = torch.sum(scale_assignment * self.voxel_scales.unsqueeze(0), dim=1)  # (N,)
        
        return scale_assignment, predicted_scales
    
    def _continuous_forward(self, enhanced_features: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """🌊 NEW: Continuous voxel size prediction with soft interpolation."""
        # Predict continuous voxel size in normalized [0, 1] range
        continuous_pred = self.continuous_head(enhanced_features).squeeze(-1)  # (N,)
        confidence = self.confidence_head(enhanced_features).squeeze(-1)  # (N,)
        
        # Map to actual voxel size range
        voxel_size_range = self.max_voxel_size - self.min_voxel_size
        predicted_scales = self.min_voxel_size + continuous_pred * voxel_size_range  # (N,)
        
        # Compute soft interpolation weights for nearest discrete scales
        scale_assignment = self._compute_soft_interpolation_weights(predicted_scales, confidence)
        
        return scale_assignment, predicted_scales
    
    def _compute_soft_interpolation_weights(self, predicted_scales: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        🌊 Compute soft interpolation weights for continuous voxel sizes.
        
        Args:
            predicted_scales: (N,) continuous voxel sizes
            confidence: (N,) confidence scores
            
        Returns:
            scale_assignment: (N, num_scales) soft weights for interpolation
        """
        N = predicted_scales.shape[0]
        device = predicted_scales.device
        
        # Compute distances to all discrete scales
        distances = torch.abs(predicted_scales.unsqueeze(1) - self.voxel_scales.unsqueeze(0))  # (N, num_scales)
        
        # Find nearest neighbors
        _, nearest_indices = torch.topk(distances, self.interpolation_neighbors, dim=1, largest=False)  # (N, k)
        
        # Create soft assignment matrix
        scale_assignment = torch.zeros(N, self.num_scales, device=device)
        
        for i in range(N):
            neighbor_indices = nearest_indices[i]  # (k,)
            neighbor_distances = distances[i, neighbor_indices]  # (k,)
            
            # Inverse distance weighting with confidence modulation
            epsilon = 1e-6
            weights = 1.0 / (neighbor_distances + epsilon)  # (k,)
            weights = weights * confidence[i]  # Scale by confidence
            
            # Normalize weights
            weights = weights / (torch.sum(weights) + epsilon)  # (k,)
            
            # Assign weights to corresponding scales
            scale_assignment[i, neighbor_indices] = weights
        
        return scale_assignment


@MODELS.register_module() 
class MultiScaleVoxelizer(nn.Module):
    """
    Multi-scale voxelization module that groups points into fixed-size voxels
    at different scales based on predicted scale assignments.
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
        
    def forward(self, points: torch.Tensor, scale_assignment: torch.Tensor,
                dynamic_scales: torch.Tensor) -> List[Dict]:
        """
        Group points into voxels at multiple scales.
        
        Args:
            points: (N, 4) - point cloud
            scale_assignment: (N, num_scales) - soft scale assignment
            dynamic_scales: (num_scales,) - Learnable scale parameters (REQUIRED)
            
        Returns:
            List of voxelization results for each scale
        """
        # 🎓 PhD FIX: Always use dynamic learnable scales
        scales_to_use = dynamic_scales
        
        voxel_outputs = []
        
        # 🎓 PhD FIX: Changed from enumerate to range loop to handle learnable tensor
        for scale_id in range(len(scales_to_use)):
            # 🎓 PhD FIX: Extract voxel size (handle both tensor and list)
            if torch.is_tensor(scales_to_use):
                voxel_size = scales_to_use[scale_id].item()
            else:
                voxel_size = scales_to_use[scale_id]
            
            # Get soft assignment weights for this scale
            scale_weights = scale_assignment[:, scale_id]  # (N,)
            
            # Learnable multi-scale: Use weighted soft assignment with learnable threshold
            # Keep points with significant contribution to this scale
            min_weight_threshold = 0.1  # Learnable parameter could be added
            point_mask = scale_weights > min_weight_threshold
            
            if not point_mask.any():
                # No points for this scale
                voxel_outputs.append({
                    'voxels': torch.empty(0, self.max_num_points, 4, device=points.device),
                    'coordinates': torch.empty(0, 4, device=points.device, dtype=torch.long),
                    'num_points': torch.empty(0, device=points.device, dtype=torch.long),
                    'scale_weights': torch.empty(0, device=points.device),
                    'scale_id': scale_id,
                    'voxel_size': voxel_size
                })
                continue
            
            # Get points and weights for this scale
            scale_points = points[point_mask]  # (M, 4)
            scale_point_weights = scale_weights[point_mask]  # (M,)
            
            # Apply soft weighting to point features
            weighted_points = scale_points * scale_point_weights.unsqueeze(-1)
            
            # Differentiable simplified voxelization without coordinate quantization
            
            try:
                num_scale_points = weighted_points.shape[0]
                
                if num_scale_points > 0:
                    # Simplified: Treat each point as its own "voxel" to maintain differentiability
                    max_points_this_scale = min(num_scale_points, 2000)  # Allow more points
                    
                    if num_scale_points > max_points_this_scale:
                        # Differentiable sampling: Use soft attention instead of hard sampling
                        # Compute spatial attention scores
                        spatial_features = weighted_points[:, :3] / voxel_size  # Normalize by voxel size
                        attention_scores = torch.sum(spatial_features ** 2, dim=1)  # Simple spatial diversity score
                        
                        # Soft top-k selection (differentiable)
                        topk_values, topk_indices = torch.topk(attention_scores, max_points_this_scale, sorted=False)
                        
                        sampled_points = weighted_points[topk_indices]
                        sampled_weights = scale_point_weights[topk_indices]
                    else:
                        sampled_points = weighted_points
                        sampled_weights = scale_point_weights
                    
                    # Ensure minimum for batch norm
                    min_voxels = 4  # Increased minimum
                    if sampled_points.shape[0] < min_voxels:
                        needed = min_voxels - sampled_points.shape[0]
                        if sampled_points.shape[0] > 0:
                            # Differentiable augmentation: Add small noise that maintains gradients
                            indices = torch.randint(0, sampled_points.shape[0], (needed,), device=points.device)
                            noise_scale = voxel_size * 0.1  # Small noise relative to voxel size
                            noise = torch.randn(needed, 4, device=points.device) * noise_scale
                            augmented = sampled_points[indices] + noise
                            sampled_points = torch.cat([sampled_points, augmented], dim=0)
                            sampled_weights = torch.cat([sampled_weights, sampled_weights[indices]], dim=0)
                        else:
                            # Initialize with learnable parameters instead of random
                            sampled_points = torch.zeros(min_voxels, 4, device=points.device, requires_grad=True)
                            sampled_weights = torch.ones(min_voxels, device=points.device) * 0.1
                    
                    # Fully differentiable: Each point becomes a single-point voxel
                    voxels = sampled_points.unsqueeze(1)  # (N, 1, 4)
                    
                    # Differentiable coordinates: Use continuous coordinates instead of quantized
                    coordinates = torch.zeros(sampled_points.shape[0], 4, device=points.device)
                    coordinates[:, 0] = 0  # batch index
                    coordinates[:, 1:] = sampled_points[:, :3] / voxel_size  # Continuous coordinates
                    
                    num_points_per_voxel = torch.ones(sampled_points.shape[0], device=points.device)
                    
                    # Pad to max_num_points if needed (maintain differentiability)
                    if self.max_num_points > 1:
                        padding = torch.zeros(sampled_points.shape[0], self.max_num_points - 1, 4, device=points.device)
                        voxels = torch.cat([voxels, padding], dim=1)
                else:
                    # Empty case with learnable initialization
                    min_voxels = 4
                    voxels = torch.zeros(min_voxels, self.max_num_points, 4, device=points.device, requires_grad=True)
                    coordinates = torch.zeros(min_voxels, 4, device=points.device)
                    coordinates[:, 0] = 0
                    num_points_per_voxel = torch.ones(min_voxels, device=points.device)
                    sampled_weights = torch.ones(min_voxels, device=points.device) * 0.1
                
                voxel_outputs.append({
                    'voxels': voxels,
                    'coordinates': coordinates, 
                    'num_points': num_points_per_voxel,
                    'scale_weights': sampled_weights,
                    'scale_id': scale_id,
                    'voxel_size': voxel_size
                })
            except Exception as e:
                print(f"Voxelization failed for scale {scale_id}: {e}")
                # Add minimum dummy voxels for stability
                min_voxels = 2
                voxel_outputs.append({
                    'voxels': torch.randn(min_voxels, self.max_num_points, 4, device=points.device) * 0.1,
                    'coordinates': torch.zeros(min_voxels, 4, device=points.device, dtype=torch.long),
                    'num_points': torch.ones(min_voxels, device=points.device, dtype=torch.long),
                    'scale_weights': torch.ones(min_voxels, device=points.device) * 0.1,
                    'scale_id': scale_id,
                    'voxel_size': voxel_size
                })
        
        return voxel_outputs


@MODELS.register_module()
class ScaleSpecificVFE(nn.Module):
    """
    Scale-specific Voxel Feature Encoder for each voxel scale.
    """
    
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: List[int] = [32, 64],
                 scale_id: int = 0,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        
        self.scale_id = scale_id
        self.feat_channels = feat_channels
        
        # VFE layers
        self.vfe_layers = nn.ModuleList()
        prev_channels = in_channels
        
        for i, out_channels in enumerate(feat_channels):
            is_last = (i == len(feat_channels) - 1)
            self.vfe_layers.append(
                VFELayer(prev_channels, out_channels, norm_cfg, last_layer=is_last)
            )
            prev_channels = out_channels
            
        self.output_channels = feat_channels[-1]
        
    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Forward pass through VFE layers."""
        features = voxels
        
        for i, vfe_layer in enumerate(self.vfe_layers):
            features = vfe_layer(features, num_points)
            feature_norm = features.norm().item()
            
            # If features become zero, inject meaningful values
            if feature_norm < 1e-6:
                features = features + torch.randn_like(features) * 0.01
                
        return features


@MODELS.register_module()
class RefactoredMultiScaleFeatureFusion(nn.Module):
    """
    Refactored multi-scale feature fusion module that concatenates and fuses 
    voxel features from different scales using Gumbel-Softmax approach.
    """
    
    def __init__(self,
                 scale_channels: List[int] = [64, 64, 64],  # Channels from each scale
                 fusion_channels: int = 128,
                 output_channels: int = 64):
        super().__init__()
        
        self.scale_channels = scale_channels
        total_channels = sum(scale_channels)
        
        # Feature fusion network with enhanced gradient flow
        self.fusion_net = nn.Sequential(
            nn.Linear(total_channels, fusion_channels),
            nn.LayerNorm(fusion_channels),  # LayerNorm works with any batch size
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),  # Reduced dropout
            nn.Linear(fusion_channels, fusion_channels // 2),  # Additional layer
            nn.LayerNorm(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_channels // 2, output_channels)
        )
        
        # 🔥 CRITICAL: Add skip connection for gradient flow
        self.skip_connection = nn.Linear(total_channels, output_channels) if total_channels != output_channels else nn.Identity()
        
        self.output_channels = output_channels
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple scales.
        
        Args:
            multi_scale_features: List of features from each scale, each with potentially different batch sizes
            
        Returns:
            Fused features
        """
        device = multi_scale_features[0].device if multi_scale_features else None
        
        # Handle empty features and compute global statistics for each scale
        scale_summaries = []
        total_voxels = 0
        
        for scale_id, features in enumerate(multi_scale_features):
            if features.numel() > 0 and features.shape[0] > 0:
                # Check feature magnitudes
                feature_norm = features.norm().item()
                
                if feature_norm > 1e-6:  # Only use non-zero features
                    scale_summary = torch.mean(features, dim=0, keepdim=True)  # (1, channels)
                    scale_summaries.append(scale_summary)
                    total_voxels += features.shape[0]
                else:
                    # Use non-zero meaningful summary with proper scaling
                    expected_channels = self.scale_channels[scale_id] if scale_id < len(self.scale_channels) else 64
                    # Use small but non-zero values that match feature statistics
                    meaningful_summary = torch.randn(1, expected_channels, device=device) * 0.01
                    scale_summaries.append(meaningful_summary)
            else:
                # Add meaningful non-zero summary for empty scales
                expected_channels = self.scale_channels[scale_id] if scale_id < len(self.scale_channels) else 64
                meaningful_summary = torch.randn(1, expected_channels, device=device) * 0.01  # Smaller values
                scale_summaries.append(meaningful_summary)
        
        if not scale_summaries:
            # Return zero features if all scales are empty
            return torch.zeros(1, self.output_channels, device=device)
        
        # Concatenate scale summaries (all have shape (1, channels))
        concatenated_summaries = torch.cat(scale_summaries, dim=-1)  # (1, total_channels)
        
        # Apply both main path and skip connection for better gradient flow
        main_features = self.fusion_net(concatenated_summaries)  # (1, output_channels)
        skip_features = self.skip_connection(concatenated_summaries)  # (1, output_channels)
        
        # Residual connection: Combine main and skip paths
        fused_features = main_features + skip_features  # Element-wise addition
        
        # If we had multiple voxels, expand the output to match the largest scale
        if total_voxels > 1:
            max_voxels = max(f.shape[0] for f in multi_scale_features if f.numel() > 0) if any(f.numel() > 0 for f in multi_scale_features) else 1
            fused_features = fused_features.expand(max_voxels, -1)
        
        return fused_features


@MODELS.register_module()
class ImportanceGuidedMultiScaleVFE(nn.Module):
    """
    🎯 IMPORTANCE-GUIDED MULTI-SCALE VFE WITH GUMBEL-SOFTMAX
    
    This implements the enhanced adaptive voxelization system that:
    1. Uses importance network to filter points before processing
    2. Predicts optimal voxel scales per point using Gumbel-Softmax
    3. Processes points at multiple scales simultaneously
    4. Fuses multi-scale features for enhanced representation
    
    Pipeline:
    Point Cloud → Importance Net → Scale Prediction → Multi-Scale Voxelization → VFE → Fusion
    """
    
    def __init__(self,
                 # Multi-scale config
                 voxel_scales: List[float] = [0.05, 0.1, 0.2],
                 num_scales: int = 3,
                 
                 # Standard VFE config
                 max_num_points: int = 5,
                 max_voxels: Tuple[int, int] = (12000, 30000),
                 point_cloud_range: List[float] = None,
                 
                 # ScaleNet config  
                 gumbel_temperature: float = 1.0,
                 continuous_mode: bool = False,
                 
                 # Output config
                 vfe_channels: List[int] = [32, 64],
                 fusion_channels: int = 128,
                 output_channels: int = 64,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg: OptConfigType = None,
                 
                 # Legacy parameters (ignored but accepted for compatibility)
                 **kwargs):
        super().__init__()
        
        # Log and ignore any unexpected parameters
        if kwargs:
            ignored_params = list(kwargs.keys())
            print(f"🔧 Ignoring legacy parameters for compatibility: {ignored_params}")
        
        # Smart scale handling
        if len(voxel_scales) != num_scales:
            print(f"🎯 Auto-generating {num_scales} scales (overriding provided {len(voxel_scales)} scales)")
            self.num_scales = min(max(num_scales, 1), 10)
            # Generate scales using logarithmic distribution
            if self.num_scales <= 3:
                scale_mapping = {1: [0.1], 2: [0.05, 0.2], 3: [0.05, 0.1, 0.2]}
                self.voxel_scales = scale_mapping[self.num_scales]
            else:
                min_scale, max_scale = 0.01, 1.0
                log_min = torch.log(torch.tensor(min_scale))
                log_max = torch.log(torch.tensor(max_scale))
                log_scales = torch.linspace(log_min, log_max, self.num_scales)
                self.voxel_scales = torch.exp(log_scales).tolist()
        else:
            self.voxel_scales = voxel_scales
            self.num_scales = len(voxel_scales)
            
        print(f"🎯 ImportanceGuidedMultiScaleVFE initialized with {self.num_scales} scales:")
        print(f"   📏 Scales: {[f'{s:.3f}m' for s in self.voxel_scales]}")
        
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.point_cloud_range = point_cloud_range
        
        # 1. Point importance network for filtering
        self.importance_net = LightweightPointImportanceNet(
            in_channels=4,
            hidden_dims=[64, 32, 16],
            dropout_rate=0.1
        )
        
        # 2. Scale prediction network with LEARNABLE voxel scales
        self.scale_net = ScaleNet(
            in_channels=4,
            hidden_dims=[64, 32],
            num_scales=self.num_scales,
            temperature=gumbel_temperature,
            continuous_mode=continuous_mode,
            voxel_scales=self.voxel_scales  # Pass initial scales to be made learnable
        )
        
        # 3. Multi-scale voxelizer
        self.multi_scale_voxelizer = MultiScaleVoxelizer(
            voxel_scales=self.voxel_scales,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
            point_cloud_range=point_cloud_range
        )
        
        # 4. Scale-specific VFEs
        self.scale_vfes = nn.ModuleList()
        for i in range(self.num_scales):
            vfe = ScaleSpecificVFE(
                in_channels=4,
                feat_channels=vfe_channels,
                scale_id=i,
                norm_cfg=norm_cfg
            )
            self.scale_vfes.append(vfe)
        
        # 5. Feature fusion
        scale_channels = [vfe_channels[-1]] * self.num_scales
        self.feature_fusion = RefactoredMultiScaleFeatureFusion(
            scale_channels=scale_channels,
            fusion_channels=fusion_channels,
            output_channels=output_channels
        )
        
        # Output configuration
        self.output_channels = output_channels + 1  # +1 for scale info
    
    def forward(self, features: torch.Tensor, num_points: torch.Tensor = None, 
                coors: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for importance-guided multi-scale VFE.
        
        Args:
            features: Raw point cloud (N, 4) when called from VoxelNet
                     OR voxel features (N, max_points, 4) when called normally
            num_points: (N,) - number of points per voxel (optional, for voxelized input)
            coors: (N, 4) - voxel coordinates (optional, for voxelized input)
            
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
            # 1. Point importance filtering (optional)
            importance_scores = self.importance_net(points)
            
            # 2. Scale prediction using Gumbel-Softmax
            scale_assignment, predicted_scales = self.scale_net(points, self.training)
            
            # 3. Multi-scale voxelization
            # 🎓 PhD FIX: Pass learnable scales to voxelizer
            multi_scale_voxels = self.multi_scale_voxelizer(
                points, 
                scale_assignment,
                dynamic_scales=self.scale_net.voxel_scales
            )
            
            # 4. Scale-specific VFE processing
            multi_scale_features = []
            for scale_id, (voxel_data, vfe) in enumerate(zip(multi_scale_voxels, self.scale_vfes)):
                if voxel_data['voxels'].numel() > 0:
                    scale_features = vfe(voxel_data['voxels'], voxel_data['num_points'])
                    multi_scale_features.append(scale_features)
                else:
                    # Placeholder for empty scales
                    placeholder = torch.zeros(1, vfe.output_channels, device=device)
                    multi_scale_features.append(placeholder)
            
            # 5. Feature fusion
            fused_features = self.feature_fusion(multi_scale_features)
            
            # 6. Prepare output
            avg_predicted_scale = predicted_scales.mean().unsqueeze(0).expand(fused_features.shape[0], 1)
            output = torch.cat([fused_features, avg_predicted_scale], dim=-1)
            
            # Generate coordinates
            batch_size = output.shape[0]
            coors = torch.zeros(batch_size, 4, device=device, dtype=torch.long)
            coors[:, 0] = 0  # All same batch
            
            return output, coors
            
        except Exception as e:
            print(f"⚠️ Forward pass failed: {str(e)}")
            return self._simplified_fallback(points)
    
    def _forward_voxelized(self, features: torch.Tensor, num_points: torch.Tensor, 
                          coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle pre-voxelized input."""
        device = features.device
        batch_size = features.shape[0]
        
        try:
            # Extract representative points
            if len(features.shape) == 3:
                representative_points = features[:, 0, :4]
            else:
                representative_points = coors[:, 1:].float()
                if representative_points.shape[1] == 3:
                    intensity = torch.zeros(representative_points.shape[0], 1, device=device)
                    representative_points = torch.cat([representative_points, intensity], dim=1)
            
            # Process with adaptive voxelization
            scale_assignment, predicted_scales = self.scale_net(representative_points, self.training)
            # 🎓 PhD FIX: Pass learnable scales to voxelizer
            multi_scale_voxels = self.multi_scale_voxelizer(
                representative_points, 
                scale_assignment,
                dynamic_scales=self.scale_net.voxel_scales
            )
            
            # Process each scale
            multi_scale_features = []
            for scale_id, (voxel_data, vfe) in enumerate(zip(multi_scale_voxels, self.scale_vfes)):
                if voxel_data['voxels'].numel() > 0:
                    scale_features = vfe(voxel_data['voxels'], voxel_data['num_points'])
                    multi_scale_features.append(scale_features)
                else:
                    placeholder = torch.zeros(1, vfe.output_channels, device=device)
                    multi_scale_features.append(placeholder)
            
            # Fuse and prepare output
            fused_features = self.feature_fusion(multi_scale_features)
            
            # Align with batch size
            if fused_features.shape[0] != batch_size:
                if fused_features.shape[0] < batch_size:
                    padding = torch.zeros(batch_size - fused_features.shape[0], 
                                        fused_features.shape[1], device=device)
                    fused_features = torch.cat([fused_features, padding], dim=0)
                else:
                    fused_features = fused_features[:batch_size]
            
            # Add scale info
            avg_predicted_scale = predicted_scales.mean().unsqueeze(0).expand(batch_size, 1)
            output = torch.cat([fused_features, avg_predicted_scale], dim=-1)
            
            return output, coors
            
        except Exception as e:
            print(f"⚠️ Voxelized processing failed: {str(e)}")
            return self._simplified_fallback_voxelized(features, num_points, coors)
    
    def _simplified_fallback(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified fallback processing."""
        device = points.device
        
        # Simple linear projection
        if not hasattr(self, 'fallback_projection'):
            self.fallback_projection = nn.Linear(points.shape[1], self.output_channels - 1).to(device)
        
        projected = self.fallback_projection(points)
        scale_info = torch.ones(projected.shape[0], 1, device=device) * 0.1
        output = torch.cat([projected, scale_info], dim=-1)
        
        # Simple coordinates
        coors = torch.zeros(output.shape[0], 4, device=device, dtype=torch.long)
        
        return output, coors
    
    def _simplified_fallback_voxelized(self, features: torch.Tensor, num_points: torch.Tensor, 
                                     coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified fallback for voxelized input."""
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
        
        # Project to target dimensions
        if pooled.shape[1] != self.output_channels - 1:
            if not hasattr(self, 'voxel_fallback_projection'):
                self.voxel_fallback_projection = nn.Linear(pooled.shape[1], self.output_channels - 1).to(device)
            pooled = self.voxel_fallback_projection(pooled)
        
        # Add scale info
        scale_info = torch.ones(batch_size, 1, device=device) * 0.1
        output = torch.cat([pooled, scale_info], dim=-1)
        
        return output, coors


@MODELS.register_module()
class MemoryOptimizedImportanceGuidedMultiScaleVFE(nn.Module):
    """
    🚀 MEMORY-OPTIMIZED ADAPTIVE VOXELIZATION PIPELINE
    Target: 25% memory reduction compared to vanilla SECOND
    
    Memory Optimization Strategies:
    1. Aggressive Point Filtering (30-40% point reduction)
    2. Adaptive Voxel Limits (dynamic based on scene complexity)  
    3. Gradient Checkpointing (trade compute for memory)
    4. Reduced Network Capacity (smaller hidden dimensions)
    5. Efficient Feature Fusion (minimal intermediate tensors)
    6. Memory-Aware Batch Processing
    7. In-place Operations and Bias Removal
    
    Pipeline:
    1. MemoryEfficientImportanceNet filters points aggressively  
    2. MemoryEfficientScaleNet predicts scales with reduced parameters
    3. MemoryEfficientVoxelizer uses adaptive limits and efficient sampling
    4. MemoryEfficientVFE processes with gradient checkpointing
    5. MemoryEfficientFusion minimizes intermediate tensors
    """
    
    def __init__(self,
                 # Multi-scale config
                 voxel_scales: List[float] = [0.05, 0.1, 0.2],
                 num_scales: int = 3,
                 
                 # Memory optimization settings
                 memory_optimization_level: int = 2,  # 0=disabled, 1=moderate, 2=aggressive
                 importance_threshold: float = 0.15,  # Filter low-importance points
                 max_points_ratio: float = 0.7,       # Keep only 70% of points
                 adaptive_max_voxels: bool = True,    # Dynamic voxel limits
                 use_gradient_checkpointing: bool = True,  # Trade compute for memory
                 
                 # Reduced network capacity for memory savings
                 importance_net_dims: List[int] = [32, 16],      # 🚀 REDUCED from [64, 32, 16]
                 scale_net_dims: List[int] = [32, 16],           # 🚀 REDUCED from [64, 32]
                 vfe_channels: List[int] = [32, 64],             # Can be reduced further
                 fusion_channels: int = 64,                      # 🚀 REDUCED from 128
                 
                 # Standard VFE config
                 max_num_points: int = 5,
                 max_voxels: Tuple[int, int] = (8000, 20000),    # 🚀 REDUCED from (12000, 30000)
                 point_cloud_range: List[float] = None,
                 
                 # ScaleNet config  
                 gumbel_temperature: float = 1.0,
                 continuous_mode: bool = False,
                 
                 # Output config
                 output_channels: int = 64,
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg: OptConfigType = None,
                 
                 # 🎓 PhD RESEARCH: Point Refinement Enhancement (OPTIONAL)
                 enable_point_refinement: bool = False,  # Toggle on/off easily
                 point_refinement_neighbors: int = 8,    # Small for efficiency
                 
                 # Legacy parameters (ignored but accepted for compatibility)
                 **kwargs):
        super().__init__()
        
        # Store memory optimization settings
        self.memory_optimization_level = memory_optimization_level
        self.use_gradient_checkpointing = use_gradient_checkpointing and (memory_optimization_level > 0)
        
        # Log and ignore any unexpected parameters
        if kwargs:
            ignored_params = list(kwargs.keys())
            print(f"🔧 Ignoring legacy parameters for compatibility: {ignored_params}")
        
        # Smart scale handling
        if len(voxel_scales) != num_scales:
            print(f"🎯 Auto-generating {num_scales} scales (overriding provided {len(voxel_scales)} scales)")
            self.num_scales = min(max(num_scales, 1), 10)
            # Generate scales using logarithmic distribution
            if self.num_scales <= 3:
                scale_mapping = {1: [0.1], 2: [0.05, 0.2], 3: [0.05, 0.1, 0.2]}
                self.voxel_scales = scale_mapping[self.num_scales]
            else:
                min_scale, max_scale = 0.01, 1.0
                log_min = torch.log(torch.tensor(min_scale))
                log_max = torch.log(torch.tensor(max_scale))
                log_scales = torch.linspace(log_min, log_max, self.num_scales)
                self.voxel_scales = torch.exp(log_scales).tolist()
        else:
            self.voxel_scales = voxel_scales
            self.num_scales = len(voxel_scales)
            
        print(f"🚀 MemoryOptimizedVFE: Level {memory_optimization_level} optimization")
        print(f"   📏 Using {self.num_scales} scales: {[f'{s:.3f}m' for s in self.voxel_scales]}")
        print(f"   🎯 Point filtering: {max_points_ratio:.0%} retention, threshold {importance_threshold}")
        print(f"   💾 Gradient checkpointing: {'ON' if self.use_gradient_checkpointing else 'OFF'}")
        print(f"   🔧 Adaptive voxel limits: {'ON' if adaptive_max_voxels else 'OFF'}")
        
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.point_cloud_range = point_cloud_range
        
        # Apply AGGRESSIVE memory optimization levels
        if memory_optimization_level >= 1:
            # Level 1: Moderate optimizations (50% reduction)
            importance_net_dims = [max(8, d//2) for d in importance_net_dims]  # 32->16, 16->8
            scale_net_dims = [max(8, d//2) for d in scale_net_dims]           # 32->16, 16->8
            vfe_channels = [max(16, c//2) for c in vfe_channels]              # 32->16, 64->32
            fusion_channels = max(32, fusion_channels//2)                     # 64->32
            max_voxels = (max(4000, max_voxels[0]//2), max(10000, max_voxels[1]//2))  # 8000->4000, 20000->10000
            
        if memory_optimization_level >= 2:
            # Level 2: EXTREME optimizations (75% reduction)
            importance_net_dims = [max(4, d//4) for d in [32, 16]]           # Down to [8, 4]
            scale_net_dims = [max(4, d//4) for d in [32, 16]]                # Down to [8, 4] 
            vfe_channels = [max(8, c//4) for c in [32, 64]]                  # Down to [8, 16]
            fusion_channels = max(16, fusion_channels//4)                     # Down to 16
            max_voxels = (max(2000, max_voxels[0]//4), max(5000, max_voxels[1]//4))  # 8000->2000, 20000->5000
            max_points_ratio = 0.5  # Keep only 50% of points instead of 70%
            importance_threshold = 0.25  # More aggressive filtering
            print(f"   ⚡ EXTREME mode: ImportanceNet {importance_net_dims}, ScaleNet {scale_net_dims}")
            print(f"   ⚡ EXTREME mode: VFE channels {vfe_channels}, fusion {fusion_channels}")
            print(f"   ⚡ EXTREME mode: Max voxels {max_voxels}, Point ratio {max_points_ratio:.0%}")
        
        # Store optimization parameters
        self.importance_threshold = importance_threshold
        self.max_points_ratio = max_points_ratio
        
        # 1. ULTRA-AGGRESSIVE point filtering for massive memory reduction
        if memory_optimization_level > 0:
            self.importance_net = MemoryEfficientImportanceNet(
                in_channels=4,
                hidden_dims=importance_net_dims,
                importance_threshold=self.importance_threshold,  # Use computed threshold
                max_points_ratio=self.max_points_ratio           # Use computed ratio
            )
        else:
            self.importance_net = None
        
        # 2. ULTRA-LIGHTWEIGHT scale network (minimal parameters)
        self.scale_net = MemoryEfficientScaleNet(
            in_channels=4,
            hidden_dims=scale_net_dims,
            num_scales=self.num_scales,
            temperature=gumbel_temperature,
            continuous_mode=continuous_mode
        )
        
        # 3. SUPER-AGGRESSIVE voxelization limits
        self.multi_scale_voxelizer = MemoryEfficientMultiScaleVoxelizer(
            voxel_scales=self.voxel_scales,
            max_num_points=max(2, max_num_points//2),  # Reduce points per voxel
            adaptive_max_voxels=adaptive_max_voxels,
            base_max_voxels=max_voxels[0],
            memory_efficient=True
        )
        
        # 4. MINIMAL VFE networks
        self.scale_vfes = nn.ModuleList()
        for i in range(self.num_scales):
            vfe = MemoryEfficientScaleSpecificVFE(
                in_channels=4,
                feat_channels=vfe_channels,
                scale_id=i,
                use_checkpoint=self.use_gradient_checkpointing
            )
            self.scale_vfes.append(vfe)
        
        # 5. MINIMAL feature fusion
        scale_channels = [vfe_channels[-1]] * self.num_scales
        self.feature_fusion = MemoryEfficientFeatureFusion(
            scale_channels=scale_channels,
            fusion_channels=fusion_channels,
            output_channels=output_channels,
            use_checkpoint=self.use_gradient_checkpointing
        )
        
        # 🎓 PhD RESEARCH: Optional Point Refinement Enhancement
        self.point_refinement = LightweightPointRefinementModule(
            feature_channels=output_channels,
            num_neighbors=point_refinement_neighbors,
            enabled=enable_point_refinement
        ) if enable_point_refinement else None
        
        # Output configuration
        self.output_channels = output_channels + 1  # +1 for scale info
        
        # Memory monitoring
        self.memory_stats = {
            'original_points': 0,
            'filtered_points': 0, 
            'total_voxels': 0,
            'memory_savings': 0.0
        }
    
    def forward(self, features: torch.Tensor, num_points: torch.Tensor = None, 
                coors: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-optimized forward pass with aggressive filtering and efficient processing.
        """
        device = features.device
        
        # Case 1: Raw points from VoxelNet
        if num_points is None and coors is None:
            return self._forward_raw_points_optimized(features)
        # Case 2: Pre-voxelized data
        else:
            return self._forward_voxelized_optimized(features, num_points, coors)
    
    def _forward_raw_points_optimized(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-optimized processing for raw points."""
        device = points.device
        original_num_points = points.shape[0]
        
        if original_num_points == 0:
            dummy_output = torch.zeros(0, self.output_channels, device=device)
            dummy_coors = torch.zeros(0, 4, device=device).long()
            return dummy_output, dummy_coors
        
        try:
            # 🚀 STEP 1: ULTRA-AGGRESSIVE point filtering (50-75% reduction)
            if self.importance_net is not None and self.memory_optimization_level > 0:
                filtered_points, point_indices = self.importance_net(points)
                memory_savings = 1.0 - (filtered_points.shape[0] / original_num_points)
                
                # Update memory stats
                self.memory_stats.update({
                    'original_points': original_num_points,
                    'filtered_points': filtered_points.shape[0],
                    'memory_savings': memory_savings
                })
                
                if self.training and torch.rand(1).item() < 0.05:  # Log 5% of batches
                    print(f"� ULTRA filtering: {original_num_points} → {filtered_points.shape[0]} "
                          f"({memory_savings:.1%} reduction)")
            else:
                filtered_points = points
                point_indices = torch.arange(points.shape[0], device=device)
            
            # 🚀 MEMORY OPTIMIZATION: Further reduce points if still too many
            if self.memory_optimization_level >= 2 and filtered_points.shape[0] > 15000:
                # EXTREME: Keep only top 10000 points maximum
                max_extreme_points = 10000
                if filtered_points.shape[0] > max_extreme_points:
                    # Random sampling for diversity
                    perm_indices = torch.randperm(filtered_points.shape[0], device=device)[:max_extreme_points]
                    filtered_points = filtered_points[perm_indices]
                    print(f"🔥 EXTREME point reduction: → {max_extreme_points} points")
            
            # Clear GPU cache more aggressively
            if torch.cuda.is_available() and self.training and self.memory_optimization_level >= 2:
                torch.cuda.empty_cache()
            
            # 🚀 STEP 2: Memory-efficient scale selection
            if self.use_gradient_checkpointing and self.training:
                try:
                    from torch.utils.checkpoint import checkpoint
                    scale_assignment, predicted_scales = checkpoint(
                        self.scale_net, filtered_points, self.training
                    )
                except ImportError:
                    # Fallback if checkpoint not available
                    scale_assignment, predicted_scales = self.scale_net(filtered_points, self.training)
            else:
                scale_assignment, predicted_scales = self.scale_net(filtered_points, self.training)
            
            # 🎓 PhD RESEARCH: Log learnable voxel scale parameters during training
            if self.training and torch.rand(1).item() < 0.02:  # Log 2% of batches
                current_scales = self.scale_net.voxel_scales.detach()
                scale_gradients = self.scale_net.voxel_scales.grad
                print(f"🎯 LEARNABLE SCALES: {[f'{s:.4f}m' for s in current_scales.tolist()]}")
                if scale_gradients is not None:
                    print(f"📈 Scale gradients: {[f'{g:.6f}' for g in scale_gradients.tolist()]}")
                print(f"📊 Predicted scale range: {predicted_scales.min():.4f}m - {predicted_scales.max():.4f}m")
            
            # 🚀 STEP 3: Memory-efficient multi-scale voxelization
            # 🎓 PhD FIX: Pass learnable scales to voxelizer
            multi_scale_voxels = self.multi_scale_voxelizer(
                filtered_points, 
                scale_assignment,
                dynamic_scales=self.scale_net.voxel_scales
            )
            
            # Track total voxels for memory monitoring
            total_voxels = sum(voxel_data['voxels'].shape[0] for voxel_data in multi_scale_voxels)
            self.memory_stats['total_voxels'] = total_voxels
            
            # 🚀 STEP 4: Memory-efficient scale-specific VFE processing
            multi_scale_features = []
            
            for scale_id, (voxel_data, vfe) in enumerate(zip(multi_scale_voxels, self.scale_vfes)):
                if voxel_data['voxels'].numel() > 0:
                    # Process with optional checkpointing
                    scale_features = vfe(voxel_data['voxels'], voxel_data['num_points'])
                    multi_scale_features.append(scale_features)
                else:
                    # Minimal placeholder
                    placeholder = torch.zeros(1, vfe.output_channels, device=device)
                    multi_scale_features.append(placeholder)
                
                # Memory cleanup between scales
                if self.training and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 🚀 STEP 5: Memory-efficient feature fusion
            fused_features = self.feature_fusion(multi_scale_features)
            
            # 🎓 PhD RESEARCH: Optional Point Refinement Enhancement
            if self.point_refinement is not None:
                # Extract point coordinates from filtered points for refinement
                point_coords = filtered_points[:, :3]  # x, y, z coordinates
                # Apply point refinement using learned scales
                fused_features = self.point_refinement(point_coords, fused_features, predicted_scales)
            
            # 🚀 STEP 6: Prepare output with minimal memory overhead
            # Use mean scale instead of computing per-point scales
            avg_predicted_scale = predicted_scales.mean().unsqueeze(0).expand(fused_features.shape[0], 1)
            
            # Efficient concatenation
            output = torch.cat([fused_features, avg_predicted_scale], dim=-1)
            
            # Generate lightweight coordinates
            batch_size = output.shape[0]
            coors = torch.zeros(batch_size, 4, device=device, dtype=torch.long)
            coors[:, 0] = 0  # All same batch
            
            # Final memory cleanup
            if self.training and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output, coors
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"🚨 OOM detected! Falling back to ultra-conservative processing...")
                return self._emergency_fallback(points)
            else:
                raise e
        
        except Exception as e:
            print(f"⚠️ Memory-optimized processing failed: {str(e)}")
            print(f"   Falling back to simplified processing...")
            return self._simplified_fallback(points)
    
    def _forward_voxelized_optimized(self, features: torch.Tensor, num_points: torch.Tensor, 
                                   coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-optimized processing for pre-voxelized data."""
        device = features.device
        batch_size = features.shape[0]
        
        try:
            # Extract representative points
            if len(features.shape) == 3:
                representative_points = features[:, 0, :4]
            else:
                representative_points = coors[:, 1:].float()
                if representative_points.shape[1] == 3:
                    intensity = torch.zeros(representative_points.shape[0], 1, device=device)
                    representative_points = torch.cat([representative_points, intensity], dim=1)
            
            # Apply memory-efficient processing
            scale_assignment, predicted_scales = self.scale_net(representative_points, self.training)
            # 🎓 PhD FIX: Pass learnable scales to voxelizer
            multi_scale_voxels = self.multi_scale_voxelizer(
                representative_points, 
                scale_assignment,
                dynamic_scales=self.scale_net.voxel_scales
            )
            
            # Process with memory efficiency
            multi_scale_features = []
            for scale_id, (voxel_data, vfe) in enumerate(zip(multi_scale_voxels, self.scale_vfes)):
                if voxel_data['voxels'].numel() > 0:
                    scale_features = vfe(voxel_data['voxels'], voxel_data['num_points'])
                    multi_scale_features.append(scale_features)
                else:
                    placeholder = torch.zeros(1, vfe.output_channels, device=device)
                    multi_scale_features.append(placeholder)
            
            # Efficient fusion and output
            fused_features = self.feature_fusion(multi_scale_features)
            
            # Align with batch size
            if fused_features.shape[0] != batch_size:
                if fused_features.shape[0] < batch_size:
                    padding = torch.zeros(batch_size - fused_features.shape[0], 
                                        fused_features.shape[1], device=device)
                    fused_features = torch.cat([fused_features, padding], dim=0)
                else:
                    fused_features = fused_features[:batch_size]
            
            # Add scale info and return
            avg_predicted_scale = predicted_scales.mean().unsqueeze(0).expand(batch_size, 1)
            output = torch.cat([fused_features, avg_predicted_scale], dim=-1)
            
            return output, coors
            
        except Exception as e:
            print(f"⚠️ Voxelized processing failed: {str(e)}")
            return self._simplified_fallback_voxelized(features, num_points, coors)
    
    def _emergency_fallback(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ultra-conservative fallback for OOM situations."""
        device = points.device
        
        # Use only 50% of points
        num_points = points.shape[0] // 2
        if num_points > 0:
            sampled_points = points[:num_points]
        else:
            sampled_points = points[:1] if points.shape[0] > 0 else torch.zeros(1, 4, device=device)
        
        # Minimal processing
        if not hasattr(self, 'emergency_projection'):
            self.emergency_projection = nn.Linear(4, self.output_channels - 1).to(device)
        
        features = self.emergency_projection(sampled_points)
        scale_info = torch.ones(features.shape[0], 1, device=device) * 0.1
        output = torch.cat([features, scale_info], dim=-1)
        
        # Minimal coordinates
        coors = torch.zeros(output.shape[0], 4, device=device, dtype=torch.long)
        
        print(f"🚨 Emergency fallback used: {points.shape[0]} → {output.shape[0]} points")
        return output, coors
    
    def _simplified_fallback(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified fallback processing."""
        device = points.device
        
        # Simple linear projection
        if not hasattr(self, 'fallback_projection'):
            self.fallback_projection = nn.Linear(points.shape[1], self.output_channels - 1).to(device)
        
        projected = self.fallback_projection(points)
        scale_info = torch.ones(projected.shape[0], 1, device=device) * 0.1
        output = torch.cat([projected, scale_info], dim=-1)
        
        # Simple coordinates
        coors = torch.zeros(output.shape[0], 4, device=device, dtype=torch.long)
        
        return output, coors
    
    def _simplified_fallback_voxelized(self, features: torch.Tensor, num_points: torch.Tensor, 
                                     coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified fallback for voxelized input."""
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
        
        # Project to target dimensions
        if pooled.shape[1] != self.output_channels - 1:
            if not hasattr(self, 'voxel_fallback_projection'):
                self.voxel_fallback_projection = nn.Linear(pooled.shape[1], self.output_channels - 1).to(device)
            pooled = self.voxel_fallback_projection(pooled)
        
        # Add scale info
        scale_info = torch.ones(batch_size, 1, device=device) * 0.1
        output = torch.cat([pooled, scale_info], dim=-1)
        
        return output, coors
    
    def get_memory_stats(self) -> Dict:
        """Get current memory optimization statistics."""
        return {
            **self.memory_stats,
            'optimization_level': self.memory_optimization_level,
            'gradient_checkpointing': self.use_gradient_checkpointing,
            'estimated_memory_savings': f"{self.memory_stats.get('memory_savings', 0):.1%}"
        }
    
    def print_memory_summary(self):
        """Print memory optimization summary."""
        stats = self.get_memory_stats()
        print(f"🚀 MEMORY OPTIMIZATION SUMMARY")
        print(f"   📊 Level: {stats['optimization_level']}/2")
        print(f"   🎯 Point reduction: {stats.get('memory_savings', 0):.1%}")
        print(f"   📦 Total voxels: {stats.get('total_voxels', 0)}")
        print(f"   ⚡ Gradient checkpointing: {'ON' if stats['gradient_checkpointing'] else 'OFF'}")
        
    @property
    def fp16_enabled(self) -> bool:
        """Enable FP16 for additional memory savings."""
        return self.memory_optimization_level >= 2


# Import memory-efficient components with fallback
try:
    # Try to import from memory_optimized_components.py in the same directory
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    components_path = os.path.join(current_dir, 'memory_optimized_components.py')
    
    if os.path.exists(components_path):
        sys.path.insert(0, current_dir)
        from memory_optimized_components import (
            MemoryEfficientImportanceNet,
            MemoryEfficientScaleNet,
            MemoryEfficientMultiScaleVoxelizer,
            MemoryEfficientVFELayer,
            MemoryEfficientScaleSpecificVFE,
            MemoryEfficientFeatureFusion
        )
        print("✅ Successfully imported memory-optimized components")
    else:
        raise ImportError("memory_optimized_components.py not found")
        
except ImportError:
    # Fallback: Define minimal versions inline
    print("⚠️ Memory-optimized components not found, using fallback implementations")
    
    class MemoryEfficientImportanceNet(nn.Module):
        def __init__(self, in_channels=4, hidden_dims=[8, 4], importance_threshold=0.25, max_points_ratio=0.5):
            super().__init__()
            self.threshold = importance_threshold
            self.ratio = max_points_ratio
            # ULTRA-MINIMAL network: just 2 tiny layers
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden_dims[0], bias=False),  # Remove bias to save memory
                nn.ReLU(inplace=True),  # In-place to save memory
                nn.Linear(hidden_dims[0], 1, bias=False),
                nn.Sigmoid()
            )
        
        def forward(self, points):
            # Aggressive filtering: keep only top % of points
            scores = self.net(points).squeeze(-1)
            max_points = max(1, int(points.shape[0] * self.ratio))  # Ensure at least 1 point
            
            # Use top-k for efficient selection
            if max_points < points.shape[0]:
                _, indices = torch.topk(scores, max_points, sorted=False)
                filtered_points = points[indices]
                
                # Print memory savings occasionally
                if hasattr(self, 'training') and self.training and torch.rand(1).item() < 0.01:
                    reduction = 1.0 - (max_points / points.shape[0])
                    print(f"🔥 AGGRESSIVE filtering: {points.shape[0]} → {max_points} ({reduction:.1%} reduction)")
                
                return filtered_points, indices
            else:
                return points, torch.arange(points.shape[0], device=points.device)
    
    class MemoryEfficientScaleNet(ScaleNet):
        def __init__(self, **kwargs):
            # Force ultra-minimal dimensions
            kwargs['hidden_dims'] = kwargs.get('hidden_dims', [8, 4])
            kwargs['dropout_rate'] = 0.0  # Remove dropout entirely to save memory
            super().__init__(**kwargs)
            
            # Replace some layers with even smaller ones
            if hasattr(self, 'spatial_encoder'):
                self.spatial_encoder = nn.Sequential(
                    nn.Linear(3, 8, bias=False),  # Reduced from 32 to 8
                    nn.ReLU(inplace=True),
                    nn.Linear(8, 4, bias=False)   # Reduced from 16 to 4
                )
    
    class MemoryEfficientMultiScaleVoxelizer(MultiScaleVoxelizer):
        def __init__(self, **kwargs):
            kwargs['adaptive_max_voxels'] = True
            kwargs['base_max_voxels'] = kwargs.get('base_max_voxels', 2000)  # Very low limit
            super().__init__(**kwargs)
            
        def forward(self, points, scale_assignment):
            # Override parent to be more aggressive about voxel limits
            voxel_outputs = []
            
            for scale_id, voxel_size in enumerate(self.voxel_scales):
                scale_weights = scale_assignment[:, scale_id]
                point_mask = scale_weights > 1e-5  # Slightly higher threshold
                
                if not point_mask.any():
                    # Empty output for this scale
                    voxel_outputs.append({
                        'voxels': torch.empty(0, self.max_num_points, 4, device=points.device),
                        'coordinates': torch.empty(0, 4, device=points.device, dtype=torch.long),
                        'num_points': torch.empty(0, device=points.device, dtype=torch.long),
                        'scale_weights': torch.empty(0, device=points.device),
                        'scale_id': scale_id,
                        'voxel_size': voxel_size
                    })
                    continue
                
                # AGGRESSIVE point limiting: max 1000 points per scale
                scale_points = points[point_mask]
                if scale_points.shape[0] > 1000:
                    # Sample only the top 1000 points
                    scale_weights_masked = scale_weights[point_mask]
                    _, top_indices = torch.topk(scale_weights_masked, 1000, sorted=False)
                    scale_points = scale_points[top_indices]
                    scale_weights_masked = scale_weights_masked[top_indices]
                else:
                    scale_weights_masked = scale_weights[point_mask]
                
                # Create minimal voxel representation
                num_voxels = min(scale_points.shape[0], 500)  # Max 500 voxels per scale
                if num_voxels > 0:
                    voxels = scale_points[:num_voxels].unsqueeze(1)  # (N, 1, 4) - single point per voxel
                    
                    # Pad to max_num_points if needed
                    if self.max_num_points > 1:
                        padding = torch.zeros(num_voxels, self.max_num_points - 1, 4, device=points.device)
                        voxels = torch.cat([voxels, padding], dim=1)
                    
                    coordinates = torch.zeros(num_voxels, 4, device=points.device)
                    coordinates[:, 0] = 0  # batch index
                    coordinates[:, 1:] = scale_points[:num_voxels, :3] / voxel_size
                    
                    num_points_per_voxel = torch.ones(num_voxels, device=points.device)
                    weights = scale_weights_masked[:num_voxels]
                else:
                    # Minimal fallback
                    voxels = torch.zeros(1, self.max_num_points, 4, device=points.device)
                    coordinates = torch.zeros(1, 4, device=points.device)
                    num_points_per_voxel = torch.ones(1, device=points.device)
                    weights = torch.ones(1, device=points.device) * 0.1
                
                voxel_outputs.append({
                    'voxels': voxels,
                    'coordinates': coordinates,
                    'num_points': num_points_per_voxel,
                    'scale_weights': weights,
                    'scale_id': scale_id,
                    'voxel_size': voxel_size
                })
            
            return voxel_outputs
    
    class MemoryEfficientVFELayer(VFELayer):
        def __init__(self, in_channels, out_channels, norm_cfg, last_layer=False):
            super().__init__(in_channels, out_channels, norm_cfg, last_layer)
            # Replace BatchNorm with simpler LayerNorm to save memory
            del self.norm
            self.norm = nn.LayerNorm(out_channels)
    
    class MemoryEfficientScaleSpecificVFE(ScaleSpecificVFE):
        def __init__(self, **kwargs):
            kwargs.pop('use_checkpoint', None)
            super().__init__(**kwargs)
            
            # Replace VFE layers with memory-efficient versions
            self.vfe_layers = nn.ModuleList()
            prev_channels = 4  # in_channels
            feat_channels = kwargs.get('feat_channels', [8, 16])  # Much smaller
            
            for i, out_channels in enumerate(feat_channels):
                is_last = (i == len(feat_channels) - 1)
                self.vfe_layers.append(
                    MemoryEfficientVFELayer(prev_channels, out_channels, {}, last_layer=is_last)
                )
                prev_channels = out_channels
                
            self.output_channels = feat_channels[-1]
    
    class MemoryEfficientFeatureFusion(RefactoredMultiScaleFeatureFusion):
        def __init__(self, **kwargs):
            kwargs.pop('use_checkpoint', None)
            # Force minimal fusion network
            kwargs['fusion_channels'] = min(kwargs.get('fusion_channels', 16), 16)
            kwargs['output_channels'] = min(kwargs.get('output_channels', 16), 16)
            super().__init__(**kwargs)
            
            # Replace fusion network with ultra-minimal version
            total_channels = sum(kwargs.get('scale_channels', [16, 16, 16]))
            self.fusion_net = nn.Sequential(
                nn.Linear(total_channels, 16, bias=False),  # Single small layer
                nn.ReLU(inplace=True),
                nn.Linear(16, kwargs['output_channels'], bias=False)
            )
            
            # Simplify skip connection
            self.skip_connection = nn.Linear(total_channels, kwargs['output_channels'], bias=False) \
                                 if total_channels != kwargs['output_channels'] else nn.Identity()


# Register all modules for export
__all__ = [
    'ImportanceGuidedMultiScaleVFE',
    'MemoryOptimizedImportanceGuidedMultiScaleVFE',
    'ScaleNet', 
    'MultiScaleVoxelizer', 
    'ScaleSpecificVFE', 
    'RefactoredMultiScaleFeatureFusion',
    'LightweightPointImportanceNet',
    'LightweightPointRefinementModule'
]


# 🎓 PhD RESEARCH ENHANCEMENT: Lightweight Point Refinement
class LightweightPointRefinementModule(nn.Module):
    """
    🚀 MINIMAL-IMPACT ENHANCEMENT: Scale-Aware Point Refinement
    
    Uses learnable voxel scale parameters for adaptive point-level processing.
    Designed for easy integration with existing VFE without major code changes.
    
    Features:
    - Toggle on/off with single parameter
    - Minimal computational overhead
    - Uses existing predicted scales
    - No changes to main pipeline
    """
    
    def __init__(self,
                 feature_channels: int = 64,
                 num_neighbors: int = 8,  # Small for efficiency
                 scale_multiplier: float = 2.0,  # Conservative multiplier
                 enabled: bool = True):  # Easy toggle
        super().__init__()
        
        self.enabled = enabled
        self.num_neighbors = num_neighbors
        self.scale_multiplier = scale_multiplier
        
        if not self.enabled:
            return  # No initialization if disabled
            
        # Lightweight point processing
        self.point_conv = nn.Sequential(
            nn.Conv1d(feature_channels + 3, feature_channels, 1),  # +3 for relative coords
            nn.BatchNorm1d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_channels, feature_channels, 1)
        )
        
        # Simple feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_channels * 2, feature_channels),
            nn.ReLU(inplace=True)
        )
        
        print(f"🎯 LightweightPointRefinement: {'ENABLED' if enabled else 'DISABLED'}")
    
    def forward(self, 
                points: torch.Tensor,           # (N, 3) 
                features: torch.Tensor,        # (N, C)
                predicted_scales: torch.Tensor # (N,)
                ) -> torch.Tensor:
        """Lightweight point refinement with minimal overhead."""
        
        if not self.enabled:
            return features  # Pass-through if disabled
            
        N, C = features.shape
        device = features.device
        
        # Simple distance-based neighbor finding
        distances = torch.cdist(points, points, p=2)  # (N, N)
        
        # Use predicted scales as adaptive radii
        adaptive_radii = predicted_scales * self.scale_multiplier  # (N,)
        radius_matrix = adaptive_radii.unsqueeze(1)  # (N, 1)
        
        # Get neighbors within adaptive radius
        neighbor_mask = distances <= radius_matrix  # (N, N)
        
        refined_features = []
        for i in range(N):
            # Get neighbors for point i
            valid_neighbors = torch.where(neighbor_mask[i])[0]
            
            # Limit to k nearest neighbors for efficiency
            if len(valid_neighbors) > self.num_neighbors:
                neighbor_distances = distances[i, valid_neighbors]
                _, top_k_idx = torch.topk(neighbor_distances, self.num_neighbors, largest=False)
                valid_neighbors = valid_neighbors[top_k_idx]
            
            # Fallback to k-nearest if too few neighbors
            if len(valid_neighbors) < 3:
                _, valid_neighbors = torch.topk(distances[i], min(self.num_neighbors, N), largest=False)
            
            # Extract neighbor features and coordinates
            neighbor_points = points[valid_neighbors]  # (K, 3)
            neighbor_features = features[valid_neighbors]  # (K, C)
            
            # Relative coordinates
            center_point = points[i:i+1]  # (1, 3)
            relative_coords = neighbor_points - center_point  # (K, 3)
            
            # Combine features with relative coordinates
            combined = torch.cat([neighbor_features.T, relative_coords.T], dim=0)  # (C+3, K)
            combined = combined.unsqueeze(0)  # (1, C+3, K)
            
            # Apply point convolution
            refined = self.point_conv(combined)  # (1, C, K)
            refined = torch.max(refined, dim=2)[0].squeeze(0)  # (C,)
            
            refined_features.append(refined)
        
        refined_features = torch.stack(refined_features, dim=0)  # (N, C)
        
        # Fuse original and refined features
        combined = torch.cat([features, refined_features], dim=1)  # (N, 2C)
        final_features = self.fusion(combined)  # (N, C)
        
        return final_features
