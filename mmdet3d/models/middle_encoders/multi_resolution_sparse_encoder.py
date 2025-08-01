# Multi-Resolution Sparse Convolution for Adaptive Voxelization
# This implements true variable voxel size processing

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from mmdet3d.registry import MODELS

try:
    from spconv.pytorch import SparseConvTensor, SparseConv3d, SubMConv3d
    from .sparse_encoder import SparseEncoder
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class SparseConvTensor:
        pass
    class SparseConv3d:
        pass
    class SubMConv3d:
        pass
    class SparseEncoder:
        pass


@MODELS.register_module()
class MultiResolutionSparseEncoder(nn.Module):
    """
    Multi-Resolution Sparse Encoder that can handle multiple voxel sizes simultaneously.
    
    This allows true adaptive voxelization by processing different regions at 
    different resolutions and fusing the results.
    
    Note: Requires spconv to be installed.
    """
    
    def __init__(self, 
                 base_voxel_size: List[float] = [0.05, 0.05, 0.1],
                 point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                 resolution_levels: List[float] = [0.5, 1.0, 2.0],  # Scale factors
                 in_channels: int = 4,
                 out_channels: int = 64,
                 assignment_threshold: float = 0.1,
                 fusion_method: str = 'weighted_concat'):  # 'weighted_concat', 'attention', 'simple'
        super().__init__()
        
        # Check if spconv is available
        if not SPCONV_AVAILABLE:
            raise ImportError(
                "spconv is required for MultiResolutionSparseEncoder. "
                "Please install spconv: pip install spconv-cu118"
            )
        
        self.base_voxel_size = base_voxel_size
        self.point_cloud_range = point_cloud_range
        self.resolution_levels = resolution_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.assignment_threshold = assignment_threshold
        self.fusion_method = fusion_method
        
        # Create sparse grids for each resolution level
        self.sparse_grids = nn.ModuleDict()
        self.voxel_sizes = {}
        self.sparse_shapes = {}
        
        for i, scale in enumerate(resolution_levels):
            level_name = f"level_{i}"
            
            # Compute voxel size for this level
            voxel_size = [base_voxel_size[j] * scale for j in range(3)]
            self.voxel_sizes[level_name] = voxel_size
            
            # Compute sparse shape for this level
            sparse_shape = [
                int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # Z
                int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # Y  
                int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])   # X
            ]
            self.sparse_shapes[level_name] = sparse_shape
            
            print(f"Resolution Level {i} (scale={scale}): voxel_size={voxel_size}, sparse_shape={sparse_shape}")
            
            # Create sparse convolution layers for this resolution
            self.sparse_grids[level_name] = self._build_sparse_layers(
                in_channels, out_channels
            )
        
        # Resolution assignment network
        self.resolution_predictor = nn.Sequential(
            nn.Linear(in_channels + 1, 64),  # +1 for point count
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, len(resolution_levels)),
            nn.Softmax(dim=-1)  # Soft assignment weights
        )
        
        # Feature fusion network
        if fusion_method == 'weighted_concat':
            fusion_in_dim = out_channels * len(resolution_levels)
        elif fusion_method == 'attention':
            fusion_in_dim = out_channels
        else:  # simple
            fusion_in_dim = out_channels
            
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # Attention mechanism for fusion
        if fusion_method == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(out_channels * len(resolution_levels), out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, len(resolution_levels)),
                nn.Softmax(dim=-1)
            )
        
    def _build_sparse_layers(self, in_channels: int, out_channels: int):
        """Build sparse convolution layers for one resolution level."""
        return nn.Sequential(
            SparseConv3d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            SubMConv3d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def assign_voxels_to_resolutions(self, 
                                   voxel_features: torch.Tensor,
                                   voxel_coords: torch.Tensor,
                                   adaptive_info: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Assign each voxel to appropriate resolution level(s).
        
        Args:
            voxel_features: [N, C] voxel features
            voxel_coords: [N, 4] voxel coordinates (batch, z, y, x)
            adaptive_info: Optional dict with adaptive information
            
        Returns:
            Dict mapping level names to {features, coords, weights}
        """
        N = voxel_features.shape[0]
        
        # Get or compute density information
        if adaptive_info and 'num_points' in adaptive_info:
            num_points = adaptive_info['num_points']
        else:
            # Fallback: assume uniform density
            num_points = torch.ones(N, device=voxel_features.device) * 5
        
        # Predict resolution assignment weights
        voxel_density = num_points.float().unsqueeze(1) / (num_points.max() + 1e-6)
        predictor_input = torch.cat([voxel_features, voxel_density], dim=1)
        assignment_weights = self.resolution_predictor(predictor_input)  # [N, num_levels]
        
        assignments = {}
        
        for i, level_name in enumerate(self.sparse_grids.keys()):
            # Get assignment weights for this level
            level_weights = assignment_weights[:, i]  # [N]
            
            # Use threshold-based assignment
            mask = level_weights > self.assignment_threshold
            
            if mask.sum() > 0:
                # Re-voxelize coordinates for this resolution level
                level_coords = self._revoxelize_coordinates(
                    voxel_coords[mask], 
                    level_name
                )
                
                assignments[level_name] = {
                    'features': voxel_features[mask],
                    'coords': level_coords,
                    'weights': level_weights[mask],
                    'original_indices': torch.where(mask)[0]
                }
            else:
                # Empty assignment for this level
                assignments[level_name] = None
                
        return assignments, assignment_weights
    
    def _revoxelize_coordinates(self, 
                               original_coords: torch.Tensor, 
                               level_name: str) -> torch.Tensor:
        """
        Convert coordinates from base voxel size to target voxel size.
        """
        target_voxel_size = self.voxel_sizes[level_name]
        batch_idx = original_coords[:, 0]  # Keep batch index unchanged
        
        # Convert voxel indices back to world coordinates
        base_voxel_size = self.base_voxel_size
        world_coords = torch.zeros_like(original_coords[:, 1:].float())
        
        world_coords[:, 0] = (original_coords[:, 3].float() + 0.5) * base_voxel_size[0] + self.point_cloud_range[0]  # X
        world_coords[:, 1] = (original_coords[:, 2].float() + 0.5) * base_voxel_size[1] + self.point_cloud_range[1]  # Y  
        world_coords[:, 2] = (original_coords[:, 1].float() + 0.5) * base_voxel_size[2] + self.point_cloud_range[2]  # Z
        
        # Convert to target voxel grid
        target_coords = torch.zeros_like(original_coords)
        target_coords[:, 0] = batch_idx  # Batch index
        target_coords[:, 1] = torch.floor((world_coords[:, 2] - self.point_cloud_range[2]) / target_voxel_size[2]).long()  # Z
        target_coords[:, 2] = torch.floor((world_coords[:, 1] - self.point_cloud_range[1]) / target_voxel_size[1]).long()  # Y
        target_coords[:, 3] = torch.floor((world_coords[:, 0] - self.point_cloud_range[0]) / target_voxel_size[0]).long()  # X
        
        # Clamp to valid range
        sparse_shape = self.sparse_shapes[level_name]
        target_coords[:, 1] = torch.clamp(target_coords[:, 1], 0, sparse_shape[0] - 1)  # Z
        target_coords[:, 2] = torch.clamp(target_coords[:, 2], 0, sparse_shape[1] - 1)  # Y
        target_coords[:, 3] = torch.clamp(target_coords[:, 3], 0, sparse_shape[2] - 1)  # X
            
        return target_coords.int()
    
    def process_multi_resolution(self, 
                               assignments: Dict[str, Dict],
                               batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Process each resolution level with its dedicated sparse convolution.
        """
        level_outputs = {}
        
        for level_name, assignment in assignments.items():
            if assignment is None:
                level_outputs[level_name] = None
                continue
                
            try:
                # Create sparse tensor for this resolution level
                sparse_shape = self.sparse_shapes[level_name]
                sparse_tensor = SparseConvTensor(
                    assignment['features'],
                    assignment['coords'], 
                    sparse_shape,
                    batch_size
                )
                
                # Process through sparse convolution layers
                processed_tensor = self.sparse_grids[level_name](sparse_tensor)
                
                level_outputs[level_name] = {
                    'features': processed_tensor.features,
                    'coords': processed_tensor.indices,
                    'weights': assignment['weights'],
                    'original_indices': assignment['original_indices']
                }
            except Exception as e:
                print(f"Error processing level {level_name}: {e}")
                level_outputs[level_name] = None
                
        return level_outputs
    
    def fuse_multi_resolution_features(self, 
                                     level_outputs: Dict[str, torch.Tensor],
                                     assignment_weights: torch.Tensor,
                                     original_voxel_count: int) -> torch.Tensor:
        """
        Fuse features from different resolution levels back to unified representation.
        """
        device = next(self.parameters()).device
        
        if self.fusion_method == 'weighted_concat':
            # Initialize output feature tensor for concatenation
            fused_features = torch.zeros(
                original_voxel_count, 
                self.out_channels * len(self.resolution_levels),
                device=device
            )
            
            # Aggregate features from each level
            for i, (level_name, output) in enumerate(level_outputs.items()):
                if output is None:
                    continue
                    
                # Weight features by assignment confidence
                weighted_features = output['features'] * output['weights'].unsqueeze(1)
                
                # Place back in original positions
                start_idx = i * self.out_channels
                end_idx = (i + 1) * self.out_channels
                fused_features[output['original_indices'], start_idx:end_idx] = weighted_features
            
        elif self.fusion_method == 'attention':
            # Use attention mechanism for fusion
            level_features = []
            indices_map = torch.zeros(original_voxel_count, dtype=torch.long, device=device) - 1
            
            for i, (level_name, output) in enumerate(level_outputs.items()):
                if output is None:
                    level_features.append(torch.zeros(original_voxel_count, self.out_channels, device=device))
                else:
                    level_feat = torch.zeros(original_voxel_count, self.out_channels, device=device)
                    level_feat[output['original_indices']] = output['features']
                    level_features.append(level_feat)
            
            # Concatenate all level features
            all_features = torch.cat(level_features, dim=1)  # [N, out_channels * num_levels]
            
            # Compute attention weights
            attention_weights = self.attention_weights(all_features)  # [N, num_levels]
            
            # Apply attention
            fused_features = torch.zeros(original_voxel_count, self.out_channels, device=device)
            for i, level_feat in enumerate(level_features):
                fused_features += level_feat * attention_weights[:, i:i+1]
                
        else:  # simple averaging
            fused_features = torch.zeros(original_voxel_count, self.out_channels, device=device)
            total_weights = torch.zeros(original_voxel_count, 1, device=device)
            
            for level_name, output in level_outputs.items():
                if output is None:
                    continue
                weighted_features = output['features'] * output['weights'].unsqueeze(1)
                fused_features[output['original_indices']] += weighted_features
                total_weights[output['original_indices']] += output['weights'].unsqueeze(1)
            
            # Normalize by total weights
            fused_features = fused_features / (total_weights + 1e-6)
        
        # Final fusion through MLP
        final_features = self.feature_fusion(fused_features)
        
        return final_features
    
    def forward(self, 
                voxel_features: torch.Tensor,
                voxel_coords: torch.Tensor, 
                batch_size: int,
                adaptive_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Main forward function for multi-resolution sparse convolution.
        
        Args:
            voxel_features: [N, in_channels] input voxel features
            voxel_coords: [N, 4] voxel coordinates 
            batch_size: Batch size
            adaptive_info: Optional dict with adaptive information
            
        Returns:
            [N, out_channels] processed features
        """
        if self.training and torch.rand(1).item() < 0.1:
            print(f"MultiResolutionSparseEncoder: Processing {voxel_features.shape[0]} voxels")
        
        # Step 1: Assign voxels to resolution levels
        assignments, assignment_weights = self.assign_voxels_to_resolutions(
            voxel_features, voxel_coords, adaptive_info
        )
        
        # Step 2: Process each resolution level
        level_outputs = self.process_multi_resolution(assignments, batch_size)
        
        # Step 3: Fuse multi-resolution features
        fused_features = self.fuse_multi_resolution_features(
            level_outputs, assignment_weights, voxel_features.shape[0]
        )
        
        if self.training and torch.rand(1).item() < 0.1:
            active_levels = sum(1 for v in level_outputs.values() if v is not None)
            print(f"Active resolution levels: {active_levels}/{len(self.resolution_levels)}")
        
        return fused_features
