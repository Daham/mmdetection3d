"""
Adaptive Sparse Bridge - SIMPLIFIED VERSION

This module:
1. Does simple adaptive feature processing per voxel
2. Compatible with standard sparse convolution 
3. Fast and lightweight
"""

try:
    import torch
    import torch.nn as nn
    from typing import List
    from mmdet3d.registry import MODELS
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Import warning in adaptive_sparse_bridge: {e}")
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

if TORCH_AVAILABLE:
    @MODELS.register_module()
    class AdaptiveSparseBridge(nn.Module):
        """
        SIMPLIFIED adaptive voxel feature encoder.
        Just does basic adaptive processing - no complex networks.
        """
        
        def __init__(self,
                     base_voxel_size: List[float] = [0.5, 0.5, 0.5],
                     point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                     min_voxel_size: List[float] = [0.25, 0.25, 0.25],
                     max_voxel_size: List[float] = [1.0, 1.0, 1.0],
                     adaptation_method: str = 'learned',
                     max_points_per_voxel: int = 5,
                     in_channels: int = 4,
                     feat_channels: List[int] = [4],
                     learnable_adaptation: bool = True):
            super().__init__()
            
            self.base_voxel_size = base_voxel_size
            self.point_cloud_range = point_cloud_range
            self.min_voxel_size = min_voxel_size
            self.max_voxel_size = max_voxel_size
            self.max_points_per_voxel = max_points_per_voxel
            self.in_channels = in_channels
            self.feat_channels = feat_channels
            self.learnable_adaptation = learnable_adaptation
            
            # Simple adaptive network - just a small MLP
            if learnable_adaptation:
                self.adaptive_net = nn.Sequential(
                    nn.Linear(4, 16),  # point density -> adaptive weight
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )
            
            print(f"🎯 AdaptiveSparseBridge (SIMPLIFIED) initialized:")
            print(f"   - Base voxel size: {base_voxel_size}")
            print(f"   - Output channels: {feat_channels[-1]}")
            print(f"   - Learning: {learnable_adaptation}")

        def forward(self, features, num_points, coors):
            """
            SIMPLIFIED forward pass - just like HardSimpleVFE but with adaptive weighting.
            """
            batch_size, max_points, feat_dim = features.shape
            device = features.device
            
            # Simple adaptive processing per voxel
            processed_features = []
            
            for i in range(batch_size):
                n_pts = num_points[i]
                if n_pts > 0:
                    # Get points in this voxel
                    voxel_points = features[i, :n_pts, :self.feat_channels[-1]]
                    
                    if self.learnable_adaptation:
                        # Simple adaptive weighting based on point density
                        density = float(n_pts) / self.max_points_per_voxel
                        density_input = torch.tensor([density, density, density, density], 
                                                   device=device, dtype=torch.float32)
                        
                        try:
                            adaptive_weight = self.adaptive_net(density_input.unsqueeze(0)).squeeze()
                        except:
                            adaptive_weight = torch.tensor(0.5, device=device)
                        
                        # Apply adaptive weighting to the mean
                        voxel_mean = voxel_points.mean(dim=0)
                        adaptive_feature = voxel_mean * (0.5 + adaptive_weight * 0.5)
                    else:
                        # Just simple mean like HardSimpleVFE
                        adaptive_feature = voxel_points.mean(dim=0)
                    
                    processed_features.append(adaptive_feature)
                else:
                    # Empty voxel
                    processed_features.append(torch.zeros(self.feat_channels[-1], device=device))
            
            if processed_features:
                result = torch.stack(processed_features)
            else:
                result = torch.zeros(batch_size, self.feat_channels[-1], device=device)
            
            return result

else:
    class AdaptiveSparseBridge:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AdaptiveSparseBridge")
