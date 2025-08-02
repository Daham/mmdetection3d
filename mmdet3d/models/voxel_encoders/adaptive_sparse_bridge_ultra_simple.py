"""
Ultra-Simple Adaptive Bridge - Nearly identical to HardSimpleVFE with minimal adaptive features
"""

try:
    import torch
    import torch.nn as nn
    from mmdet3d.registry import MODELS
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

if TORCH_AVAILABLE:
    @MODELS.register_module()
    class AdaptiveSparseBridge(nn.Module):
        """
        Ultra-simple adaptive VFE - almost identical to HardSimpleVFE.
        """
        
        def __init__(self, 
                     num_features: int = 4,
                     learnable_adaptation: bool = False,  # Default to False for speed
                     **kwargs):  # Accept any other parameters but ignore them
            super().__init__()
            self.num_features = num_features
            self.learnable_adaptation = learnable_adaptation
            
            # Tiny adaptive network if enabled
            if learnable_adaptation:
                self.adaptive_net = nn.Linear(1, 1)  # Minimal network
            
            print(f"🎯 AdaptiveSparseBridge (ULTRA-SIMPLE) initialized:")
            print(f"   - Features: {num_features}")
            print(f"   - Adaptive learning: {learnable_adaptation}")

        def forward(self, features, num_points, coors):
            """
            Almost identical to HardSimpleVFE forward pass.
            """
            # Standard mean calculation like HardSimpleVFE
            points_mean = features[:, :, :self.num_features].sum(
                dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
            
            # Optional tiny adaptive modification
            if self.learnable_adaptation:
                try:
                    # Just a small learned scaling factor
                    density = (num_points.float() / features.size(1)).mean().unsqueeze(0)
                    scale = self.adaptive_net(density)
                    points_mean = points_mean * (0.9 + 0.2 * scale)  # Very conservative scaling
                except:
                    pass  # If anything fails, just use standard mean
            
            return points_mean.contiguous()

else:
    class AdaptiveSparseBridge:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")
