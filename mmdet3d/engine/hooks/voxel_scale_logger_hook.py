"""
📊 VOXEL SCALE LOGGING HOOK FOR REVIEWER PROOF

Custom MMEngine hook that logs voxel scale parameters (θ) and their gradients
during training. This provides evidence for learnable voxel scales.

Usage:
    In config, add:
    custom_hooks = [dict(type='VoxelScaleLoggerHook', interval=20)]
"""

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class VoxelScaleLoggerHook(Hook):
    """
    Hook to log learnable voxel scale parameters during training.
    
    Logs:
    - θ₁, θ₂, θ₃: Current voxel scale values (in cm for readability)
    - ||∇θ||: Gradient norm (evidence of gradient flow)
    - τ: Gumbel-Softmax temperature
    """
    
    def __init__(self, interval: int = 20, log_grad: bool = True):
        self.interval = interval
        self.log_grad = log_grad
        self._voxel_encoder = None
        
    def _get_voxel_encoder(self, runner):
        """Get voxel encoder from model."""
        if self._voxel_encoder is None:
            model = runner.model
            # Handle DistributedDataParallel wrapper
            if hasattr(model, 'module'):
                model = model.module
            # Access voxel_encoder
            if hasattr(model, 'voxel_encoder'):
                self._voxel_encoder = model.voxel_encoder
        return self._voxel_encoder
    
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        """Log voxel scale values after each training iteration."""
        if (runner.iter + 1) % self.interval != 0:
            return
            
        voxel_encoder = self._get_voxel_encoder(runner)
        if voxel_encoder is None:
            return
            
        # Get scale_net with learnable voxel_scales
        if not hasattr(voxel_encoder, 'scale_net'):
            return
            
        scale_net = voxel_encoder.scale_net
        if not hasattr(scale_net, 'voxel_scales'):
            return
            
        voxel_scales = scale_net.voxel_scales
        
        # Convert to cm for readability and log
        scales_cm = (voxel_scales.detach().cpu() * 100).tolist()
        
        # Prepare log dict
        log_dict = {
            'θ₁_cm': round(scales_cm[0], 2),
            'θ₂_cm': round(scales_cm[1], 2),
            'θ₃_cm': round(scales_cm[2], 2) if len(scales_cm) > 2 else 0,
        }
        
        # Add gradient info if available
        if self.log_grad and voxel_scales.grad is not None:
            grad_norm = voxel_scales.grad.norm().item()
            log_dict['||∇θ||'] = round(grad_norm, 2)
        
        # Add temperature
        if hasattr(scale_net, 'temperature'):
            temp = scale_net.temperature.item() if torch.is_tensor(scale_net.temperature) else scale_net.temperature
            log_dict['τ'] = round(temp, 3)
        
        # Log to runner's message hub
        runner.message_hub.update_scalars(log_dict)


@HOOKS.register_module()
class VoxelScaleCSVLoggerHook(Hook):
    """
    Hook to log voxel scale parameters to CSV file for plotting.
    """
    
    def __init__(self, 
                 log_file: str = '/tmp/voxel_scale_gradients.csv',
                 interval: int = 1):
        self.log_file = log_file
        self.interval = interval
        self._voxel_encoder = None
        self._initialized = False
        
    def _init_log_file(self):
        """Initialize CSV file with header."""
        if not self._initialized:
            with open(self.log_file, 'w') as f:
                f.write('iteration,theta_0,theta_1,theta_2,grad_0,grad_1,grad_2,grad_norm,temperature\n')
            self._initialized = True
            print(f"📊 Voxel scale CSV logging initialized: {self.log_file}")
    
    def _get_voxel_encoder(self, runner):
        """Get voxel encoder from model."""
        if self._voxel_encoder is None:
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            if hasattr(model, 'voxel_encoder'):
                self._voxel_encoder = model.voxel_encoder
        return self._voxel_encoder
    
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        """Log to CSV after each iteration."""
        if (runner.iter + 1) % self.interval != 0:
            return
            
        self._init_log_file()
        
        voxel_encoder = self._get_voxel_encoder(runner)
        if voxel_encoder is None or not hasattr(voxel_encoder, 'scale_net'):
            return
            
        scale_net = voxel_encoder.scale_net
        if not hasattr(scale_net, 'voxel_scales'):
            return
            
        voxel_scales = scale_net.voxel_scales
        scales = voxel_scales.detach().cpu().numpy()
        
        # Get gradients
        if voxel_scales.grad is not None:
            grads = voxel_scales.grad.detach().cpu().numpy()
            grad_norm = float(voxel_scales.grad.norm().item())
        else:
            grads = [0.0] * len(scales)
            grad_norm = 0.0
        
        # Get temperature
        temp = 0.0
        if hasattr(scale_net, 'temperature'):
            temp = scale_net.temperature.item() if torch.is_tensor(scale_net.temperature) else scale_net.temperature
        
        # Write to CSV
        with open(self.log_file, 'a') as f:
            row = f"{runner.iter + 1},{scales[0]:.6f},{scales[1]:.6f},{scales[2]:.6f},"
            row += f"{grads[0]:.4f},{grads[1]:.4f},{grads[2]:.4f},{grad_norm:.4f},{temp:.4f}\n"
            f.write(row)
