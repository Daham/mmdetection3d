# Absolute Minimal Adaptive Config
# Strategy: Only change VFE to adaptive, keep middle encoder as standard SparseEncoder

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Minimal custom imports - only adaptive VFE
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe'],
    allow_failed_imports=False)

data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# ULTRA-MINIMAL: Only replace VFE, keep everything else standard
model = dict(
    # Only change: Adaptive VFE that outputs same channels as HardSimpleVFE
    voxel_encoder=dict(
        type='EnhancedAdaptiveVFE',
        in_channels=4,
        feat_channels=[4],  # Output exactly 4 channels like HardSimpleVFE
        with_distance=False,
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        base_sparse_shape=[41, 1600, 1408],
        adaptation_method='density',  # Simple density-based adaptation
        num_scales=1,  # Single scale - just adaptive voxel sizing
        provide_multi_res_info=False),  # No multi-res info needed
    
    # Everything else stays exactly the same (including standard SparseEncoder)
    bbox_head=dict(num_classes=1))

# Minimal training
train_cfg = dict(max_epochs=2, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01))
