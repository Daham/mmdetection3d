# Ultra-Minimal Multi-Resolution Adaptive Config
# Strategy: Change ONLY the voxel encoder and middle encoder, keep everything else identical

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Minimal custom imports - only what we need
custom_imports = dict(
    imports=[
        'mmdet3d.models.voxel_encoders.enhanced_adaptive_vfe',
        'mmdet3d.models.middle_encoders.multi_resolution_sparse_encoder'
    ],
    allow_failed_imports=False)

# ONLY override what's absolutely necessary
data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Minimal model override - only VFE and middle encoder
model = dict(
    # Step 1: Replace VFE with minimal adaptive version
    voxel_encoder=dict(
        type='EnhancedAdaptiveVFE',
        in_channels=4,
        feat_channels=[4, 8],  # Minimal increase: 4->8 channels
        with_distance=False,   # Keep simple, no distance features
        voxel_size=(0.05, 0.05, 0.1),  # Same as base
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),  # Same as base
        base_sparse_shape=[41, 1600, 1408],  # Same as base
        adaptation_method='density',  # Simplest method
        num_scales=2,  # Only 2 scales instead of 3
        provide_multi_res_info=True),
    
    # Step 2: Replace middle encoder with minimal multi-resolution
    middle_encoder=dict(
        type='MultiResolutionSparseEncoder',
        base_voxel_size=[0.05, 0.05, 0.1],  # Same as base
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],  # Same as base
        resolution_levels=[1.0, 2.0],  # Only 2 levels: base + coarse
        in_channels=8,   # Match VFE output
        out_channels=64, # Same as original SECOND middle encoder
        assignment_threshold=0.2,  # Higher threshold = simpler assignment
        fusion_method='weighted_concat'),   # Simpler than attention
    
    # Keep everything else exactly the same as base model
    # (backbone, neck, head all inherit from base)
    
    # Only override for single class
    bbox_head=dict(num_classes=1))

# Minimal training override
train_cfg = dict(max_epochs=2, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01))
