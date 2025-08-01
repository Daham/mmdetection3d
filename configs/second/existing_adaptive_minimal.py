# Ultra-Minimal Config Using Existing AdaptiveVFE
# This uses your existing working AdaptiveVFE without any new dependencies

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Use your existing adaptive VFE that already works
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.adaptive_vfe'],
    allow_failed_imports=False)

data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# MINIMAL: Only replace VFE with your existing AdaptiveVFE
model = dict(
    # Use your existing AdaptiveVFE that was working
    voxel_encoder=dict(
        type='AdaptiveVFE',
        in_channels=4,
        feat_channels=[4],  # Keep same as HardSimpleVFE
        with_distance=False,
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        base_sparse_shape=[41, 1600, 1408],
        adaptation_method='density',
        num_scales=3),
    
    # Keep everything else exactly the same
    bbox_head=dict(num_classes=1))

# Quick training for testing
train_cfg = dict(max_epochs=2, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01))
