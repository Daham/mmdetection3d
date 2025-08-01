# Adaptive-to-Regular Bridge Config
# This solves the sparse convolution compatibility problem

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/schedules/cyclic-2e.py', 
    '../_base_/default_runtime.py'
]

# Import the adaptive-to-regular bridge
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.adaptive_to_regular_bridge'],
    allow_failed_imports=False)

data_root = '/home/daham/mmdetection_project/dataset/KITTI/'

# Use Adaptive-to-Regular Bridge (adaptive voxelization + sparse conv compatibility)
model = dict(
    # Bridge that performs adaptive voxelization and maps to regular grid
    voxel_encoder=dict(
        type='AdaptiveToRegularBridge',
        base_voxel_size=[0.05, 0.05, 0.1],        # Base voxel size for regular grid
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        min_voxel_size=[0.025, 0.025, 0.05],      # Adaptive range: fine
        max_voxel_size=[0.2, 0.2, 0.4],           # Adaptive range: coarse
        adaptation_method='density',               # How to adapt sizes
        grid_resolution=32,                       # Adaptation grid resolution
        max_points_per_voxel=32,
        in_channels=4,
        feat_channels=[64],
        conflict_resolution='weighted_average',   # How to handle mapping conflicts
        regular_grid_size=[41, 1600, 1408]),     # Output grid size (same as base SECOND)
    
    # Standard SparseEncoder works seamlessly with regular grid output
    bbox_head=dict(num_classes=1))

# Training settings
train_cfg = dict(max_epochs=2, val_interval=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01))
