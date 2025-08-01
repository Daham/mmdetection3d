# ADAPTIVE VOXELIZATION WITH SPARSE CONVOLUTION
# This is the ONLY config you need for adaptive voxel sizes + sparse convolution

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py', 
    '../_base_/schedules/cyclic-2e.py',
    '../_base_/default_runtime.py'
]

# Import the adaptive voxelization bridge
custom_imports = dict(
    imports=['mmdet3d.models.voxel_encoders.adaptive_sparse_bridge'],
    allow_failed_imports=False)

# This bridge:
# 1. Learns adaptive voxel sizes during training
# 2. Maps variable voxels to regular grid for sparse convolution
# 3. Handles the compatibility automatically

model = dict(
    voxel_encoder=dict(
        type='AdaptiveSparseBridge',
        base_voxel_size=[0.05, 0.05, 0.1],          # Regular grid base size
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        min_voxel_size=[0.025, 0.025, 0.05],        # Smallest adaptive voxel
        max_voxel_size=[0.2, 0.2, 0.4],             # Largest adaptive voxel  
        adaptation_method='learned',                 # Learn voxel sizes
        max_points_per_voxel=32,
        in_channels=4,
        feat_channels=[64],
        learnable_adaptation=True),                  # Enable learning
    
    # Standard sparse convolution works with the bridge output
    bbox_head=dict(num_classes=1))

# Training settings
train_cfg = dict(max_epochs=80, val_interval=10)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01))
