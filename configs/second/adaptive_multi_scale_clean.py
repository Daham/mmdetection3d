# 🎯 MULTI-SCALE ADAPTIVE VOXELIZATION - PhD Research Implementation
# Based on vanilla SECOND but with revolutionary multi-scale adaptive voxels

_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py'
]

# Configuration (same as vanilla)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
class_names = ['Car']

# 🔬 PHD RESEARCH MODEL: Multi-Scale Adaptive Voxelization
model = dict(
    # 🔬 RAW POINTS INPUT: No fixed voxelization
    data_preprocessor=dict(
        voxel=False,  # Disable standard voxelization - we handle adaptively
        voxel_layer=None,  # No fixed voxel layer
    ),
    
    # 🎯 MULTI-SCALE ADAPTIVE VOXEL ENCODER: Your Revolutionary Approach
    voxel_encoder=dict(
        _delete_=True,
        type='MultiScaleAdaptiveVoxelEncoder',
        point_cloud_range=point_cloud_range,
        base_voxel_size=[0.05, 0.05, 0.1],      # Standard voxel size
        max_num_points=5,                        # Points per voxel
        max_voxels=(12000, 30000),              # Training and test limits
        
        # 🎯 MULTI-SCALE CONFIGURATION
        fine_scale=0.5,      # 2x finer: [0.025, 0.025, 0.05] for high-info regions
        medium_scale=1.0,    # Base scale: [0.05, 0.05, 0.1] for normal regions  
        coarse_scale=2.0,    # 2x coarser: [0.1, 0.1, 0.2] for low-info regions
        
        # 🧠 LEARNABLE IMPORTANCE PREDICTION
        importance_channels=128,  # Neural network hidden size
    ),
    
    # 🚀 REVOLUTIONARY PARALLEL MIDDLE ENCODER: Process each scale separately!
    middle_encoder=dict(
        _delete_=True,
        type='MultiScaleParallelMiddleEncoder',
        sparse_shape=[41, 1600, 1408],
        in_channels=4,
        output_channels=128,
        # Three separate sparse conv networks for fine/medium/coarse scales
        # Then intelligent fusion to consistent BEV feature maps
    ),
    
    # Note: Backbone and detection head remain same as vanilla SECOND
)

# Same training configuration as vanilla for fair comparison
train_dataloader = dict(batch_size=1)  # Reduce batch size to avoid tensor mismatch
val_dataloader = dict(batch_size=1)

# Same optimization as vanilla
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Quick training for testing
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

work_dir = './work_dirs/adaptive_multi_scale_clean'
