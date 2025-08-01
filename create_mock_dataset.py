#!/usr/bin/env python3
"""
Create minimal mock KITTI dataset for testing adaptive components
Run this before using the test configuration.
"""

import os
import pickle
import numpy as np

def create_mock_kitti_dataset():
    """Create a minimal mock KITTI dataset for testing."""
    mock_data_root = '/tmp/mock_kitti/'
    
    # Create directory structure
    os.makedirs(mock_data_root, exist_ok=True)
    velodyne_dir = os.path.join(mock_data_root, 'training', 'velodyne_reduced')
    os.makedirs(velodyne_dir, exist_ok=True)
    
    # Create a minimal info file with one sample
    mock_infos = [{
        'sample_idx': 0,
        'point_cloud': {
            'lidar_path': 'training/velodyne_reduced/000000.bin',
            'num_features': 4
        },
        'annos': {
            'name': ['Car'],
            'location': [[5.0, 0.0, 0.0]],  # x, y, z
            'dimensions': [[2.0, 1.5, 4.0]],  # h, w, l
            'rotation_y': [0.0],
            'gt_boxes_lidar': [[5.0, 0.0, 0.0, 4.0, 1.5, 2.0, 0.0]],  # x, y, z, l, w, h, yaw
            'difficulty': [0],
            'index': [0]
        }
    }]
    
    # Save the info file
    info_path = os.path.join(mock_data_root, 'kitti_infos_train.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(mock_infos, f)
    
    # Create a minimal point cloud file
    # Generate points around the car bbox for realistic testing
    num_points = 1000
    
    # Car points (inside the bbox)
    car_points = np.random.uniform(
        low=[4.0, -0.75, -1.0], 
        high=[6.0, 0.75, 1.0], 
        size=(200, 3)
    )
    
    # Background points
    bg_points = np.random.uniform(
        low=[-10, -10, -2], 
        high=[20, 10, 2], 
        size=(num_points - 200, 3)
    )
    
    # Combine points
    points_xyz = np.vstack([car_points, bg_points])
    
    # Add intensity values
    intensities = np.random.uniform(0, 1, size=(num_points, 1))
    
    # Create final point cloud (x, y, z, intensity)
    mock_points = np.hstack([points_xyz, intensities]).astype(np.float32)
    
    # Save point cloud
    point_path = os.path.join(velodyne_dir, '000000.bin')
    mock_points.tofile(point_path)
    
    print(f"✅ Mock KITTI dataset created at {mock_data_root}")
    print(f"   - Info file: {info_path}")
    print(f"   - Point cloud: {point_path}")
    print(f"   - Points shape: {mock_points.shape}")
    print(f"   - Sample annotation: {mock_infos[0]['annos']}")

if __name__ == "__main__":
    create_mock_kitti_dataset()
