#!/usr/bin/env python3
"""
Quick demonstration of how ScaleNet identifies information and assigns 
points to appropriate voxel sizes from 10 available scales.
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/home/daham/mmdetection_project/mmdetection3d')

from mmdet3d.models.voxel_encoders.importance_guided_multi_scale_vfe import ScaleNet

def demonstrate_scalenet_10_scales():
    """Show how ScaleNet works with 10 voxel sizes."""
    
    print("🎯 ScaleNet with 10 Voxel Sizes - How It Works")
    print("=" * 60)
    
    # Create ScaleNet with 10 scales
    scale_net = ScaleNet(
        in_channels=4,  # x, y, z, intensity
        hidden_dims=[64, 32],
        num_scales=10,  # 🚀 10 different voxel sizes!
        temperature=2.0
    )
    
    # Show the 10 voxel sizes it generates
    print(f"\n📏 Available 10 Voxel Sizes:")
    scales_info = scale_net.get_scale_info()
    for i, scale in enumerate(scales_info['scales']):
        print(f"  Scale {i}: {scale:.3f}m ({scale*100:.1f}cm)")
    
    print(f"\n📊 Scale Range: {scales_info['scale_range']}")
    
    # Create sample points representing different object types
    print(f"\n🔍 Point Analysis - How ScaleNet Identifies Information:")
    
    # Different point scenarios
    scenarios = [
        {
            'name': 'Small Details (e.g., pedestrian features)',
            'points': torch.tensor([
                [1.0, 2.0, 0.5, 0.8],   # Close, low height
                [1.1, 2.1, 0.6, 0.7],   # Close, low height  
                [1.2, 1.9, 0.4, 0.9],   # Close, low height
            ])
        },
        {
            'name': 'Medium Objects (e.g., cars)',
            'points': torch.tensor([
                [5.0, 8.0, 1.2, 0.6],   # Medium distance, car height
                [6.0, 9.0, 1.4, 0.5],   # Medium distance, car height
                [7.0, 10.0, 1.1, 0.7],  # Medium distance, car height
            ])
        },
        {
            'name': 'Large Context (e.g., buildings, far objects)',
            'points': torch.tensor([
                [20.0, 30.0, 5.0, 0.3], # Far, high
                [25.0, 35.0, 6.0, 0.2], # Far, high
                [30.0, 40.0, 4.5, 0.4], # Far, high
            ])
        }
    ]
    
    scale_net.eval()  # Set to evaluation mode for consistent results
    
    for scenario in scenarios:
        print(f"\n🎯 {scenario['name']}:")
        points = scenario['points']
        
        # Get scale predictions
        with torch.no_grad():
            scale_assignment, predicted_scales = scale_net(points, training=False)
        
        # Show results for each point
        for i, (point, pred_scale) in enumerate(zip(points, predicted_scales)):
            # Find which scale was selected (highest probability)
            selected_scale_id = torch.argmax(scale_assignment[i]).item()
            selected_voxel_size = scales_info['scales'][selected_scale_id]
            confidence = scale_assignment[i][selected_scale_id].item()
            
            print(f"  Point {i+1}: xyz=({point[0]:.1f},{point[1]:.1f},{point[2]:.1f})")
            print(f"    → Selected Scale {selected_scale_id}: {selected_voxel_size:.3f}m ({selected_voxel_size*100:.1f}cm)")
            print(f"    → Confidence: {confidence:.2f}")
            print(f"    → Predicted Scale: {pred_scale:.3f}m")
    
    # Show probability distribution for one example
    print(f"\n📈 Detailed Scale Assignment for Small Detail Point:")
    test_point = torch.tensor([[1.0, 2.0, 0.5, 0.8]])
    
    with torch.no_grad():
        scale_assignment, _ = scale_net(test_point, training=False)
        probs = scale_assignment[0]  # Get probabilities for first point
    
    print(f"  Point: xyz=(1.0, 2.0, 0.5), intensity=0.8")
    for i, (scale, prob) in enumerate(zip(scales_info['scales'], probs)):
        bar = "█" * int(prob * 20)  # Simple bar chart
        print(f"  Scale {i} ({scale:.3f}m): {prob:.3f} {bar}")
    
    print(f"\n✅ How ScaleNet Works:")
    print(f"  1. 🧠 Analyzes point features (x,y,z,intensity)")
    print(f"  2. 🎯 Predicts optimal voxel size from 10 options")
    print(f"  3. 📊 Assigns probability to each scale (soft assignment)")
    print(f"  4. 🔄 Points go to their identified optimal voxel size")
    print(f"  5. 🚀 Each scale processes points optimally")

if __name__ == "__main__":
    demonstrate_scalenet_10_scales()
