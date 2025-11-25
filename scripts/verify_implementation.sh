#!/bin/bash
# Quick checklist script to verify all files were created successfully

echo "🔍 Verifying Adaptive Octree Implementation Files..."
echo ""

# Track status
all_good=true

# Core implementation files
files=(
    # Octree infrastructure
    "mmdet3d/models/voxel_encoders/octree/__init__.py"
    "mmdet3d/models/voxel_encoders/octree/octree_node.py"
    "mmdet3d/models/voxel_encoders/octree/octree_builder.py"
    "mmdet3d/models/voxel_encoders/octree/adaptive_octree_vfe.py"
    
    # Adaptive backbone
    "mmdet3d/models/backbones/adaptive/__init__.py"
    "mmdet3d/models/backbones/adaptive/adaptive_point_backbone.py"
    
    # Middle encoder
    "mmdet3d/models/middle_encoders/adaptive/__init__.py"
    "mmdet3d/models/middle_encoders/adaptive/adaptive_to_fixed_grid.py"
    
    # Configuration files
    "configs/adaptive_voxelnet/README.md"
    "configs/adaptive_voxelnet/single_scale_0.1m.py"
    "configs/adaptive_voxelnet/multi_scale_fixed.py"
    "configs/adaptive_voxelnet/multi_scale_learnable_fusion.py"
    "configs/adaptive_voxelnet/adaptive_octree.py"
    
    # Tools
    "tools/experiments/run_baseline_comparison.py"
    "tools/analysis_tools/visualize_octree.py"
    
    # Scripts
    "scripts/clean_uncommitted.sh"
    
    # Documentation
    "IMPLEMENTATION_SUMMARY.md"
)

echo "Checking ${#files[@]} files..."
echo ""

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -l < "$file" 2>/dev/null || echo "0")
        echo "✅ $file ($size lines)"
    else
        echo "❌ $file - NOT FOUND"
        all_good=false
    fi
done

echo ""
echo "---"
echo ""

if [ "$all_good" = true ]; then
    echo "✅ All files created successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Register modules in mmdet3d/models/__init__.py"
    echo "2. Test imports: python -c 'from mmdet3d.models import AdaptiveOctreeVFE'"
    echo "3. Quick test: python tools/train.py configs/adaptive_voxelnet/adaptive_octree.py --cfg-options train_dataloader.dataset.indices=10"
    echo "4. Full training: python tools/train.py configs/adaptive_voxelnet/adaptive_octree.py"
    echo ""
    echo "📖 Read IMPLEMENTATION_SUMMARY.md for complete guide!"
else
    echo "❌ Some files are missing. Please check the output above."
    exit 1
fi
