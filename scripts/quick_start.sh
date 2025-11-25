#!/bin/bash
# Quick start script for testing the adaptive octree implementation

set -e  # Exit on error

echo "🚀 ADAPTIVE OCTREE QUICK START"
echo "================================"
echo ""

# Check if in correct directory
if [ ! -f "setup.py" ]; then
    echo "❌ Please run this script from the mmdetection3d root directory"
    exit 1
fi

echo "📂 Current directory: $(pwd)"
echo ""

# Step 1: Check environment
echo "1️⃣  Checking Python environment..."
if command -v python &> /dev/null; then
    python_version=$(python --version)
    echo "✅ Python found: $python_version"
else
    echo "❌ Python not found. Please activate your environment:"
    echo "   source ~/mmdet_env/bin/activate"
    exit 1
fi

# Step 2: Test imports
echo ""
echo "2️⃣  Testing imports..."

# Test PyTorch
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || {
    echo "❌ PyTorch import failed"
    exit 1
}

# Test CUDA
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" || {
    echo "⚠️  CUDA check failed"
}

# Test mmdet3d
python -c "import mmdet3d; print(f'✅ mmdet3d {mmdet3d.__version__}')" || {
    echo "❌ mmdet3d import failed"
    exit 1
}

# Step 3: Check dataset
echo ""
echo "3️⃣  Checking KITTI dataset..."
if [ -d "data/kitti" ]; then
    echo "✅ KITTI dataset found"
    train_files=$(find data/kitti -name "*train*.pkl" 2>/dev/null | wc -l)
    val_files=$(find data/kitti -name "*val*.pkl" 2>/dev/null | wc -l)
    echo "   Train info files: $train_files"
    echo "   Val info files: $val_files"
else
    echo "❌ KITTI dataset not found at data/kitti"
    echo "   Please set up KITTI dataset first"
    exit 1
fi

# Step 4: Verify implementation files
echo ""
echo "4️⃣  Verifying implementation files..."
bash scripts/verify_implementation.sh | grep -E "(✅|❌)" | head -n 20

# Step 5: Quick syntax check
echo ""
echo "5️⃣  Checking Python syntax..."
echo "   Checking octree_node.py..."
python -m py_compile mmdet3d/models/voxel_encoders/octree/octree_node.py 2>/dev/null && echo "   ✅ octree_node.py" || echo "   ❌ octree_node.py"

echo "   Checking octree_builder.py..."
python -m py_compile mmdet3d/models/voxel_encoders/octree/octree_builder.py 2>/dev/null && echo "   ✅ octree_builder.py" || echo "   ❌ octree_builder.py"

echo "   Checking adaptive_octree_vfe.py..."
python -m py_compile mmdet3d/models/voxel_encoders/octree/adaptive_octree_vfe.py 2>/dev/null && echo "   ✅ adaptive_octree_vfe.py" || echo "   ❌ adaptive_octree_vfe.py"

echo "   Checking adaptive_point_backbone.py..."
python -m py_compile mmdet3d/models/backbones/adaptive/adaptive_point_backbone.py 2>/dev/null && echo "   ✅ adaptive_point_backbone.py" || echo "   ❌ adaptive_point_backbone.py"

echo "   Checking adaptive_to_fixed_grid.py..."
python -m py_compile mmdet3d/models/middle_encoders/adaptive/adaptive_to_fixed_grid.py 2>/dev/null && echo "   ✅ adaptive_to_fixed_grid.py" || echo "   ❌ adaptive_to_fixed_grid.py"

# Step 6: Check configs
echo ""
echo "6️⃣  Checking configuration files..."
for config in configs/adaptive_voxelnet/*.py; do
    if [ -f "$config" ]; then
        basename=$(basename "$config")
        python -m py_compile "$config" 2>/dev/null && echo "   ✅ $basename" || echo "   ❌ $basename"
    fi
done

echo ""
echo "================================"
echo "✅ Quick start checks complete!"
echo "================================"
echo ""

echo "📋 NEXT STEPS:"
echo ""
echo "Option 1: Test with small dataset (RECOMMENDED FIRST)"
echo "------------------------------------------------------"
echo "python tools/train.py configs/adaptive_voxelnet/single_scale_0.1m.py \\"
echo "    --cfg-options train_dataloader.dataset.indices=100 \\"
echo "    --cfg-options train_cfg.max_epochs=1"
echo ""

echo "Option 2: Run baseline comparison (full training)"
echo "--------------------------------------------------"
echo "python tools/experiments/run_baseline_comparison.py"
echo ""

echo "Option 3: Train adaptive octree only"
echo "-------------------------------------"
echo "python tools/train.py configs/adaptive_voxelnet/adaptive_octree.py"
echo ""

echo "📖 For more details, read: IMPLEMENTATION_SUMMARY.md"
echo ""
