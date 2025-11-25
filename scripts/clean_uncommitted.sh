#!/bin/bash
# Clean uncommitted files from the project

echo "🧹 Cleaning uncommitted files from MMDetection3D project..."

# Navigate to project root
cd /home/daham/mmdetection_project/mmdetection3d

# Show what will be deleted (dry run)
echo "Files to be removed:"
git status --short

# Ask for confirmation
read -p "⚠️  Proceed with deletion? This will remove all uncommitted changes (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Cleaning..."
    
    # Remove untracked files and directories
    git clean -fdx
    
    # Reset any modified files
    git reset --hard HEAD
    
    # Remove work directories
    rm -rf work_dirs/
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    
    # Remove temporary files
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    rm -rf *.egg-info/
    rm -rf build/
    rm -rf dist/
    
    echo "✅ Cleanup complete!"
    echo "Current git status:"
    git status
else
    echo "❌ Cleanup cancelled"
fi
