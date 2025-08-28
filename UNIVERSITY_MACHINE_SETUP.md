# University Machine Setup Guide for MMDetection3D

This guide helps you set up MMDetection3D on university lab machines or shared computing resources.

## Prerequisites

- Access to university computing lab
- Linux-based system (Ubuntu/CentOS)
- NVIDIA GPU access
- Internet connectivity (may require proxy)

## Step 1: Check System Access

```bash
# Check your permissions
whoami
groups
nvidia-smi  # Check GPU access
```

## Step 2: Module System (if available)

Many university systems use module systems:

```bash
# Common module commands
module avail  # See available modules
module load cuda/12.1
module load python/3.9
module list   # Check loaded modules
```

## Step 3: Proxy Configuration (if needed)

```bash
# Set proxy if university requires it
export http_proxy=http://proxy.university.edu:8080
export https_proxy=http://proxy.university.edu:8080
export HTTP_PROXY=http://proxy.university.edu:8080
export HTTPS_PROXY=http://proxy.university.edu:8080
```

## Step 4: User Space Installation

Since you may not have sudo access, install in user space:

```bash
# Create local installation directory
mkdir -p ~/local/bin
mkdir -p ~/local/lib
export PATH=$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```

## Step 5: Python Environment

```bash
# Option 1: Use system Python with virtual environment
python3 -m venv ~/mmdet_env
source ~/mmdet_env/bin/activate

# Option 2: Install Miniconda in user space (if allowed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
conda create -n mmdet_env python=3.9
conda activate mmdet_env
```

## Step 6: CUDA Setup

```bash
# Check CUDA version
nvcc --version || echo "CUDA not in PATH"

# If CUDA not available, load module or set path
export CUDA_HOME=/usr/local/cuda  # Adjust path as needed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Step 7: Install Dependencies

```bash
# Upgrade pip and essential tools
pip install --upgrade pip setuptools wheel

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# Install MMCV
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

## Step 8: Clone and Setup MMDetection3D

```bash
# Clone the repository
cd ~/
git clone https://github.com/Daham/mmdetection3d.git
cd mmdetection3d
git checkout develop

pip install mmdet opencv-python matplotlib tqdm mmengine

# Install additional dependencies
pip install -r requirements/build.txt

# Install MMDetection3D in editable mode
pip install -e .
```

## Step 9: Dataset Setup

```bash
# Create dataset directory in your user space
mkdir -p ~/datasets/kitti

# If dataset is shared on the system, create symlink
ln -s /shared/datasets/kitti ~/datasets/kitti

# Update config files to point to your dataset location
```

## Step 10: Job Submission (if SLURM/PBS available)

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=mmdet3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB

module load cuda/12.1
source ~/mmdet_env/bin/activate
cd ~/mmdetection3d

python tools/train.py configs/second/second_hv_secfpn_memory_optimized_kitti.py
```

### PBS Example

```bash
#!/bin/bash
#PBS -N mmdet3d
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=24:00:00
#PBS -l mem=32gb

cd $PBS_O_WORKDIR
module load cuda/12.1
source ~/mmdet_env/bin/activate

python tools/train.py configs/second/second_hv_secfpn_memory_optimized_kitti.py
```

## Step 11: Interactive Testing

```bash
# For interactive sessions
srun --pty --gres=gpu:1 --mem=16G bash
# or
qsub -I -l nodes=1:ppn=4:gpus=1

# Then run your training
source ~/mmdet_env/bin/activate
cd ~/mmdetection3d
python tools/train.py configs/second/second_hv_secfpn_memory_optimized_kitti.py --dry-run
```

## Troubleshooting University-Specific Issues

### Permission Issues
```bash
# Check disk quotas
quota -u
df -h ~/

# Clean cache if quota exceeded
pip cache purge
rm -rf ~/.cache/
```

### Network Issues
```bash
# Test connectivity
ping google.com
wget -q --spider https://pypi.org

# Use university mirrors if available
pip install -i https://university-mirror.edu/pypi/simple/ package_name
```

### Module System Issues
```bash
# Reset modules if conflicts
module purge
module load cuda/12.1 python/3.9

# Check module dependencies
module show cuda/12.1
```

### Storage Issues
```bash
# Use scratch space for temporary files
export TMPDIR=/scratch/$USER
mkdir -p $TMPDIR

# Symlink work_dirs to scratch
ln -s /scratch/$USER/work_dirs ~/mmdetection3d/work_dirs
```

## University-Specific Optimizations

### For Shared Resources
- Use nice command for CPU-intensive tasks
- Monitor resource usage with `htop`, `nvidia-smi`
- Schedule training during off-peak hours

### For Limited Storage
- Use dataset symlinks
- Clean work_dirs regularly
- Compress checkpoint files

### For Queue Systems
- Request appropriate resources
- Use checkpointing for long jobs
- Test with short jobs first

## Support Resources

- Contact your university IT support
- Check university-specific documentation
- Join university HPC user groups
- Use university computing workshops

This setup ensures compliance with university policies while providing optimal performance for your research.
