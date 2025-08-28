# Remote Machine Setup Guide for MMDetection3D

This guide helps you set up MMDetection3D on a remote machine (like AWS, GCP, or university servers).

## Prerequisites

- Ubuntu 18.04+ or CentOS 7+
- NVIDIA GPU with CUDA support
- SSH access to the remote machine
- At least 16GB RAM and 50GB storage

## Step 1: SSH Connection

```bash
ssh username@remote-machine-ip
```

## Step 2: System Update

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl
```

## Step 3: CUDA Installation

Download and install CUDA 12.1:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

Add CUDA to PATH:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 4: Python Environment Setup

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create virtual environment
VENV_NAME="mmdet_env"
python -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Optional debug setting for PyTorch GPU debugging
export CUDA_LAUNCH_BLOCKING=1

pip install --upgrade pip setuptools wheel build

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

## Step 5: Clone and Install MMDetection3D

```bash
git clone https://github.com/Daham/mmdetection3d.git
cd mmdetection3d
git checkout feature/importance_guided_multi_scale_second

pip install mmdet opencv-python matplotlib tqdm mmengine
pip install -r requirements/build.txt
pip install -e .
```

## Step 6: Dataset Setup

```bash
# Create dataset directory
mkdir -p data/kitti

# Download KITTI dataset (replace with your download method)
# Upload your KITTI dataset to data/kitti/
```

## Step 7: Verify Installation

```bash
python -c "import mmdet3d; print('MMDetection3D installed successfully!')"
python tools/train.py configs/second/second_hv_secfpn_memory_optimized_kitti.py --dry-run
```

## Troubleshooting

### CUDA Issues
- Check CUDA installation: `nvcc --version`
- Verify GPU availability: `nvidia-smi`

### Memory Issues
- Reduce batch size in configs
- Use gradient checkpointing
- Monitor GPU memory: `watch -n 1 nvidia-smi`

### Network Issues
- Use proxy if needed
- Download packages manually if pip fails

## Performance Optimization

### For Training
```bash
# Set optimal GPU settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
```

### For Multi-GPU Training
```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/second/second_hv_secfpn_memory_optimized_kitti.py --launcher pytorch
```

## Security Considerations

- Use SSH keys instead of passwords
- Set up firewall rules
- Keep system updated
- Use non-root user for training

## Monitoring Training

```bash
# Monitor training progress
tail -f work_dirs/*/train.log

# Monitor system resources
htop
nvidia-smi -l 1
```

This setup provides a complete environment for running your PhD research on remote machines with optimal performance and security.
