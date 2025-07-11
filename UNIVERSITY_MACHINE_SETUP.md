# ✅ Full Setup Instructions: Python 3.8 + PyTorch + CUDA Toolkit 12.1 + MMCV on Ubuntu (for NVIDIA RTX 4070/4070 Super)

This guide walks you through installing Python 3.8, PyTorch, CUDA Toolkit 12.1, and MMCV on Ubuntu. It assumes you have a compatible NVIDIA GPU (e.g., RTX 4070 Super).

---

## 📦 Step 1: Install Python 3.8 and Dependencies

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y software-properties-common python3.8 python3.8-venv python3.8-dev
```

---

## 🖥️ Step 2: Install NVIDIA Driver

```bash
sudo apt install -y nvidia-driver-535
```

> 🔄 **REBOOT NOW is MANDATORY to activate the NVIDIA driver:**
```bash
sudo reboot
```

---

## 🔍 Step 3: Verify NVIDIA Driver After Reboot

```bash
nvidia-smi
```

---

## ⚙️ Step 4: Install CUDA Toolkit 12.1 (Offline .deb)

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda-toolkit-12-1
```

### ✨ Add CUDA to Environment Variables

```bash
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### ✅ Verify CUDA Installation

```bash
nvcc --version
```

---

## 🐍 Step 5: Set Up Python Virtual Environment

```bash
mkdir -p ~/mmdetection_project
cd ~/mmdetection_project
python3.8 -m venv mmdet_env
source mmdet_env/bin/activate
```

---

## 🔧 Step 6: Install PyTorch and MMCV

```bash
# Optional: Enable CUDA kernel debugging
export CUDA_LAUNCH_BLOCKING=1

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel build

# Install PyTorch for CUDA 12.1
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# Install MMCV for CUDA 12.1 and PyTorch 2.1
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

---

## 🔍 Step 7: Verify Setup

```bash
python3 -c "import mmcv, torch, sys; print(f'Python: {sys.version.split()[0]}'); print(f'MMCV: {mmcv.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No CUDA'); import mmcv.ops; from mmcv.ops import RoIAlign; print('MMCV ops OK')"
```
