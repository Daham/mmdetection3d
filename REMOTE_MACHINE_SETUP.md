
# MMDetection3D and KITTI Dataset Setup Guide

This comprehensive guide helps you set up a remote Ubuntu server for 3D object detection using the `mmdetection3d` framework and the KITTI dataset. It includes environment setup, dataset preparation, and a test training run using Jupyter Notebook.

---

## Contents

- [0. Remote Server Setup Script](#0-remote-server-setup-script)
- [1. Prerequisites](#1-prerequisites)
- [2. Download KITTI Raw Data](#2-download-kitti-raw-data)
- [3. Unzip Data and Create KITTI Structure](#3-unzip-data-and-create-kitti-structure)
- [4. Generate KITTI Info Files (.pkl)](#4-generate-kitti-info-files-pkl)
- [5. CUDA Toolkit and Debugging](#5-cuda-toolkit-and-debugging)
- [6. Start Training!](#6-start-training)

---

## 0. Remote Server Setup Script

Create and execute the setup script below to prepare your environment:

```bash
# Save this as setup_remote.sh and execute on your Ubuntu server
# Script installs Python 3.8, PyTorch, MMCV, MMDetection3D, and Jupyter Notebook

#!/bin/bash
set -e

echo "--- Starting MMDetection and Jupyter Notebook Setup on Remote Machine ---"

PYTHON_VERSION="3.8"
PROJECT_DIR="$HOME/mmdetection_project"
VENV_NAME="mmdet_env"
JUPYTER_PORT="8888"
REMOTE_SSH_USER="$(whoami)"

sudo apt update
sudo apt install -y software-properties-common python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
sudo apt install -y software-properties-common python3.8 python3.8-venv python3.8-dev
sudo apt install -y nvidia-cuda-toolkit  # Required for building MMCV CUDA ops

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
python${PYTHON_VERSION} -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Optional debug setting for PyTorch GPU debugging
export CUDA_LAUNCH_BLOCKING=1

pip install --upgrade pip setuptools wheel build

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

python -c "
import mmcv, torch, sys
print(f'Python: {sys.version.split()[0]}')
print(f'MMCV: {mmcv.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No CUDA')
try:
    import mmcv.ops
    from mmcv.ops import RoIAlign
    print('MMCV ops OK')
except Exception as e:
    print('MMCV ops error:', e)
    exit(1)
"

pip install mmdet opencv-python matplotlib tqdm mmengine
python -c "import mmdet; print('MMDetection:', mmdet.__version__)"

pip install notebook
jupyter notebook --generate-config -y

sed -i '/#c.NotebookApp.allow_origin/c\c.NotebookApp.allow_origin = "*"' ~/.jupyter/jupyter_notebook_config.py
sed -i '/#c.NotebookApp.ip/c\c.NotebookApp.ip = "0.0.0.0"' ~/.jupyter/jupyter_notebook_config.py
sed -i '/#c.NotebookApp.open_browser/c\c.NotebookApp.open_browser = False' ~/.jupyter/jupyter_notebook_config.py
sed -i '/#c.NotebookApp.port/c\c.NotebookApp.port = '${JUPYTER_PORT} ~/.jupyter/jupyter_notebook_config.py

echo "Run 'jupyter notebook password' and press Enter after setting it."
read -p "Press Enter AFTER you have set your Jupyter password..."

nohup jupyter notebook --port=$JUPYTER_PORT &

echo "--- Setup Complete ---"
```

---

## 1. Prerequisites

Ensure you have:

* Ubuntu-based remote server with SSH access.
* Internet access to download PyTorch, KITTI dataset.
* Disk space: ~41 GB (KITTI) + space for processed data.
* GPU with CUDA support and compatible drivers.

---

## 2. Download KITTI Raw Data

```bash
cd ~/mmdetection_project
mkdir -p data/kitti
cd data/kitti

# Download raw training components
wget -P . https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget -P . https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
wget -P . https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
wget -P . https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
```

---

## 3. Unzip Data and Create KITTI Structure

```bash
unzip data_object_image_2.zip
unzip data_object_velodyne.zip
unzip data_object_calib.zip
unzip data_object_label_2.zip
rm *.zip  # Optional cleanup
```

Expected structure:

```
data/kitti/
├── training/
│   ├── image_2/
│   ├── velodyne/
│   ├── calib/
│   └── label_2/
└── testing/
    ├── image_2/
    ├── velodyne/
    └── calib/
```

---

## 4. Generate KITTI Info Files (.pkl)

Before generating KITTI info files, create the `ImageSets` directory and add the required split files: `train.txt`, `val.txt`, and `test.txt`.

For your setup, **all three files should contain the exact same entries (000000 to 000049).**

Run these commands:

```bash
cd ~/mmdetection_project/data/kitti/
mkdir -p ImageSets

# Create a common list of 50 samples for train, val, and test splits
for i in $(seq -f "%06g" 0 49); do echo $i; done | tee ImageSets/train.txt ImageSets/val.txt ImageSets/test.txt
```

Expected directory structure including `ImageSets`:

```
data/kitti/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── training/
│   ├── image_2/
│   ├── velodyne/
│   ├── calib/
│   └── label_2/
└── testing/
    ├── image_2/
    ├── velodyne/
    └── calib/
```

Now run the data info generation:

```bash
cd ~/mmdetection_project
source mmdet_env/bin/activate

python mmdetection3d/tools/create_data.py kitti     --root-path ./data/kitti/     --out-dir ./data/kitti/     --extra-tag kitti
```

This generates:

- `kitti_infos_train.pkl`
- `kitti_infos_val.pkl`
- `kitti_dbinfos_train.pkl`

### Update Default Config Path

Edit the config file:

```bash
mmdetection3d/configs/_base_/datasets/kitti-3d-car.py
```

Update the `data_root` variable to your dataset path:

```python
data_root = '/home/{username}/mmdetection_project/data/kitti/'
```

---

## 5. CUDA Toolkit and Debugging

Install CUDA Toolkit for compiling MMCV CUDA ops:

```bash
sudo apt install nvidia-cuda-toolkit
```

Set environment variable for clearer runtime error debugging:

```bash
export CUDA_LAUNCH_BLOCKING=1
```

You may add this line to your `.bashrc` or `.zshrc` for persistence:

```bash
echo "export CUDA_LAUNCH_BLOCKING=1" >> ~/.bashrc
```

---

## 6. Start Training!

```bash
cd ~/mmdetection_project
source mmdet_env/bin/activate

python mmdetection3d/tools/train.py mmdetection3d/configs/point_rcnn/point_rcnn_kitti-3d-3class.py
```

This launches training using the KITTI dataset.

---

## 7. Testing
Run the test command on your trained model:

```bash

python tools/test.py configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py work_dirs/second_hv_secfpn_8xb6-80e_kitti-3d-car/epoch_40.pth --task lidar_det

```


## Access Jupyter Notebook (Optional)

```bash
ssh -L 8888:localhost:8888 user@your-server-ip
```

Then open: [http://localhost:8888](http://localhost:8888)

---

**Happy Training!**
