
# **MMDetection3D Setup and Workflow for KITTI Dataset**

This repository provides a complete workflow for setting up **MMDetection3D**, preparing the KITTI dataset, and training and testing 3D object detection models (e.g., SECOND).

---

## **1. KITTI Dataset Folder Structure and Download**

### **Dataset Download**
1. Visit the official KITTI dataset website: [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/).
2. Download the following components:
   - **3D Object Detection Data**: Contains `image_2/`, `label_2/`, `velodyne/`, and `calib/` folders.
   - Optionally, download `planes/` for plane information.
3. Extract the downloaded files into a directory, ensuring the following structure:

```plaintext
KITTI/
├── training/
│   ├── image_2/           # Left camera images (for training)
│   ├── label_2/           # Ground truth labels for training
│   ├── velodyne/          # LiDAR point clouds
│   ├── calib/             # Calibration files
│   └── planes/            # Optional plane information (if available)
├── testing/
│   ├── image_2/           # Left camera images (for testing)
│   ├── velodyne/          # LiDAR point clouds
│   ├── calib/             # Calibration files
│   └── planes/            # Optional plane information (if available)
```

---

## **2. Commands for Setup, Training, and Testing**

### **Step 1: Environment Setup**
Install all necessary dependencies and libraries.

```bash
# Uninstall conflicting versions of PyTorch and related libraries
pip uninstall -y torch torchvision torchaudio

# Install compatible PyTorch, torchvision, and torchaudio versions
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Install MMEngine
pip install mmengine==0.9.1

# Install MMCV with CUDA support
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# Install MMDetection
pip install mmdet==3.0.0

# Install MMDetection3D
pip install mmdet3d==1.1.0
```

### **Step 2: Clone MMDetection3D and Install Dependencies**

```bash
# Clone the MMDetection3D repository
git clone https://github.com/open-mmlab/mmdetection3d.git

# Navigate to the repository directory
cd mmdetection3d

# Install additional dependencies
pip install -r requirements/build.txt

# Install MMDetection3D in editable mode
pip install -e .
```

### **Step 3: Prepare KITTI Dataset**
Convert the KITTI dataset into a format compatible with **MMDetection3D**:

```bash
python tools/create_data.py kitti \
    --root-path /content/drive/MyDrive/KITTI \
    --out-dir /content/drive/MyDrive/KITTI_CONVERTED \
    --extra-tag kitti
```

### **Step 4: Train the Model**
Train the SECOND model on the KITTI dataset:

```bash
python tools/train.py configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
    --work-dir work_dirs/second/
```

### **Step 5: Test the Model**
Evaluate the trained model using the `epoch_40.pth` checkpoint:

```bash
python tools/test.py configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
    work_dirs/second/epoch_40.pth --task lidar_det
```

---

## **3. Full Colab Notebook**

Save the following code as `full-process-colab.ipynb` to automate the entire process in Google Colab:

```python
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the Google Drive module from Google Colab to interact with Google Drive
",
    "from google.colab import drive
",
    "
",
    "# Mount your Google Drive to access files stored in it
",
    "drive.mount('/content/drive')
",
    "
",
    "# Import PyTorch to verify the installed version
",
    "import torch
",
    "
",
    "# Print the version of PyTorch installed in the Colab environment
",
    "print(f"PyTorch version: {torch.__version__}")
",
    "
",
    "# Uninstall existing versions of PyTorch, torchvision, and torchaudio to ensure a clean environment
",
    "!pip uninstall -y torch torchvision torchaudio
",
    "
",
    "# Install specific versions of PyTorch, torchvision, and torchaudio compatible with CUDA 11.8
",
    "!pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
",
    "
",
    "# Install MMEngine, the engine that supports OpenMMLab frameworks like MMDetection3D
",
    "!pip install mmengine==0.9.1
",
    "
",
    "# Install MMCV (OpenMMLab's foundational library) with CUDA 11.8 and PyTorch 2.0.0 support
",
    "!pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
",
    "
",
    "# Install MMDetection (2D object detection framework by OpenMMLab)
",
    "!pip install mmdet==3.0.0
",
    "
",
    "# Install MMDetection3D (3D object detection framework by OpenMMLab)
",
    "!pip install mmdet3d==1.1.0
",
    "
",
    "# Clone the MMDetection3D GitHub repository to the current working directory
",
    "!git clone https://github.com/open-mmlab/mmdetection3d.git
",
    "
",
    "# Change the current working directory to the cloned MMDetection3D directory
",
    "%cd mmdetection3d
",
    "
",
    "# Install additional build dependencies required by MMDetection3D
",
    "!pip install -r requirements/build.txt
",
    "
",
    "# Install MMDetection3D in editable mode to allow direct modifications to the code
",
    "!pip install -e .
",
    "
",
    "# Generate KITTI dataset information files needed for training and testing
",
    "!python tools/create_data.py kitti \
",
    "    --root-path /content/drive/MyDrive/KITTI \
",
    "    --out-dir /content/drive/MyDrive/KITTI \
",
    "    --extra-tag kitti
",
    "
",
    "# Train the SECOND (Sparse Encoders for Object Detection) model on the KITTI dataset
",
    "!python tools/train.py configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
",
    "    --work-dir work_dirs/second/
",
    "
",
    "# Test the trained SECOND model using a specific checkpoint
",
    "!python tools/test.py configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
",
    "    work_dirs/second/epoch_40.pth --task lidar_det
"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

---

## **4. Code Modifications**

### **Fixing `np.long` Issue**
Replace `np.long` with `np.int64` in the following file:
- **File**: `mmdet3d/datasets/transforms/dbsampler.py`
- **Line**: Around line 283
- **Change**:
  ```python
  dtype=np.long  # Original
  ```
  to
  ```python
  dtype=np.int64  # Fixed
  ```

---

## **5. References**
- [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io/en/latest/)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [Google Colab Setup](https://colab.research.google.com/)

---
