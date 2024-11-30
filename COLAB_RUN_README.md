
# **MMDetection3D Setup and Workflow for KITTI Dataset**

This repository provides a complete workflow for setting up **MMDetection3D**, preparing the KITTI dataset, and training and testing 3D object detection models (e.g., SECOND).

---

## **1. KITTI Dataset Folder Structure and Download**

### **Dataset Download in Colab**
To download and prepare the KITTI dataset directly in the Colab environment, run the following commands in the shell:

```bash
# Install required tools
apt-get install -y wget unzip

# Create the dataset directory
mkdir -p /content/drive/MyDrive/KITTI

# Download KITTI dataset files
# Left images for training
wget -P /content/drive/MyDrive/KITTI/ http://www.cvlibs.net/download.php?file=data_object_image_2.zip
# Velodyne point clouds
wget -P /content/drive/MyDrive/KITTI/ http://www.cvlibs.net/download.php?file=data_object_velodyne.zip
# Calibration files
wget -P /content/drive/MyDrive/KITTI/ http://www.cvlibs.net/download.php?file=data_object_calib.zip
# Labels for training
wget -P /content/drive/MyDrive/KITTI/ http://www.cvlibs.net/download.php?file=data_object_label_2.zip

# Extract the files
unzip /content/drive/MyDrive/KITTI/data_object_image_2.zip -d /content/drive/MyDrive/KITTI
unzip /content/drive/MyDrive/KITTI/data_object_velodyne.zip -d /content/drive/MyDrive/KITTI
unzip /content/drive/MyDrive/KITTI/data_object_calib.zip -d /content/drive/MyDrive/KITTI
unzip /content/drive/MyDrive/KITTI/data_object_label_2.zip -d /content/drive/MyDrive/KITTI
```

Ensure that the dataset follows this structure after extraction:

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
└── ImageSets/
    ├── train.txt          # List of training samples
    ├── val.txt            # List of validation samples
    ├── trainval.txt       # Combined list of training and validation samples
    ├── test.txt           # List of testing samples
```

### **Control the Sample Size with `ImageSets`**
The `ImageSets` folder allows you to control the number of samples used for training, validation, and testing by editing the following files:
- **`train.txt`**: Contains indices of samples used for training.
- **`val.txt`**: Contains indices of samples used for validation.
- **`test.txt`**: Contains indices of samples used for testing.
- **`trainval.txt`**: A combined list of training and validation samples.

To reduce the sample size, modify these files by including only the desired sample indices.

Example:
```plaintext
000000
000001
000002
```

This will limit the dataset to three samples.

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

## **3. Google Colab Workflow**

To streamline the entire process in Google Colab, follow this sequence of steps:

1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Setup Environment**:
   Install required libraries as detailed in the **Environment Setup** section above.

3. **Download KITTI Dataset**:
   Use the commands provided in the **Dataset Download in Colab** section to download and extract the dataset.

4. **Prepare KITTI Dataset**:
   ```bash
   python tools/create_data.py kitti \
       --root-path /content/drive/MyDrive/KITTI \
       --out-dir /content/drive/MyDrive/KITTI_CONVERTED \
       --extra-tag kitti
   ```

5. **Train and Test Models**:
   Use the commands in **Training** and **Testing** sections.

6. **Check Results**:
   Review the evaluation metrics for model performance, including AP metrics for BBox, BEV, and 3D.

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
