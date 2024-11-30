
# **MMDetection3D Setup and Workflow for KITTI Dataset**

This repository provides a complete workflow for setting up **MMDetection3D**, preparing the KITTI dataset, and training and testing 3D object detection models (e.g., SECOND).

---

## **1. KITTI Dataset Folder Structure**

To ensure compatibility with **MMDetection3D**, the KITTI dataset must follow this structure:

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

Ensure all file names (e.g., `000000.bin`, `000000.txt`) align with the KITTI dataset's naming conventions.

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

## **3. Code Modifications**

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

## **4. Output Results**

After running the test command, you will see results like the following:

| Metric       | IoU Threshold       | Difficulty Level | BBox AP | BEV AP  | 3D AP   | AOS  |
|--------------|---------------------|------------------|---------|---------|---------|------|
| **AP11**     | **@0.70, 0.70, 0.70** | Easy             | 0.0250  | 0.0273  | 0.0000  | 0.02 |
|              |                     | Moderate         | 0.0249  | 0.0273  | 0.0000  | 0.02 |
|              |                     | Hard             | 0.0249  | 0.0273  | 0.0000  | 0.02 |
| **AP11**     | **@0.70, 0.50, 0.50** | Easy             | 0.0250  | 0.0751  | 0.0751  | 0.02 |
|              |                     | Moderate         | 0.0249  | 0.0996  | 0.0996  | 0.02 |
|              |                     | Hard             | 0.0249  | 0.0996  | 0.0996  | 0.02 |

Interpret these metrics to analyze the model's performance in 3D object detection.

---

## **5. Notes and Troubleshooting**
1. **Work Directory**:
   Ensure `--work-dir` is specified correctly for saving checkpoints and logs.

2. **Dataset Issues**:
   - Verify that all required folders (`image_2`, `velodyne`, etc.) exist in the dataset directory.
   - Ensure `.bin` files and `.txt` labels are formatted as per KITTI's structure.

3. **Checkpoint Path**:
   - Use the correct checkpoint path (e.g., `epoch_40.pth`) during testing.

4. **Environment Conflicts**:
   - If you face library version conflicts, use a clean environment or container (e.g., Docker).

---

## **6. References**
- [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io/en/latest/)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [Google Colab Setup](https://colab.research.google.com/)

---
