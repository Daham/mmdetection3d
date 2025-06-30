# 🛠️ Fixing CUDA GPU Unavailability and Missing GUI After NVIDIA Driver Installation

This README documents all steps to resolve:

- CUDA GPU not available in PyTorch (`torch.cuda.is_available() → False`)
- `nvidia-smi` command not found or failing
- GUI (GNOME) not launching after NVIDIA driver installation

---

## 🔍 Problem Description

When running:

```bash
python test_cuda_segmentation.py
```

Output showed:

```
CUDA visible: False
CUDA version seen by torch: 12.1
torch version: 2.1.1+cu121
CUDA test failed: No CUDA GPUs are available
```

And:

```bash
nvidia-smi
```

Returned:
```
Command 'nvidia-smi' not found...
```

---

## ✅ Resolution Steps

---

### Step 1: Confirm GPU Is Detected by the System

```bash
lspci | grep -i nvidia
```

Expected output (example):

```
01:00.0 VGA compatible controller: NVIDIA Corporation AD103 [GeForce RTX 4070 Ti SUPER]
```

---

### Step 2: Check for and Disable `nouveau` Driver

Check if `nouveau` is active:

```bash
lsmod | grep nouveau
```

If present, disable it:

```bash
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot
```

After reboot, recheck:

```bash
lsmod | grep nouveau
```

✅ No output means it's disabled.

---

### Step 3: Install NVIDIA Driver

Install the driver (example: version 550, which supports CUDA 12.1):

```bash
sudo apt update
sudo apt install nvidia-driver-550 nvidia-utils-550
sudo reboot
```

---

### Step 4: Confirm Driver Installation

After reboot:

```bash
nvidia-smi
```

Expected: GPU status and CUDA version visible.

Then check in Python:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```
True
```

---

## 💥 Problem: GUI (GNOME) Not Launching

You may notice your system boots into a terminal (no desktop). GDM might be inactive.

---

### Step 5: Check the Display Manager

```bash
systemctl status display-manager
```

If `gdm.service` is `inactive (dead)`, restart it:

```bash
sudo systemctl start gdm
```

Press `Ctrl + Alt + F1` or `Ctrl + Alt + F2` to return to the GUI if it comes up.

---

### Step 6: Reconfigure GDM and NVIDIA X Settings

```bash
sudo dpkg-reconfigure gdm3
sudo nvidia-xconfig
sudo reboot
```

---

### Step 7: If GDM Fails, Use LightDM

Install LightDM as a fallback:

```bash
sudo apt install lightdm
```

Choose `lightdm` when prompted, then:

```bash
sudo reboot
```

✅ This should bring back the graphical interface.

---

## ✅ Final Checks

After GUI login:

```bash
nvidia-smi
```

✅ GPU should be visible.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

✅ Should return `True`.

---

## 🧪 Sample Output

```bash
$ python -c "import torch; print(torch.cuda.is_available())"
True

$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx.xx     Driver Version: 550.xx.xx     CUDA Version: 12.1  |
|-------------------------------+----------------------+----------------------|
| 0  RTX 4070 Ti SUPER          | ...                  |                      |
+-----------------------------------------------------------------------------+
```

---

## 🧹 Troubleshooting

- **Still no GUI?** Try:
  ```bash
  startx
  ```

- **See display errors:**
  ```bash
  cat /var/log/Xorg.0.log | grep -E "(EE|WW)"
  ```

- **General logs:**
  ```bash
  journalctl -xe
  ```

- **Fallback GUI:**
  ```bash
  sudo apt install xfce4
  startxfce4
  ```

---

## ℹ️ Notes

- ACPI BIOS errors like `AE_ALREADY_EXISTS` may appear in logs but are often harmless.
- This guide assumes Ubuntu 24.04 or similar Debian-based distributions.

---

**Author:** Your troubleshooting session  
**Last updated:** June 30, 2025
