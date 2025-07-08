
# 🛠️ Setting Up Ubuntu 22.04 with NVIDIA RTX 4070 Super and CUDA Toolkit 12.1

This guide details the full setup process to resolve boot issues with Ubuntu 22.04 on a system featuring an NVIDIA RTX 4070 Super GPU. It covers configuring Secure Boot, installing NVIDIA drivers, and setting up the CUDA Toolkit 12.1 for GPU-accelerated development.

---

## ⚠️ Problem Summary

After installation, Ubuntu 22.04 would get stuck on the loading screen. The GRUB menu wasn't visible, showing only the vendor logo and Ubuntu splash. This meant manual booting via the GRUB prompt was necessary every time, using commands like:

```grub
set root=(hd0,gpt3)
linux /boot/vmlinuz-6.8.0-60-generic root=UUID=<your-uuid> ro nomodeset
initrd /boot/initrd.img-6.8.0-60-generic
boot
```

---

## ✅ Root Cause

The issues stemmed from a missing or corrupted GRUB installation, boot hangs caused by unsupported or unconfigured NVIDIA graphics drivers, and Secure Boot interfering with kernel module loading.

---

## ✅ Final Working Solution

Follow these steps to get your system fully functional with NVIDIA drivers and CUDA.

### 1. 🔧 Reinstall GRUB from Live USB

To fix the GRUB issues, you'll need to reinstall it from an Ubuntu Live USB.

Boot from your Ubuntu Live USB.

Open a terminal and run these commands:

```bash
sudo mount /dev/nvme0n1p2 /mnt
sudo mount /dev/nvme0n1p1 /mnt/boot/efi
for dir in /dev /dev/pts /proc /sys /run; do
  sudo mount --bind $dir /mnt/$dir
done
sudo chroot /mnt
grub-install /dev/nvme0n1
update-grub
update-initramfs -u
exit
sudo reboot
```

> Replace `/dev/nvme0n1p2` with your root partition and `/dev/nvme0n1p1` with your EFI partition.

---

### 2. 🕵️ Add `nomodeset` to Prevent Boot Freeze

Before installing NVIDIA drivers, add `nomodeset` to your GRUB configuration to prevent the system from freezing during boot.

Edit the GRUB configuration file:

```bash
sudo nano /etc/default/grub
```

Update the GRUB_CMDLINE_LINUX_DEFAULT line to include `nomodeset`:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nomodeset"
```

Save the file, then update GRUB and reboot:

```bash
sudo update-grub
sudo reboot
```

---

### 3. 📦 Install NVIDIA Drivers (with Secure Boot)

Install NVIDIA driver 535, which supports CUDA 12.1 and newer.

Update your package lists and install the driver:

```bash
sudo apt update
sudo apt install nvidia-driver-535
```

During installation, you'll be prompted to set a MOK (Machine Owner Key) password. Choose a strong, memorable password.

After rebooting, a blue screen will appear. Follow these steps:

- Choose **"Enroll MOK"**
- Enter the password you set during the driver installation
- Confirm and reboot

Verify the driver installation:

```bash
nvidia-smi
```

---

### 4. 🧠 Install CUDA Toolkit 12.1 (without downgrading driver)

Install CUDA Toolkit 12.1, ensuring it works seamlessly with your new NVIDIA driver.

Download the CUDA local installer:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
```

Install the Debian package:

```bash
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda-toolkit-12-1
```

Add CUDA to your system's PATH and LD_LIBRARY_PATH by appending these lines to your `~/.bashrc` file:

```bash
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify the CUDA installation:

```bash
nvcc --version
```

---

## ✅ Verify CUDA is Working

After completing the installation, confirm that CUDA is properly recognized and functional.

Check driver and GPU information:

```bash
nvidia-smi
```

Check CUDA compiler version:

```bash
nvcc --version
```

Test with PyTorch (if you have it installed):

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## ✅ Optional: Remove `nomodeset` After Driver Works

Once your NVIDIA drivers are confirmed to be working, you can optionally remove the `nomodeset` parameter from GRUB.

Edit the GRUB configuration file:

```bash
sudo nano /etc/default/grub
```

Change the GRUB_CMDLINE_LINUX_DEFAULT line back:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```

Save the file, then update GRUB and reboot:

```bash
sudo update-grub
sudo reboot
```

---

## 📌 System Summary

| Component     | Value                 |
| ------------- | --------------------- |
| OS            | Ubuntu 22.04 LTS      |
| GPU           | NVIDIA RTX 4070 Super |
| Driver        | NVIDIA 535            |
| CUDA Toolkit  | 12.1                  |
| Secure Boot   | Enabled (with MOK)    |

---

## ✅ End Result

Your Ubuntu system should now boot without manual GRUB input, NVIDIA drivers will be active, and CUDA 12.1 will be fully available for development with tools like MMDetection3D, PyTorch, or custom CUDA code.
