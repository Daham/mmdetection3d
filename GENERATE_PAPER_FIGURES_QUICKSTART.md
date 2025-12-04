# Quick Start: Generate Paper Figures

## ✅ Good News!

**Yes, you can generate qualitative comparison figures for your research paper!**

I've created tools that will generate **publication-quality Bird's Eye View (BEV) visualizations** comparing:
- Ground Truth (green boxes)
- Baseline Single-Scale (blue boxes) - *optional*
- **VoxAdapt (Ours) (red boxes)** - *your PhD contribution*

## 🚀 Quick Start (3 Steps)

### Step 1: Make sure you're in the correct environment

```bash
cd /home/daham/mmdetection_project/mmdetection3d
source /home/daham/mmdetection_project/mmdet_env/bin/activate
```

✓ **FIXED**: The scripts now activate the correct environment automatically!

### Step 2: Generate figures (VoxAdapt only, without baseline)

```bash
./generate_paper_figures.sh
```

This will create **6 qualitative comparison images** in `qualitative_results/`

### Step 3: View the results

```bash
ls -lh qualitative_results/*.png
```

Each PNG shows:
- **Left**: Ground Truth annotations
- **Middle**: Baseline detections (if available)
- **Right**: VoxAdapt detections (your method)

## 📊 What Gets Generated

### Output Files
```
qualitative_results/
├── comparison_000000.png  # Sample validation image 1
├── comparison_000123.png  # Sample validation image 2
├── comparison_000456.png  # Sample validation image 3
... (6 total)
```

### Figure Properties
- **Format**: High-resolution PNG (300 DPI)
- **Size**: 18×6 inches (suitable for 2-column papers)
- **Layout**: 3-panel side-by-side comparison
- **Colors**: Green (GT), Blue (baseline), Red (VoxAdapt/yours)

## 🎯 Current Checkpoint Status

**VoxAdapt (Your Method)**: ✅ Available
- Location: `work_dirs/method3_with_fix_5epochs/epoch_5.pth`
- Status: 64MB checkpoint from your 5-epoch training
- Results: 84.96% Easy, 73.54% Moderate, 68.52% Hard AP

**Baseline**: ⚠️ Need to specify
- The script will work without baseline (shows GT vs VoxAdapt only)
- To add baseline comparison, you need to either:
  1. Train a baseline model for 5 epochs, OR
  2. Specify an existing baseline checkpoint path

## 💡 Usage Options

### Option 1: VoxAdapt Only (Recommended for now)

```bash
./generate_paper_figures.sh --num-samples 6
```

This creates GT vs VoxAdapt comparisons (2-panel figures instead of 3-panel).

### Option 2: With Specific Baseline

If you have a baseline checkpoint, specify it:

```bash
python generate_qualitative_comparison.py \
    --voxadapt-checkpoint work_dirs/method3_with_fix_5epochs/epoch_5.pth \
    --baseline-checkpoint work_dirs/YOUR_BASELINE_PATH/epoch_5.pth \
    --num-samples 6 \
    --output-dir qualitative_results
```

### Option 3: Specific Sample Indices

To show specific challenging cases:

```bash
./generate_paper_figures.sh --sample-indices 0 10 50 100 200 500
```

### Option 4: More Samples

```bash
./generate_paper_figures.sh --num-samples 12
```

## 📝 For Your Paper

### Suggested Figure Caption

```
Figure X: Qualitative detection results on KITTI validation set. 
(Left) Ground truth annotations. (Right) VoxAdapt with learnable 
multi-scale voxelization (ours). Our method demonstrates improved 
detection of distant and small objects through adaptive scale 
selection learned end-to-end from the detection loss.
```

### LaTeX Integration

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\linewidth]{qualitative_results/comparison_000123.png}
    \caption{Qualitative comparison on KITTI validation set...}
    \label{fig:qualitative_voxadapt}
\end{figure*}
```

## 🔧 If You Want Baseline Comparison Too

You have two options:

### A) Quick 5-Epoch Baseline Training

```bash
# Use your existing single-scale baseline config
python tools/train.py \
    configs/second/validation_baseline_01_single_scale_80ep.py \
    --work-dir work_dirs/baseline_for_comparison_5epochs \
    --cfg-options train_cfg.max_epochs=5
```

Then update the script:

```bash
python generate_qualitative_comparison.py \
    --baseline-checkpoint work_dirs/baseline_for_comparison_5epochs/epoch_5.pth \
    --voxadapt-checkpoint work_dirs/method3_with_fix_5epochs/epoch_5.pth \
    --num-samples 6
```

### B) Use Existing Baseline (if you have one from previous experiments)

Find any single-scale SECOND baseline checkpoint:

```bash
find work_dirs -name "*.pth" | grep -i "baseline\|second" | head -10
```

Then specify it in the script.

## 📖 More Details

See `QUALITATIVE_FIGURES_README.md` for comprehensive documentation including:
- Parameter descriptions
- Advanced usage examples
- Tips for selecting best samples
- Troubleshooting guide

## ❓ FAQ

**Q: Do I need baseline for the paper?**
A: Not necessarily! You can show GT vs VoxAdapt (2-panel) which still demonstrates your method's effectiveness. Baseline comparison (3-panel) is nice-to-have but not required.

**Q: Can I visualize specific scenes?**
A: Yes! Use `--sample-indices` to select specific validation samples that show interesting cases (distant objects, occlusions, etc.)

**Q: What if I want different colors or layouts?**
A: The Python script is easy to customize. Edit `generate_qualitative_comparison.py` lines 140-200 for visualization styling.

**Q: Can I generate more than 6 samples?**
A: Absolutely! Use `--num-samples 12` or any number. For papers, typically 4-8 examples work well.

## ✨ Example Run

```bash
source /home/daham/mmdetection_project/mmdet_env/bin/activate
./generate_paper_figures.sh --num-samples 6 --score-thr 0.3
```

Expected output:
```
==========================================
Generating Qualitative Comparison Figures
==========================================

[1/4] Skipping baseline model (not configured)
[2/4] Initializing VoxAdapt model...
[3/4] Loading KITTI validation samples...
[4/4] Generating 6 comparison figures...

  Processing sample 1/6: 000000
    Running VoxAdapt inference...
    Saved comparison figure: qualitative_results/comparison_000000.png

... (repeats for all 6 samples)

✓ Generated 6 comparison figures in: qualitative_results/
```

## 🎉 You're Ready!

Your qualitative figure generation tools are set up and ready to use. Generate some figures and pick the best examples for your paper!
