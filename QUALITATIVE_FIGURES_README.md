# Qualitative Comparison Figure Generation

This directory contains tools to generate publication-quality qualitative comparison figures for your research paper, comparing VoxAdapt (learnable multi-scale voxelization) against baseline methods.

## Overview

The visualization script generates **side-by-side Bird's Eye View (BEV)** comparisons showing:
1. **Ground Truth** (green boxes)
2. **Baseline Single-Scale** (blue boxes)
3. **VoxAdapt (Ours)** (red boxes)

## Quick Start

### Option 1: Generate 6 Representative Samples

```bash
./generate_paper_figures.sh
```

This will:
- Use checkpoints from your 5-epoch training experiments
- Select 6 evenly-spaced validation samples
- Save figures to `qualitative_results/`

### Option 2: Generate Specific Samples

```bash
./generate_paper_figures.sh --sample-indices 0 10 50 100 200 500
```

This generates figures for specific validation indices (useful for highlighting challenging cases).

### Option 3: Custom Configuration

```bash
python generate_qualitative_comparison.py \
    --baseline-checkpoint work_dirs/baseline_single_scale_comparison/epoch_5.pth \
    --voxadapt-checkpoint work_dirs/method3_with_fix_5epochs/epoch_5.pth \
    --num-samples 10 \
    --output-dir paper_figures \
    --score-thr 0.3
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--baseline-config` | Baseline config file | `configs/second/validation_baseline_01_single_scale_80ep.py` |
| `--baseline-checkpoint` | Baseline checkpoint | `work_dirs/baseline_single_scale_comparison/epoch_5.pth` |
| `--voxadapt-config` | VoxAdapt config file | `configs/adaptive_voxelnet/adaptive_octree_simple.py` |
| `--voxadapt-checkpoint` | VoxAdapt checkpoint | `work_dirs/method3_with_fix_5epochs/epoch_5.pth` |
| `--data-root` | KITTI dataset root | `data/kitti` |
| `--num-samples` | Number of samples | `6` |
| `--output-dir` | Output directory | `qualitative_results` |
| `--score-thr` | Detection score threshold | `0.3` |
| `--sample-indices` | Specific sample indices | Auto-selected |
| `--device` | Inference device | `cuda:0` |

## Output Format

Each generated figure contains:
- **3 panels** (Ground Truth, Baseline, VoxAdapt)
- **BEV projection** (Bird's Eye View from above)
- **Color-coded boxes**:
  - Green: Ground truth
  - Blue: Baseline predictions
  - Red: VoxAdapt predictions
- **Detection scores** shown on each box
- **High resolution** (300 DPI) suitable for publication

## Example Usage for Paper

### Select Challenging Cases

To highlight VoxAdapt's improvements on challenging scenarios:

```bash
# Find samples with distant objects, occlusions, or small objects
# Example: validation indices that showed large improvements
./generate_paper_figures.sh --sample-indices 15 42 89 123 201 456
```

### Generate Different Difficulty Levels

```bash
# Easy samples (close, clear)
./generate_paper_figures.sh --sample-indices 10 20 30 --output-dir figures_easy

# Moderate samples (medium distance)
./generate_paper_figures.sh --sample-indices 100 150 200 --output-dir figures_moderate

# Hard samples (distant, occluded)
./generate_paper_figures.sh --sample-indices 300 400 500 --output-dir figures_hard
```

### Adjust Score Threshold

```bash
# Higher threshold (fewer, more confident detections)
./generate_paper_figures.sh --score-thr 0.5 --output-dir figures_high_conf

# Lower threshold (more detections, including uncertain ones)
./generate_paper_figures.sh --score-thr 0.1 --output-dir figures_all_dets
```

## Tips for Paper Figures

1. **Select Diverse Cases**: Choose samples that show:
   - Clear improvements (VoxAdapt detects objects missed by baseline)
   - Challenging scenarios (distant objects, occlusions, crowded scenes)
   - Different object scales (small vs large vehicles)

2. **Figure Layout in Paper**:
   ```
   Suggested layout: 2 rows × 3 columns = 6 samples
   Each sample shows: GT | Baseline | VoxAdapt (Ours)
   ```

3. **Caption Template**:
   ```
   Figure X: Qualitative detection results on KITTI validation set comparing 
   VoxAdapt (learnable multi-scale voxelization) with single-scale baseline. 
   From left to right: Ground truth, baseline detector, VoxAdapt (ours). 
   Green boxes: ground truth, blue boxes: baseline predictions, red boxes: 
   VoxAdapt predictions. VoxAdapt demonstrates improved detection of distant 
   and small objects (rows 1-2) and better localization in crowded scenes 
   (rows 3-4) through adaptive scale selection. Best viewed in color.
   ```

4. **Highlighting Improvements**:
   - Look for samples where baseline **misses objects** but VoxAdapt detects them
   - Find cases with **better localization** (tighter bounding boxes)
   - Identify **distant/small objects** where multi-scale helps

## Understanding the Visualization

### BEV Coordinate System
- **X-axis**: Left-right (lateral), range: -40m to +40m
- **Y-axis**: Forward (longitudinal), range: 0m to +70m
- **Origin**: LiDAR sensor position

### Box Information
Each detection box shows:
- **Class name**: Car, Pedestrian, or Cyclist
- **Confidence score**: 0.0 to 1.0
- **Box orientation**: Arrow indicates heading direction

### Color Coding
- **Gray points**: Raw LiDAR point cloud
- **Green boxes**: Ground truth annotations
- **Blue boxes**: Baseline single-scale detections
- **Red boxes**: VoxAdapt learnable multi-scale detections

## Requirements

The script uses:
- `mmdet3d.apis.LidarDet3DInferencer` for inference
- `matplotlib` for visualization
- Trained checkpoints from both baseline and VoxAdapt

## Troubleshooting

### Issue: "Point cloud not found"
**Solution**: Ensure KITTI dataset is properly set up:
```bash
ls data/kitti/training/velodyne/*.bin
```

### Issue: "Checkpoint not found"
**Solution**: Update paths to your trained checkpoints:
```bash
# List available checkpoints
ls work_dirs/*/epoch_*.pth
```

### Issue: "Out of memory"
**Solution**: Reduce batch size or number of samples:
```bash
./generate_paper_figures.sh --num-samples 3
```

### Issue: No detections shown
**Solution**: Lower the score threshold:
```bash
./generate_paper_figures.sh --score-thr 0.1
```

## Advanced: Finding Best Samples

To identify samples where VoxAdapt shows largest improvements:

```bash
# 1. Run evaluation and save per-sample results (if not already done)
# 2. Analyze results to find samples with:
#    - Low baseline AP but high VoxAdapt AP
#    - Many false negatives in baseline
#    - Distant/small objects

# Then visualize those specific samples
./generate_paper_figures.sh --sample-indices [YOUR_BEST_SAMPLES]
```

## Expected Output

After running, you should see:
```
qualitative_results/
├── comparison_000000.png  # Sample 0
├── comparison_000123.png  # Sample 123
├── comparison_000456.png  # Sample 456
├── ...
```

Each PNG file is:
- **High resolution**: 300 DPI
- **Size**: ~18×6 inches (3 panels side-by-side)
- **Format**: Ready for LaTeX/Word inclusion

## Integration with Paper

### LaTeX Example

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/comparison_000123.png}
    \caption{Qualitative comparison on KITTI validation sample 000123. 
             VoxAdapt (right) detects distant vehicles missed by the 
             baseline (middle) through adaptive multi-scale voxelization.}
    \label{fig:qualitative}
\end{figure*}
```

### Multi-Sample Figure

```latex
\begin{figure*}[t]
    \centering
    \begin{subfigure}{\linewidth}
        \includegraphics[width=\linewidth]{figures/comparison_000123.png}
        \caption{Sample 1: Distant objects}
    \end{subfigure}
    \vspace{0.2cm}
    \begin{subfigure}{\linewidth}
        \includegraphics[width=\linewidth]{figures/comparison_000456.png}
        \caption{Sample 2: Crowded scene}
    \end{subfigure}
    \caption{Qualitative detection examples comparing VoxAdapt with baseline.}
    \label{fig:qualitative_multi}
\end{figure*}
```

## Next Steps

1. **Generate initial figures**: `./generate_paper_figures.sh`
2. **Review results**: Check `qualitative_results/` for best examples
3. **Refine selection**: Re-run with specific `--sample-indices` for best cases
4. **Include in paper**: Use high-DPI PNGs in your manuscript
5. **Highlight key improvements**: Annotate or add arrows in your paper to emphasize VoxAdapt advantages

## Contact

For issues or questions about figure generation, check:
- Script output for error messages
- KITTI dataset setup
- Checkpoint availability
- GPU memory usage
