# 📊 Table Captions for 3-Difficulty Comparison

## Your Table Data:

| **Method**              | **Easy** | **Moderate** | **Hard** |
|-------------------------|:--------:|:------------:|:--------:|
| Fixed Single-Scale      | 80.16    | 70.87        | 66.17    |
| Naive Multi-Scale       | 78.19    | 68.40        | 65.01    |
| **VoxAdapt (Ours)**     | **84.96** | **73.54**   | **68.52** |

---

## 📝 CAPTION OPTIONS

### **Option 1: Ultra-Concise (1 sentence, ~25 words)** ⭐ For space-constrained journals

```
Table X: 3D object detection performance (AP@IoU=0.7) on KITTI validation set 
across three difficulty levels. VoxAdapt outperforms both baselines consistently.
```

---

### **Option 2: Concise (2 sentences, ~35-40 words)** ⭐⭐ MOST COMMON

```
Table X: Comparison of 3D Average Precision (AP@IoU=0.7, 40 recall points) across 
KITTI difficulty levels for Car detection. All methods trained for 5 epochs with 
identical hyperparameters on KITTI training set and evaluated on validation set.
```

---

### **Option 3: Standard (3 sentences, ~50-60 words)** ⭐⭐⭐ RECOMMENDED

```
Table X: 3D object detection performance comparison across KITTI difficulty levels. 
All methods use SECOND backbone trained for 5 epochs with batch size 6 and AdamW 
optimizer. Fixed Single-Scale uses 0.05m voxels, Naive Multi-Scale uses fixed 
{0.05m, 0.10m, 0.20m} without fusion, and VoxAdapt employs learned adaptive scale 
selection. Best results shown in bold.
```

---

### **Option 4: Detailed (4-5 sentences, ~80-100 words)** ⭐⭐ For comprehensive tables

```
Table X: Quantitative comparison of 3D Average Precision (AP@IoU=0.7, 40 recall 
points) for Car detection on KITTI validation set across Easy, Moderate, and Hard 
difficulty levels. Fixed Single-Scale baseline uses uniform 0.05m voxelization. 
Naive Multi-Scale employs three fixed scales {0.05m, 0.10m, 0.20m} with simple 
concatenation but no adaptive fusion. VoxAdapt (ours) uses the same three scales 
with learned attention-based scale selection and adaptive feature aggregation. 
All methods trained identically (5 epochs, batch size 6, AdamW lr=0.001) to isolate 
the impact of voxelization strategy. Best results highlighted in bold.
```

---

### **Option 5: Comprehensive (Technical, ~120-140 words)** ⭐ For methodology-focused papers

```
Table X: Cross-difficulty evaluation of 3D object detection methods on KITTI Car 
detection task. We report 3D Average Precision at IoU threshold 0.7 with 40 recall 
points following the official KITTI evaluation protocol. Easy difficulty includes 
objects with >40 pixels height, fully visible, and truncation <15%; Moderate allows 
partially occluded objects (occlusion level 1) and >25 pixels height; Hard includes 
heavily occluded objects (level 2) with >25 pixels height. Fixed Single-Scale 
baseline employs uniform 0.05m voxelization across all spatial regions. Naive 
Multi-Scale uses three fixed voxel sizes {0.05m, 0.10m, 0.20m} with direct 
concatenation but no learned fusion mechanism. VoxAdapt (our proposed method) 
adaptively selects and fuses multi-scale voxel features through learned attention 
weights based on local point density. All architectures use SECOND backbone with 
sparse 3D convolutions, trained for 5 epochs using AdamW optimizer (lr=0.001, 
batch size=6) on KITTI training split (3,712 samples) and evaluated on validation 
split (3,769 samples). Best performance in each column shown in bold.
```

---

## 🎯 **Quick Selection Guide**

### Use **Option 1 (Ultra-Concise)** if:
- ❌ Extremely tight space constraints (conference papers, letters)
- ✅ Table is self-explanatory
- ✅ Details provided in text

### Use **Option 2 (Concise)** if:
- ✅ Standard conference paper (CVPR, ICCV, ECCV)
- ✅ Limited caption space
- ✅ Supplementary details in main text

### Use **Option 3 (Standard)** if: ⭐⭐⭐ **BEST FOR 80% OF CASES**
- ✅ Journal papers (IEEE, Elsevier, Springer)
- ✅ Need to explain methods briefly
- ✅ Self-contained caption
- ✅ **This is my strong recommendation!**

### Use **Option 4 (Detailed)** if:
- ✅ Comprehensive experimental section
- ✅ Need to explain naive multi-scale approach
- ✅ Want caption to stand alone
- ✅ Journal encourages detailed captions

### Use **Option 5 (Comprehensive)** if:
- ✅ Methodology-focused journal (TPAMI, IJCV, T-PAMI)
- ✅ Need full technical specifications
- ✅ Reviewers expect complete details in captions
- ✅ Table is primary contribution

---

## ✅ **MY STRONG RECOMMENDATION**

### **Use Option 3 (Standard) with slight enhancement:**

```
Table X: 3D object detection performance comparison across KITTI difficulty levels 
for Car detection. All methods use SECOND backbone trained for 5 epochs (batch size 6, 
AdamW optimizer, lr=0.001) on KITTI training set (3,712 samples). Fixed Single-Scale 
uses uniform 0.05m voxels. Naive Multi-Scale uses fixed {0.05m, 0.10m, 0.20m} scales 
with simple concatenation. VoxAdapt (ours) employs learned adaptive scale selection 
with attention-based fusion. Metrics: 3D AP@IoU=0.7, 40 recall points. Best results 
in bold.
```

**Word count:** 73 words  
**Why this works:**
- ✅ Explains all three methods clearly
- ✅ Includes key training details
- ✅ Specifies evaluation metrics
- ✅ Self-contained but not verbose
- ✅ Reviewers can understand without reading main text

---

## 📊 **Key Improvements to Highlight**

Add these statistics to your table or caption:

| **Method**              | **Easy** | **Moderate** | **Hard** | **Avg** |
|-------------------------|:--------:|:------------:|:--------:|:-------:|
| Fixed Single-Scale      | 80.16    | 70.87        | 66.17    | 72.40   |
| Naive Multi-Scale       | 78.19    | 68.40        | 65.01    | 70.53   |
| **VoxAdapt (Ours)**     | **84.96** | **73.54**   | **68.52** | **75.67** |
| **Δ vs. Single-Scale**  | **+4.80** | **+2.67**   | **+2.35** | **+3.27** |
| **Δ vs. Naive**         | **+6.77** | **+5.14**   | **+3.51** | **+5.14** |

### **Enhanced Caption with Improvements:**

```
Table X: 3D object detection performance comparison across KITTI difficulty levels 
for Car detection (AP@IoU=0.7, 40 recall points). All methods use SECOND backbone 
trained for 5 epochs with identical settings (batch size 6, AdamW optimizer lr=0.001). 
Fixed Single-Scale uses uniform 0.05m voxels. Naive Multi-Scale uses fixed {0.05m, 
0.10m, 0.20m} scales with concatenation. VoxAdapt employs learned adaptive scale 
selection. VoxAdapt achieves +2.67% (Moderate) improvement over single-scale and 
+5.14% over naive multi-scale, demonstrating the importance of adaptive fusion. 
Best results in bold.
```

---

## 🎨 **LaTeX Formatting Options**

### **Minimal LaTeX Table:**

```latex
\begin{table}[t]
\centering
\caption{3D object detection performance comparison across KITTI difficulty levels 
for Car detection. All methods use SECOND backbone trained for 5 epochs with 
identical hyperparameters. Best results in \textbf{bold}.}
\label{tab:difficulty_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Easy} & \textbf{Moderate} & \textbf{Hard} \\
\midrule
Fixed Single-Scale & 80.16 & 70.87 & 66.17 \\
Naive Multi-Scale & 78.19 & 68.40 & 65.01 \\
\textbf{VoxAdapt (Ours)} & \textbf{84.96} & \textbf{73.54} & \textbf{68.52} \\
\bottomrule
\end{tabular}
\end{table}
```

---

### **Enhanced LaTeX Table with Improvements:**

```latex
\begin{table}[t]
\centering
\caption{3D object detection performance comparison across KITTI difficulty levels 
for Car detection (AP@IoU=0.7). All methods trained for 5 epochs with SECOND backbone. 
VoxAdapt consistently outperforms both baselines across all difficulty levels.}
\label{tab:difficulty_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Easy} & \textbf{Moderate} & \textbf{Hard} \\
\midrule
Fixed Single-Scale & 80.16 & 70.87 & 66.17 \\
Naive Multi-Scale & 78.19 & 68.40 & 65.01 \\
\textbf{VoxAdapt (Ours)} & \textbf{84.96} & \textbf{73.54} & \textbf{68.52} \\
\midrule
\textit{Improvement over Single} & \textit{+4.80} & \textit{+2.67} & \textit{+2.35} \\
\textit{Improvement over Naive} & \textit{+6.77} & \textit{+5.14} & \textit{+3.51} \\
\bottomrule
\end{tabular}

\vspace{2mm}
{\footnotesize Fixed Single-Scale: 0.05m voxels. Naive Multi-Scale: \{0.05m, 0.10m, 
0.20m\} with concatenation. VoxAdapt: learned adaptive scale selection with 
attention-based fusion.}
\end{table}
```

---

### **Complete LaTeX Table with Full Details:**

```latex
\begin{table}[t]
\centering
\caption{Quantitative comparison of 3D Average Precision (AP@IoU=0.7, 40 recall 
points) across KITTI difficulty levels for Car detection. All methods use SECOND 
backbone trained identically (5 epochs, batch size 6, AdamW lr=0.001) to isolate 
the impact of voxelization strategy. VoxAdapt demonstrates consistent superiority 
across all difficulty levels, with particularly strong gains on Easy (+4.80\%) and 
Moderate (+2.67\%) scenarios.}
\label{tab:difficulty_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Easy} & \textbf{Moderate} & \textbf{Hard} \\
\midrule
Fixed Single-Scale\textsuperscript{\dag} & 80.16 & 70.87 & 66.17 \\
Naive Multi-Scale\textsuperscript{\ddag} & 78.19 & 68.40 & 65.01 \\
\textbf{VoxAdapt (Ours)} & \textbf{84.96} & \textbf{73.54} & \textbf{68.52} \\
\midrule
\multicolumn{4}{l}{\textit{Absolute Improvement over Fixed Single-Scale:}} \\
\quad VoxAdapt & +4.80 & +2.67 & +2.35 \\
\multicolumn{4}{l}{\textit{Relative Improvement:}} \\
\quad VoxAdapt & +5.99\% & +3.77\% & +3.55\% \\
\bottomrule
\end{tabular}

\vspace{2mm}
{\footnotesize 
\textsuperscript{\dag} Uniform 0.05m voxel size across all regions. \\
\textsuperscript{\ddag} Fixed scales \{0.05m, 0.10m, 0.20m\} with simple concatenation, 
no adaptive fusion. \\
VoxAdapt: Learned attention-based scale selection and adaptive feature aggregation 
using the same three scales.
}
\end{table}
```

---

## 📝 **Paragraph to Accompany Table**

**Place this after the table in your Results section:**

```
Table X presents performance comparison across KITTI difficulty levels, revealing 
several key findings. First, VoxAdapt consistently outperforms both baselines across 
all difficulty levels: Easy (+4.80pp over single-scale), Moderate (+2.67pp), and 
Hard (+2.35pp). Second, the Naive Multi-Scale approach actually degrades performance 
compared to Fixed Single-Scale (-1.97pp on Easy, -2.47pp on Moderate), validating 
our hypothesis that simply using multiple scales without adaptive fusion is 
insufficient and can introduce noise. Third, VoxAdapt shows the strongest gains on 
Easy difficulty (+4.80pp), where abundant point clouds benefit most from adaptive 
scale selection, while still providing consistent improvements on challenging 
scenarios (Hard: +2.35pp). This cross-difficulty consistency demonstrates VoxAdapt's 
robustness to varying object conditions including occlusion, truncation, and distance.
```

---

## 🔑 **Key Insights to Emphasize**

### In your discussion:

1. **VoxAdapt > Single-Scale > Naive Multi-Scale**
   - Shows learned fusion is essential
   - Naive approach actually hurts performance

2. **Consistent across difficulties**
   - Easy: +4.80pp (+5.99% relative)
   - Moderate: +2.67pp (+3.77% relative)
   - Hard: +2.35pp (+3.55% relative)

3. **Larger gains on easier cases**
   - More points → more benefit from adaptive scales
   - Still robust on hard cases (occluded/truncated)

---

## ✅ **FINAL RECOMMENDED CAPTION** (Copy-Paste Ready)

```
Table X: 3D object detection performance comparison across KITTI difficulty levels 
for Car detection (AP@IoU=0.7, 40 recall points). All methods use SECOND backbone 
trained for 5 epochs with identical hyperparameters (batch size 6, AdamW lr=0.001). 
Fixed Single-Scale uses uniform 0.05m voxels. Naive Multi-Scale employs fixed 
{0.05m, 0.10m, 0.20m} scales with simple concatenation. VoxAdapt (ours) uses learned 
adaptive scale selection with attention-based fusion. VoxAdapt achieves consistent 
improvements across all difficulties: +4.80% (Easy), +2.67% (Moderate), +2.35% (Hard) 
over single-scale baseline. Notably, naive multi-scale without adaptive fusion 
degrades performance (-1.97% Easy, -2.47% Moderate) compared to single-scale, 
validating the necessity of learned scale adaptation. Best results in bold.
```

**Word count:** 107 words  
**Perfect for:** IEEE Transactions, Elsevier journals, Springer journals

---

Good luck with your paper! 🎓📊
