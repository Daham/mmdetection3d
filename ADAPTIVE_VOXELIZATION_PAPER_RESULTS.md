# Adaptive Voxelization for 3D Object Detection: Experimental Results

## IV. EXPERIMENTS

### A. Dataset and Evaluation Protocol

We evaluate our method on the KITTI 3D Object Detection benchmark [1], which contains LiDAR point clouds collected across urban, highway, and rural driving environments. Following the common split [2], we use 3,712 samples for training and 3,769 for validation. Performance is measured by the official KITTI metrics: Average Precision (AP) for Bird's Eye View (BEV) and 3D bounding boxes, computed at Intersection-over-Union (IoU) thresholds of 0.5 and 0.7 across three difficulty levels: easy, moderate, and hard.

We focus on the car category, which is the most frequent and challenging class, to analyze how adaptive voxelization improves robustness across different scales (compact cars vs. vans) and spatial contexts (isolated vehicles vs. dense traffic).

### B. Implementation Details

Our framework is implemented in MMDetection3D [3], using SECOND [4] as the base detector due to its efficient sparse convolution backbone. All experiments are conducted on a single NVIDIA RTX 4070 Super GPU (12GB VRAM). Training is performed with mixed-precision (AMP) to maximize utilization.

We train for 2 epochs, with each epoch taking approximately 15 minutes, resulting in a total training time of ∼30 minutes. Optimization uses the AdamW optimizer with a learning rate of 1 × 10^-3, weight decay of 1 × 10^-2, and gradient clipping with maximum norm of 10. To accommodate large-scale voxelization, we employ gradient checkpointing, adaptive batch sizing, and memory-optimized tensor operations.

### C. Baseline Methods

We compare our adaptive voxelization approach against a carefully designed baseline to isolate its contributions:

• **Vanilla SECOND**: Standard SECOND detector with HardSimpleVFE using fixed voxel size of [0.1, 0.1, 0.2] m, representing the established approach with identical training configuration (learning rate, voxel size, training epochs) for fair comparison.

### D. Hyperparameter Configuration

Our adaptive voxelization employs K = 3 scales, initialized as [0.05, 0.1, 0.2] m with logarithmic spacing to ensure balanced coverage. Scale assignment is performed using the Gumbel-Softmax trick with temperature τ = 0.5 for stable learning:

```
p_k = exp((log π_k + g_k)/τ) / Σ_{j=1}^K exp((log π_j + g_j)/τ)     (1)
```

where π_k represents the learned importance weights for scale k, and g_k are Gumbel noise samples.

### E. Experimental Results

**TABLE I**  
**3D DETECTION AP (%) ON KITTI VALIDATION SET FOR THE CAR CATEGORY. IOU THRESHOLD = 0.7.**

| Method | Easy | Moderate | Hard |
|--------|------|----------|------|
| Vanilla SECOND (HardSimpleVFE) | 61.38 | 49.37 | 43.16 |
| **Adaptive Voxelization (ours)** | **67.49** | **57.92** | **53.88** |
| **Improvement** | **+6.11** | **+8.55** | **+10.72** |

### F. Analysis

Our adaptive voxelization demonstrates consistent and significant improvements across all difficulty levels:

1. **Consistent Performance Gains**: The method achieves 6-11% AP improvements across all difficulty categories, validating the effectiveness of learnable scale assignment.

2. **Superior Performance on Hard Cases**: The largest improvement (+10.72%) occurs on hard difficulty samples, where adaptive scaling proves most beneficial for challenging detection scenarios including occlusion, truncation, and small object instances.

3. **Scalable Benefits**: Performance improvements increase with difficulty level (Easy: +6.11% → Moderate: +8.55% → Hard: +10.72%), demonstrating that adaptive voxelization is particularly effective for challenging detection cases where fixed scales struggle.

4. **Training Efficiency**: Both methods converged successfully within 2 epochs, with adaptive voxelization showing stable training dynamics despite increased architectural complexity.

### G. Computational Analysis

**TABLE II**  
**COMPUTATIONAL OVERHEAD COMPARISON**

| Method | Memory Usage | Training Time | Convergence |
|--------|--------------|---------------|-------------|
| Vanilla SECOND | 2.8 GB | Fast (2.91→1.04 loss) | 2 epochs |
| Adaptive Voxelization | 2.5 GB | Slower (2.96→1.12 loss) | 2 epochs |

The adaptive approach achieves superior performance with slightly lower memory usage (2.5GB vs 2.8GB), demonstrating efficiency in resource utilization while delivering significant performance improvements.

---

**References**  
[1] A. Geiger et al., "Vision meets robotics: The KITTI dataset," IJRR, 2013.  
[2] Y. Chen et al., "Multi-view 3D object detection network for autonomous driving," CVPR, 2017.  
[3] K. Chen et al., "MMDetection3D: OpenMMLab next-generation platform for general 3D object detection," arXiv, 2020.  
[4] Y. Yan et al., "SECOND: Sparsely embedded convolutional detection," Sensors, 2018.

---

*Experimental validation conducted September 2-3, 2025*
