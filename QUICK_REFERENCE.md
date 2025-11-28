# 🎯 VALIDATION EXPERIMENT - QUICK REFERENCE

## One Command to Rule Them All

```bash
cd /home/daham/mmdetection_project/mmdetection3d
./run_validation_experiment.sh
```

**Wait ~18 hours**, then:

```bash
python tools/analysis_tools/analyze_validation_results.py \
    --work-dir work_dirs/validation_experiment \
    --output VALIDATION_RESULTS.md

cat VALIDATION_RESULTS.md
```

---

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Baseline_03 vs Baseline_01 | **> +2.0%** | ? |
| Statistical Significance | **p < 0.05** | ? |
| Training Stability (Std) | **< 1.0%** | ? |

**If all ✅:** Proceed with paper improvements  
**If any ❌:** Debug method first

---

## What's Running

**9 training runs total (3 baselines × 3 seeds):**

1. Baseline_01: Single-scale HardVFE (control)
2. Baseline_02: Fixed multi-scale (no learning)
3. **Baseline_03: Your adaptive method** ← Should be best

**Each run:** 80 epochs × ~90s/epoch ≈ 2 hours

---

## Monitor Progress

```bash
# See what's running
ps aux | grep train.py

# Watch current run
tail -f work_dirs/validation_experiment/baseline_03_adaptive_seed0/training.log

# Check all completed results
grep "KITTI/Car_3d_moderate" work_dirs/validation_experiment/*/training.log | sort
```

---

## Files Created

✅ `configs/second/validation_baseline_01_single_scale_80ep.py`  
✅ `configs/second/validation_baseline_02_fixed_multiscale_80ep.py`  
✅ `configs/second/validation_baseline_03_adaptive_80ep.py`  
✅ `run_validation_experiment.sh`  
✅ `tools/analysis_tools/analyze_validation_results.py`  
✅ `VALIDATION_EXPERIMENT_GUIDE.md` ← Full instructions  
✅ `PAPER_IMPROVEMENT_ACTION_PLAN.md` ← Next steps if successful

---

## Emergency Stops

**Stop current training:**
```bash
pkill -f train.py
```

**Resume from checkpoint:**
```bash
python tools/train.py configs/second/validation_baseline_03_adaptive_80ep.py \
    --resume \
    --work-dir work_dirs/validation_experiment/baseline_03_adaptive_seed0
```

---

## Expected Results

**Good scenario:**
```
Baseline_01: 72.3 ± 0.4%
Baseline_03: 76.5 ± 0.7%  (+4.2%, p=0.003) ✅
→ PROCEED with paper
```

**Bad scenario:**
```
Baseline_01: 72.3 ± 0.4%
Baseline_03: 72.8 ± 2.1%  (+0.5%, p=0.42) ❌
→ FIX method first
```

---

## Read More

- **Quick Guide:** `VALIDATION_SETUP_COMPLETE.md`
- **Detailed Instructions:** `VALIDATION_EXPERIMENT_GUIDE.md`
- **Paper Plan:** `PAPER_IMPROVEMENT_ACTION_PLAN.md`

---

**Status:** Ready to run  
**Next:** Execute `./run_validation_experiment.sh`  
**Then:** Check back in ~18 hours
