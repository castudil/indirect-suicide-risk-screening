# Audit Report

**Repository:** `indirect-suicide-risk-screening`

**Date:** 24 September 2026

**Execution Environment:** macOS (Apple Silicon / ARM64), Python 3.14.6 (`.venv`), Fish shell

**Objective:** Independent replication of the experiments and comparation against the provided.


## 1. Chronological Execution Procedure

The audit was executed sequentially using the following steps:

### Step 0: Backup original results

Before running any scripts or cleaning directories, the existing results were copied to a dedicated directory to preserve the authors' baseline:

```fish
cp -r results results_org

```

### Step 1: Target directory preparation

The target directory was ensured and wiped to eliminate any residual files:

```fish
mkdir -p results
rm -f results/*

```

### Step 2: Stochastic noise robustness simulation

The label-noise simulation (100 repetitions) was executed to generate the inputs required by experiment P1-E8:

```fish
python src/e2_label_noise_robustness.py

```

### Step 3: Execution of primary experiment pipeline

The full suite of Phase 1 experiments (P1-E1 through P1-E10) was executed:

```fish
python src/p1_experiments.py

```

### Step 4: Internal cryptographic verification

The integrity of the newly generated files was validated against the SHA-256 hashes in `RUN_MANIFEST.json`:

```fish
python src/_audit.py verify

```

### Step 5: Systematic file comparison against the originals

A binary comparison was conducted matching every newly generated CSV in `results/` against its counterpart in `results_org/`:

```fish
python -c '
import os, filecmp, glob
for f in sorted(glob.glob("results/*.csv")):
    base = os.path.basename(f)
    ref = os.path.join("results_org", base)
    if not os.path.exists(ref):
        print(f"[NEW]       {base}")
    elif filecmp.cmp(f, ref, shallow=False):
        print(f"[IDENTICAL] {base}")
    else:
        print(f"[DIFF]      {base}")
'

```

### Step 6: Line-by-line inspection of discrepancies

For every file flagged with `[DIFF]`, an unified diff was inspected:

```fish
diff -u results_org/P1E4_selection_stability.csv results/P1E4_selection_stability.csv
diff -u results_org/P1E5_optimism_bootstrap.csv results/P1E5_optimism_bootstrap.csv
diff -u results_org/P1E6_distal_comparator.csv results/P1E6_distal_comparator.csv
diff -u results_org/P1E7_calibration_advanced.csv results/P1E7_calibration_advanced.csv

```

## 2. Verification and Comparison Results

### Internal Audit Verification (Step 4)

All generated artifacts matched their registered cryptographic signatures:

```text
Summary: 16 intact, 0 altered, 0 missing.

```

### Baseline Comparison Summary (Step 5)

```text
[IDENTICAL] E2_crossover_points.csv
[IDENTICAL] E2_label_noise_curves.csv
[IDENTICAL] P1E1_loso_per_school.csv
[IDENTICAL] P1E2_criterion_sensitivity.csv
[IDENTICAL] P1E3_net_benefit_curve.csv
[IDENTICAL] P1E3_triage_tiers_nns.csv
[DIFF]      P1E4_selection_stability.csv
[DIFF]      P1E5_optimism_bootstrap.csv
[DIFF]      P1E6_distal_comparator.csv
[DIFF]      P1E7_calibration_advanced.csv
[IDENTICAL] P1E9_scale_totals.csv
[IDENTICAL] P1E10_triage_thresholds.csv

```

Of the 12 CSV tables evaluated, **8 are bit-for-bit identical** (including the 100-repetition stochastic noise simulations in `E2`).


## 3. Summary of Discrepancies (Step 6)

Inspection of the 4 files flagged with `[DIFF]` confirmed that none represent actual analytical or statistical discrepancies:

* **`P1E4_selection_stability.csv`:** Text encoding only. The difference stems from Latin accented characters in two column names (`género` and `repetición`). All numerical selection frequencies and sign consistencies are 100% identical.
* **`P1E5`, `P1E6`, `P1E7`:** Negligible floating-point differences. All primary model metrics, classifications, and clinical findings remain identical.
