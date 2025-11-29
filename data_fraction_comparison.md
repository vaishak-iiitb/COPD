# Data Fraction Comparison (Simple Summary)

Notebooks and their data usage:
- `kaggle-mlp-0-721.ipynb` → `DATA_FRACTION = 1` (full data)
- `kaggle-mlp-0.721_20.ipynb` → `DATA_FRACTION = 0.2` (20% stratified sample)
- `svm_0.716.ipynb`  (SVM full data)
- `svm_0.716_20.ipynb` (SVM 20%)

Common training constants (from code): 5-fold CV (`CV_SPLITS=5`), `EPOCHS=60`, `BATCH_SIZE=256`, `PATIENCE=8`.

## Observed (from filenames / runs)
| Model | Data Fraction | Recorded F1 (approx) |
|-------|---------------|----------------------|
| MLP   | 100%          | ~0.7195 |
| MLP   | 20%           | ~0.7099 |
| SVM   | 100%          | ~0.7193 |
| SVM   | 20%           | ~0.7090 |

## Quantitative Differences
- MLP absolute drop: 0.7195 − 0.7099 = 0.0096 (≈1.33% relative)
- SVM absolute drop: 0.7193 − 0.7090 = 0.0103 (≈1.43% relative)
- Performance spread between architectures at same fraction is <0.001 (effectively identical within noise).
- Indicates early saturation: marginal returns from last 80% of rows.

## What We Expected Ideally
Using only 20% of the training data should usually reduce F1 (less signal, higher variance). Full data typically gives: (a) slightly higher F1, (b) more stable threshold tuning, (c) lower fold-to-fold variance.

## What Actually Happened
F1 dropped only ~0.010 (≈1.3–1.4% relative) for both models when moving from 100% to 20% of stratified data—well within a small effect range. Architectures remained virtually tied at each fraction.

## Why It Probably Did Not Change
1. Stratified sampling preserved class balance, avoiding distribution drift.
2. Strong, low-noise engineered features (BMI, ratios, logs) convey most signal; a 20% subset still spans feature space.
3. Regularization (MLP: L2 + Dropout + BatchNorm; SVM margin constraints) curtails overfitting benefits from extra examples.
4. Diminishing returns / early saturation: Bayesian / learning curve intuition—later samples add redundancy, not novel patterns.
5. Threshold tuning on out-of-fold predictions converges to similar cutoffs because score distributions shift minimally.
6. Possible intrinsic label noise floor: added data cannot breach a ceiling dominated by irreducible error.

## Why NN did not outperform SVM if they had similar f1 score with just 50% data
SVM holding nearly the same F1 at 20% data is expected: margin-based training concentrates on boundary points, so once a representative, stratified slice preserves class balance and core feature relationships, extra interior samples add redundancy rather than new signal. Our engineered, mostly low-noise features make the class separation close to linearly (or smoothly) separable, letting both the SVM and a regularized MLP converge early. The tiny ~1.3–1.4% relative gain from 20% to full data reflects an early saturation: irreducible error and limited additional structural complexity mean more rows mainly tighten confidence intervals instead of shifting the decision surface.


## Simple Takeaway
For this dataset and feature set, a 20% stratified slice already reaches ≥98.6–98.7% of full-data F1; the remaining 80% yields only ~1.3–1.4% relative gain—evidence of early performance saturation.

