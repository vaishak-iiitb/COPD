# Data Fraction Comparison: 20% vs 100%

This analysis compares model performance when training with 20% versus 100% of available data for SVM and MLP classifiers, using deterministic sampling (`random_state=42`).

## Executive Summary

Training on the full dataset yields consistent improvements: SVM gains ~3% ROC AUC and produces more stable probability calibration, while MLP shows smaller but meaningful gains in cross-validation F1 (~1%). The 20% constraint particularly impacts SVM through feature pruning and calibration shifts, while MLP's oversampling strategy provides greater resilience to reduced data.

---

## SVM Results

**Configuration:** RBF kernel SVC with `C=2.0`, `gamma='scale'`, `class_weight='balanced'`, standardized inputs

| Metric | 100% Data | 20% Data | Delta |
|--------|-----------|----------|-------|
| Validation ROC AUC | 0.8426 | 0.8136 | -0.029 |
| Validation F1 @ 0.5 | 0.7093 | 0.6648 | -0.045 |
| Best F1 (optimized threshold) | 0.7193 @ 0.340 | 0.7090 @ 0.180 | -0.010 |
| Engineered features retained | 91 / 191 | 83 / 172 | -8 |
| Final feature dimensionality | 115 | 107 | -8 |

**Key Observations:**
- **Calibration shift:** Optimal decision threshold drops dramatically from 0.340 to 0.180 with reduced data, indicating the model becomes more conservative in assigning positive class probability
- **Feature attrition:** Correlation-based pruning removes more engineered features at 20% (83 vs 91 retained), weakening the nonlinear signal available to the RBF kernel
- **Margin quality:** Smaller training sets produce less stable support vectors and decision boundaries, directly impacting the ROC AUC gap

---

## MLP Results

**Configuration:** Early-stopping MLPClassifier with median imputation, standardization, custom oversampling wrapper, 3-fold RandomizedSearchCV

| Metric | 100% Data | 20% Data | Delta |
|--------|-----------|----------|-------|
| Best CV F1 | 0.7172 | 0.7086 | -0.009 |
| OOF threshold F1 | 0.7175 @ 0.475 | 0.7100 @ 0.475 | -0.008 |
| Hidden layers (best) | (64,) | (128,) | — |
| Alpha (L2 reg) | 0.001 | 0.0001 | — |
| Learning rate strategy | constant | adaptive | — |
| Engineered features | 36 | 36 | 0 |

**Key Observations:**
- **Stable threshold:** Optimal decision point remains consistent at ~0.475 across both data fractions, suggesting robust probability calibration from oversampling
- **Hyperparameter adaptation:** The search automatically compensates for reduced data by selecting larger capacity (128 vs 64 units), lower regularization, and adaptive learning rates
- **Fixed feature engineering:** Unlike SVM, the feature set remains constant (36 features), preserving model capacity regardless of training size

---

## Cross-Model Insights

### Feature Stability
Top predictive features remained consistent across both fractions:
- `hemoglobin_level`, `height_cm`, `log1p_ggt_enzyme_level`
- `weight_kg`, `triglycerides`
- BMI derivatives, waist-to-height ratio, lipid ratios

### Class Imbalance Handling (~37% positive class)
- **SVM:** Uses `class_weight='balanced'` but remains sensitive to per-fold minority representation variance at 20%, particularly affecting probability calibration
- **MLP:** Oversampling strategy provides more consistent minority class exposure, reducing calibration variance

### Data Efficiency
- **SVM:** High sensitivity to reduced data due to margin-based learning and pairwise distance computations; requires larger samples for stable support vector identification
- **MLP:** More resilient through oversampling, early stopping, and gradient-based optimization that aggregates signal across mini-batches

---

## Recommendations

### When Using Full Data
- **SVM:** Expect strong performance with well-calibrated probabilities; default threshold of 0.340 provides optimal F1
- **MLP:** Smaller networks (64 units) with constant learning rates suffice; regularization can be moderate (α=0.001)

### When Constrained to 20%
1. **For SVM:**
   - Aggressively tune decision thresholds (expect values 0.15–0.25)
   - Relax correlation-based feature pruning to retain more engineered interactions
   - Apply post-hoc probability calibration (isotonic regression or Platt scaling)
   - Consider increasing `C` to allow more flexible decision boundaries

2. **For MLP:**
   - Expand hyperparameter search iterations to find optimal capacity-regularization tradeoffs
   - Prefer larger networks (100–150 units) with lower regularization
   - Use adaptive learning rates for better convergence
   - Increase cross-validation folds (5+) to reduce per-fold variance

3. **General:**
   - Validate threshold optimization on holdout sets to avoid overfitting
   - Monitor probability calibration curves to detect miscalibration early
   - Consider ensemble methods to reduce variance from smaller samples

---

## Technical Notes

### Why SVM Degrades More
1. **Margin estimation:** RBF kernel relies on pairwise distances; fewer samples create noisier similarity matrices
2. **Support vector sparsity:** Reduced data increases variance in which points become support vectors
3. **Probability calibration:** Platt scaling (internal to `probability=True`) requires sufficient positive/negative pairs for stable sigmoid fitting
4. **Feature interactions:** Engineered features are pruned more aggressively at 20% due to correlation threshold behavior with smaller covariance estimates

### Why MLP Is More Resilient
1. **Oversampling:** Creates synthetic training signal, partially offsetting reduced real samples
2. **Gradient aggregation:** Mini-batch optimization pools information across samples more effectively than margin-based methods
3. **Early stopping:** Validation-based stopping prevents overfitting even with limited data
4. **Fixed features:** Consistent feature engineering preserves input dimensionality and learned representations