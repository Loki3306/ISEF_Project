# Block 4 — S2 Results (Dynamic Electrode Attention)

## 1. Hypothesis (H2)
**Hypothesis:** The optimum spatial mapping from the 8 wearable electrodes shifts dynamically across trials due to noise, artifacts, or phase inversions. Therefore, **input-dependent spatial weighting** (Dynamic Electrode Attention) should recover attention features better than a static linear or non-linear spatial filter.

## 2. Implementation & Controls
- **Architecture:** `EEGNetS2` (Global Average Pooling over time to create electrode tokens -> Self-Attention across 8 electrodes -> Dynamic spatial filter weights).
- **Control:** Strict parameter matching.
  - S0 (E1 Baseline): 1,249 total parameters
  - S2 (Attention): 1,427 total parameters (+178 parameters)
- **Evaluation:** 4-Subject Smoke Test (S2, S6, S12, S14) representing a mix of best, worst, and average subjects from the E1 baseline.

## 3. Results (4-Subject Smoke Test)
| Subject | E1 Baseline (S0) | S2 (Attention) | Delta |
| :--- | :--- | :--- | :--- |
| S2 (Best) | 69.67% | 69.00% | -0.67% |
| S6 (Worst) | 51.00% | 63.67% | +12.67% |
| S12 (Avg) | 59.67% | 29.00% | **-30.67% (Catastrophic)** |
| S14 (Avg) | 63.00% | 67.67% | +4.67% |
| **Smoke Test Average** | **60.84%** | **57.34%** | **-3.50%** |

## 4. Scientific Interpretation
Based on the established smoke-test gating criteria:

1. **Catastrophic Fold:** Fold S12 collapsed entirely (29.00%, well below chance). This strongly suggests that the attention mechanism either learned an inverted dipole mapping (anti-correlation) for this subject or severely overfit to the training folds, failing to generalize to S12's specific phenotype.
2. **High Variance:** While S2 caused a catastrophic failure on S12, it *massively* improved S6 (51.00% -> 63.67%). This indicates that dynamic weighting *can* uncover signal in noisy subjects, but the current mechanism is dangerously unstable.
3. **Overall Degradation:** The 4-subject average (57.34%) is substantially below the E1 baseline (60.84%).

**Conclusion:** 
The S2 architecture fails the smoke test. While dynamic attention shows promise for rescuing "worst" subjects (like S6), its instability causes catastrophic failure on other subjects. 

Therefore, we **ABORT** the full 18-subject LOSO run. We must now decide whether to refine the dynamic routing (e.g., S3/S4) or pivot to investigating fundamental information bottlenecks in the wearable montage.
