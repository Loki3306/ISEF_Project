# Block 4 — S1 Results: Static Nonlinear Cross-Electrode Mixing

## 1. Experiment Overview
- **Objective:** Determine if the 8.91 percentage point collapse in E1 (wearable montage) was due to the spatial constraint of EEGNet's linear depthwise convolution.
- **Hypothesis Tested:** Replacing the static linear mixing with a more expressive static nonlinear mixing (Spatial MLP) will recover performance if the mapping between wearable electrodes and cortical sources is highly non-linear.
- **Model:** `EEGNetS1`
- **Spatial Operation:** 2-layer MLP applied independently across the 8 physical electrodes (`Linear(8, 16) -> GELU -> Linear(16, 2)`).
- **Control Variables:** E1 wearable montage (`[0, 33, 6, 41, 22, 59, 15, 52]`), preprocessing, temporal encoder, loss, and LOSO evaluation protocol were completely frozen.

## 2. Parameter Control Verification
- **S0 (Baseline EEGNet):** 1,249 total parameters (128 spatial)
- **S1 (EEGNetS1):** 1,299 total parameters (178 spatial)
- **Delta:** +50 parameters. The parameter increase is so negligible (~4%) that it eliminates capacity inflation as a confound. S0+ capacity control was deemed mathematically unnecessary for this comparison.

## 3. Results (Full 18-Fold LOSO)
Based on all 18 subjects:

| Subject | S1 Accuracy (10s) |
| :--- | :--- |
| S1 | 61.67% |
| S2 | 69.67% |
| S3 | 51.67% |
| S4 | 63.67% |
| S5 | 61.33% |
| S6 | 51.00% |
| S7 | 54.33% |
| S8 | 60.67% |
| S9 | 56.33% |
| S10 | 67.67% |
| S11 | 54.33% |
| S12 | 59.67% |
| S13 | 59.33% |
| S14 | 63.00% |
| S15 | 68.67% |
| S16 | 62.33% |
| S17 | 33.67% |
| S18 | 69.33% |
| **Final Average (18 subjects)** | **59.35%** |

## 4. Scientific Interpretation
- **S0 (E1 Baseline):** 59.28%
- **S1 (Nonlinear Spatial):** 59.35%

According to the established Interpretation Matrix (Section 9):
**Case C: S1 ≈ S0.** 

**Conclusion:** There is **no evidence that static non-linear spatial mixing helps.** 

The non-linear capacity did not unlock any hidden spatial mappings. The performance is structurally identical to the linear baseline. This proves that the strict linearity of the spatial filter was *not* the bottleneck causing the 8.91% collapse in the wearable montage.

## 5. Next Steps
The failure of static nonlinear mixing leaves two primary hypotheses for the wearable montage collapse:
1. **Dynamic Noise (H2 variant):** The noise or optimal mapping in the wearable electrodes shifts dynamically trial-to-trial, requiring input-dependent spatial weighting (Block 4 - S2: Dynamic Electrode Attention).
2. **Fundamental Information Bottleneck (H1):** The specific E1 wearable montage (`[Fp1, Fp2, F7, F8, P7, P8, TP7, TP8]`) simply does not capture enough stable Auditory Evoked Potentials, regardless of how we mix it.

Pending user approval to proceed to the next stage of the experimental ladder.
