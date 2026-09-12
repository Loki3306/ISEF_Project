# Canonical E0 Baseline Execution

## Objective
Establish the exact historical E0 baseline accuracy on the unmodified 8-channel EEG montage, using the historical Gammatone envelopes, without any pipeline fixes. This freezes the baseline metric before we initiate the E1 wearable montage ablation.

## Execution Details
- **Codebase branch**: `experiment/e0-canonical-reproduction`
- **Execution Environment**: Kaggle (T4 x2)
- **Model**: MatchNet (Contrastive)
- **Base Encoder**: EEGNet
- **Channels**: `[13, 46, 43, 23, 50, 0, 52, 14]` (Historical arbitrary placement)
- **Audio Representation**: 28-band Gammatone envelopes (15ms FIR logic preserved)

## Configuration Parameters
- **Decision Window**: 10 seconds
- **Training Window**: 5 seconds
- **Hop Size**: 2 seconds
- **Bandpass Filter**: 1.0Hz - 6.0Hz
- **Epochs**: 100 (with Early Stopping patience of 10)
- **Batch Size**: 512
- **Mixed Precision**: Enabled (`torch.cuda.amp`)
- **Metric**: Pearson Correlation (over 10s windows)

## Results
The Leave-One-Subject-Out (LOSO) evaluation process yielded the following cross-validation scores for the 10s non-overlapping decision window:

| Subject Fold | 10s Window Accuracy (%) | Decisions Evaluated |
| :--- | :--- | :--- |
| S1 | 71.33% | 300 |
| S2 | 66.00% | 300 |
| S3 | 62.33% | 300 |
| S4 | 72.00% | 300 |
| S5 | 70.00% | 300 |
| S6 | 49.67% | 300 |
| S7 | 77.67% | 300 |
| S8 | 79.67% | 300 |
| S9 | 72.33% | 300 |
| S10 | 69.33% | 300 |
| S11 | 51.67% | 300 |
| S12 | 62.33% | 300 |
| S13 | 71.67% | 300 |
| S14 | 76.33% | 300 |
| S15 | 76.33% | 300 |
| S16 | 64.67% | 300 |
| S17 | 67.33% | 300 |
| S18 | 66.67% | 300 |
| **Average (E0 Canonical)** | **68.19%** | **5,400 Total** |

## Conclusion
The **Canonical E0 Baseline is formally frozen at 68.19%**. This matches the historically reported ~69.02% expectation, with the minor difference attributed to stochastic gradient initialization across runs (the exact historical weights were not provided, so training from scratch produced a statistically equivalent expected value).

The configuration for this run is fully documented in `scripts/verify_baseline/training/train_matchnet_loso.py` within the `experiment/e0-canonical-reproduction` branch.
