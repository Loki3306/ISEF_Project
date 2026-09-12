# E1 — Wearable Montage Experiment

## 1. Research Question
Does replacing the historical 8-channel EEG montage with the intended wearable 8-channel montage change strict LOSO performance when everything else remains unchanged?

## 2. Hypothesis
The wearable montage (frontal + temporal + parietal layout) captures sufficient spatial auditory attention features such that performance will either be maintained or slightly improved compared to the historical montage, which was centered heavily on the motor cortex.

## 3. E0 Configuration (Reference)
- **Model:** Contrastive MatchNet (`eegnet` encoder)
- **Audio:** 28-band Gammatone Envelopes
- **EEG Montage:** `[13, 46, 43, 23, 50, 0, 52, 14]` (C5, FCz, FC6, P9, C6, Fp1, TP8, T7)
- **Window Length:** 5.0 seconds
- **Hop/Stride:** 2.0 seconds
- **Filtering:** 1-6 Hz Bandpass
- **Evaluation Protocol:** Strict 18-Fold Leave-One-Subject-Out (LOSO)
- **Expected Accuracy:** ~69.02% (Historical baseline)

## 4. E1 Configuration
- **EEG Montage:** `[0, 33, 6, 41, 22, 59, 15, 52]` (Fp1, Fp2, F7, F8, P7, P8, TP7, TP8)

## 5. Controlled Variables
Every other aspect of the pipeline was kept identical to E0, including:
- EEG preprocessing, filters, sampling rate (64 Hz)
- Normalization logic
- Audio representation and preprocessing
- MatchNet architecture and embedding dimension
- Loss function (Triplet Margin/InfoNCE proxy)
- Optimizer, learning rate, batch size, epochs
- Validation split (90/10 by trial)
- LOSO cross-validation framework

## 6. Implementation
- **Branch:** `experiment/e1-wearable-montage`
- **Files Changed:** `scripts/verify_baseline/training/train_matchnet_loso.py`
- **Exact Change:** The default `--channels` argument was swapped from `[13, 46, 43, 23, 50, 0, 52, 14]` to `[0, 33, 6, 41, 22, 59, 15, 52]`. A verification print block was added to ensure the shapes entering MatchNet are `[Batch, 8, 320]` and the channel identities map correctly.

## 7. Verification
- **E0 channel indices:** [13, 46, 43, 23, 50, 0, 52, 14]
- **E0 channel names:** ['C5', 'FCz', 'FC6', 'P9', 'C6', 'Fp1', 'TP8', 'T7']
- **E1 (Current) channel indices:** [0, 33, 6, 41, 22, 59, 15, 52]
- **E1 channel names:** ['Fp1', 'Fp2', 'F7', 'F8', 'P7', 'P8', 'TP7', 'TP8']
- **Verified EEG tensor shape entering MatchNet:** `[16, 8, 320]` (for batch size 16)

## 8. Results
*(To be populated after Kaggle execution)*

### Overall
- **E0 accuracy:** ~69.02%
- **E1 accuracy:** [PENDING]
- **Absolute change (Δ):** [PENDING]

### Per Subject (10s Window)
| Subject | E0 accuracy | E1 accuracy | Δ |
|---|---:|---:|---:|
| S1 | | | |
| S2 | | | |
| S3 | | | |
| S4 | | | |
| S5 | | | |
| S6 | | | |
| S7 | | | |
| S8 | | | |
| S9 | | | |
| S10 | | | |
| S11 | | | |
| S12 | | | |
| S13 | | | |
| S14 | | | |
| S15 | | | |
| S16 | | | |
| S17 | | | |
| S18 | | | |

## 9. Interpretation
*(To be populated based on results)*

## 10. Limitations
*(To be populated based on results)*

## 11. Reproduction Instructions
Run the following script on Kaggle to execute the E1 experiment:
```bash
python scripts/verify_baseline/training/train_matchnet_loso.py --channels 0 33 6 41 22 59 15 52
```
