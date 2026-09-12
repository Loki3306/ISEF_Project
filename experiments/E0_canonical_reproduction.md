# E0 — Canonical Reproduction Experiment

## OBJECTIVE
Establish a scientifically defensible canonical E0 baseline before testing the wearable montage constraint. The primary goal is to reproduce the historical ~69.02% result and identify the exact evaluation window that produced it, resolving any conflicting documentation.

## A. Historical Evidence & Forensic Audit
The historical 69.02% result was imported as a reference benchmark. However, the documentation for this benchmark is internally inconsistent regarding the evaluation window (3s, 10s, or 30s). The only way to confidently establish the E0 baseline is to execute the mathematically exact historical pipeline and observe which window length produces ~69.02%.

## B. Exact Recovered Configuration
1. **Dataset:** DTU Auditory Attention Decoding (18 normal hearing subjects)
2. **Subject count:** 18
3. **Trial count:** 60 trials per subject
4. **EEG sampling rate:** 64 Hz
5. **EEG preprocessing:** 50 Hz notch, 0.1 Hz high-pass, EOG denoised, Common Average Reference (CAR)
6. **EEG normalization:** Per-trial Z-score standardization (computed *before* windowing)
7. **Historical channel indices:** `[13, 46, 43, 23, 50, 0, 52, 14]`
8. **Channel names:** `['C5', 'FCz', 'FC6', 'P9', 'C6', 'Fp1', 'TP8', 'T7']`
9. **Audio representation:** Gammatone Envelopes
10. **Gammatone implementation:** `scipy.signal.gammatone(..., 'fir')` (Flawed 15ms truncation)
11. **Number of bands:** 28
12. **Audio sampling rate:** 64 Hz
13. **Audio normalization:** Per-trial Z-score standardization (computed *before* windowing)
14. **EEG training window length:** 5.0 seconds
15. **EEG training stride:** 2.0 seconds
16. **Audio training window length/stride:** 5.0s / 2.0s
17. **Evaluation window length:** Evaluated simultaneously at [2s, 5s, 10s, 20s, 30s]
18. **Evaluation stride:** Non-overlapping (stride = window length)
19. **Prediction aggregation method:** Majority voting / sum of pairwise similarities per window
20. **Similarity metric:** Pearson Correlation & Cosine Similarity
21. **EEG encoder architecture:** EEGNet
22. **Audio encoder architecture:** 1D-CNN (built into ContrastiveMatchNet)
23. **Embedding dimensionality:** 64
24. **Loss:** Contrastive Margin Loss (`margin=0.1`)
25. **Optimizer:** Adam (Weight Decay: 1e-4) with AMP Scaler
26. **Learning rate:** 1e-3
27. **Batch size:** 512
28. **Number of epochs:** 100
29. **Early stopping/checkpoint selection:** Patience of 10 epochs on Validation Accuracy. Final evaluation uses the checkpoint with the highest validation accuracy.
30. **Train/validation/test split:** 1 Subject out for Testing. Remaining subjects: 90% Training / 10% Validation (split by trial index).
31. **LOSO procedure:** Strict 18-Fold Leave-One-Subject-Out cross-validation.
32. **Random seed:** 42 (used for dataset shuffling and validation splitting)
33. **Final accuracy calculation:** `Total Correct / Total Decisions` across all 18 test folds. Correct decision = `Similarity(Z_eeg, Z_A) > Similarity(Z_eeg, Z_B)`.
34. **Other details:** None.

## C. Contradictions Found
The documentation implies a 10s evaluation window for the 69.02% benchmark, but other sources suggest 3s or 30s. 

## D. Resolution of Evaluation-Window Discrepancy
[PENDING Kaggle Execution: We will observe the E0 accuracy across all windows (2s, 5s, 10s, 20s, 30s) to definitively locate the ~69.02% benchmark.]

## E. Exact Command Used to Run E0
```bash
python scripts/verify_baseline/training/train_matchnet_loso.py --channels 13 46 43 23 50 0 52 14
```

## F. Dataset Integrity Checks
- **Audio Map:** `audio_mapping.json` verified to exist and map `wavA` to attended audio.
- **Audio Envelopes:** `gammatone_envelopes.pkl` verified to use identical FIR generation script as historical baseline.

## G. LOSO Integrity Checks
- The script uses `folds = list(iter_leave_one_subject_out(all_paths))` spanning all 18 subjects.
- No leakage exists between training subjects and the held-out test subject.

## H. Result
[PENDING Kaggle Execution]

## I. Comparison against historical ~69.02%
[PENDING Kaggle Execution]

## J. Any remaining discrepancy
[PENDING Kaggle Execution]
