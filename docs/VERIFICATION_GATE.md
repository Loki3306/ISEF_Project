# CODE & BASELINE VERIFICATION GATE

## 1. Repository / Code Audit

The actual implementation was verified by auditing `train_matchnet_loso.py` and `baselines/ridge_aad.py` in the reference repository.

| Component | File | Function/Class | Input | Output | Notes |
|---|---|---|---|---|---|
| Dataset Loading | `ridge_aad.py` | `load_subject_examples` | `.mat` path | List of `TrialExample` | Loads full 50s trials into memory. |
| EEG Loading | `train_matchnet_loso.py` | `prepare_dataset` | `TrialExample` | EEG numpy array | Extracts only `args.channels`. |
| Audio Loading | `train_matchnet_loso.py` | `get_mapping_data` | None | `mapping`, `envelopes` | Reads precomputed `gammatone_envelopes.pkl`. |
| Channel Selection | `train_matchnet_loso.py` | `argparse` | CLI args | `[13, 46, 43, 23, 50, 0, 52, 14]` | Defaults to old MatchNet baseline. |
| Preprocessing | `train_matchnet_loso.py` | `butter_bandpass_filter` | EEG array | 1-6 Hz array | Applied to the entire trial at once. |
| Normalization | `train_matchnet_loso.py` | `normalize_array` | Array | Z-scored Array | **Applied per-trial before windowing (uses future samples).** |
| Windowing | `train_matchnet_loso.py` | `chunk_trial` | 50s trial | 5s chunks (2s hop) | Training uses overlapping windows; evaluation uses non-overlapping. |
| Label Construction| `train_matchnet_loso.py` | `DataLoader` | X, YA, YB | Batch | Implicit contrastive setup: YA is always attended, YB is unattended. |
| EEG Model | `train_matchnet_loso.py` | `ContrastiveMatchNet` | EEG tensor | Z_eeg | Defaults to `eegnet` encoder. |
| Audio Model | `train_matchnet_loso.py` | `ContrastiveMatchNet` | Audio tensor | Z_audio | Handled inside MatchNet. |
| Similarity | `train_matchnet_loso.py` | `contrastive_loss` / `evaluate_model` | Z_eeg, Z_a, Z_b | Cosine / Pearson | Training uses `contrastive_loss`, evaluation compares metrics. |
| LOSO Split | `train_matchnet_loso.py` | `iter_leave_one_subject_out` | All subjects | Folds | **DISCREPANCY: Script is hardcoded to `folds = list(...)[:1]`.** |
| Validation Split | `train_matchnet_loso.py` | `train_matchnet_loso` | Train trials | 90/10 split | Split occurs at the trial level (no leakage across folds). |
| Metrics | `train_matchnet_loso.py` | `evaluate_model` | Z_eeg, Z_a, Z_b | Accuracy % | If `sim_a > sim_b`, +1. If equal, +0.5. |

---

## 2. Verify the Historical Baseline

- **Exact model architecture:** `ContrastiveMatchNet` (EEGNet encoder, 28-band Audio encoder, 64-D latent).
- **Exact EEG channels:** `[13, 46, 43, 23, 50, 0, 52, 14]` (Historical MatchNet).
- **Exact audio representation:** Precomputed Gammatone Envelopes (28 bands).
- **Exact window length:** Training: 5s window, 2s hop. Evaluation: 2s, 5s, 10s, 20s, 30s non-overlapping windows.
- **Exact preprocessing:** 1–6 Hz Butterworth bandpass on EEG.
- **Exact normalization:** Per-trial Z-score standardization (causal leakage present, but consistent for offline baseline).
- **Exact loss:** `contrastive_loss` with `margin=0.1`.
- **Optimizer:** Adam (LR 1e-3, Weight Decay 1e-4) with AMP Scaler.
- **Batch size:** 512.
- **Epochs / Early Stopping:** Max 100 epochs, Patience 10 on validation accuracy.
- **LOSO procedure:** One subject strictly held out for testing. 

---

## 3. Identify the Current Experimental Gaps

### VERIFIED (Code Audit)
- The pipeline correctly isolates attended vs. unattended speech into a Siamese/Contrastive training setup.
- The default channels match the historical baseline exactly `[13, 46, 43, 23, 50, 0, 52, 14]`.
- The dataset loading strictly maintains subject separation (except for the validation split which correctly uses remaining training trials).
- Normalization is applied per-trial *before* chunking, which implies it uses future information from the trial (safe for offline evaluation, unacceptable for real-time wearables).

### NOT VERIFIED (Requires Execution)
- The actual ~69.02% LOSO result.
- The tensor shapes and batch statistics of a real dataloader batch.
- Whether the audio mappings precisely align `wavA` and `wavB` correctly without off-by-one errors.

### DISCREPANCIES (Code vs Expected)
- **LOSO Truncation:** `train_matchnet_loso.py` line 201 is hardcoded to `folds = list(iter_leave_one_subject_out(all_paths))[:1]`. This means the script only evaluates Subject 1 and quits. It does NOT produce a full 18-subject LOSO result by default.

### BLOCKERS
- We cannot proceed to Block 4 until we remove the `[:1]` truncation and execute a full LOSO run to mechanically verify the 69.02% baseline performance.

---

## 4. Experimental Baseline Ladder

To prevent attributing accuracy changes to confounding factors, the experiments must be run in this sequence:

- **E0 (Historical Baseline):** Old channels `[13, 46, 43, 23, 50, 0, 52, 14]` + EEGNet MatchNet. (Validates reproduction).
- **E1 (New Montage):** Wearable channels `[0, 33, 6, 41, 22, 59, 15, 52]` + EEGNet MatchNet. (Isolates channel drop/shift penalty).
- **E2 (Temporal Change):** Wearable channels + Block 2 MSCA Temporal Encoder. (Isolates temporal architecture effect).
- **E4 (Spatial Change):** Block 4 spatial modeling (Only after E2 is stable).

---

## 5. Decision Gate

### 🟡 YELLOW
**Baseline mostly works but there are unresolved discrepancies.**
Do NOT start Block 4 yet. The codebase was audited and the pipeline logic is sound, but the script is currently truncated to 1 fold and has not been mechanically verified end-to-end to yield ~69%. 

**Required Action:** Execute the provided `verify_baseline.py` script on Kaggle to generate the mechanical batch tensor stats and run a full LOSO evaluation for E0.
