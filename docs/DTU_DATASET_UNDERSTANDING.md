# DTU Dataset â€” Complete Data Understanding

## 1. Dataset overview
- **Subjects:** 18 normal hearing subjects (VERIFIED).
- **Trials:** 60 trials per subject (VERIFIED).
- **Duration:** ~50 seconds per trial (VERIFIED).
- **Sampling:** 512 Hz raw, downsampled to 64 Hz (VERIFIED).
- **Channels:** 66 hardware channels (64 BioSemi + EXG1, EXG2) downselected to 8 channels (VERIFIED).

## 2. Directory / file structure
- Preprocessed data exists as `S{n}_data_preproc.mat` (INFERRED FROM CODE: `preproc_data.m` saves them as such).
- Audio is stored as `.wav` files and later processed into `.pkl` envelopes (VERIFIED).
- An `audio_mapping.json` ties trials to specific `.wav` files (VERIFIED).

## 3. EEG raw data structure
- Based on `preproc_data.m`, the raw data is loaded from `S{n}.mat` containing a `data` structure with continuous recordings before being split.
- `data.eeg` contains the multi-channel recordings.

## 4. EEG preprocessing pipeline
- **MATLAB preprocessing (`preproc_data.m`):**
  1. 50 Hz line noise removal (VERIFIED).
  2. Downsample from 512 Hz to 64 Hz (VERIFIED).
  3. High-pass filter at 0.1 Hz (2nd order butterworth) (VERIFIED).
  4. EOG derivation and regression-based denoising using `co_denoise` (VERIFIED).
  5. Re-referencing to Common Average Reference (VERIFIED).
  6. Splicing into cells (trials) based on `data.event.eeg.sample` (VERIFIED).

## 5. EEG channel inventory
- Raw channels: 66 (BioSemi + EXG).
- `EEG_Training_New` selects 8 peripheral channels: `[13, 46, 43, 23, 50, 0, 52, 14]` corresponding to `Fp1, Fp2, F7, F8, T7, T8, P7, P8` (VERIFIED: documented in `DATASETS.md`).

## 6. Trial structure
- **Trials/Subject:** 60.
- **Representation:** Stored as a cell array of matrices `(T, 66)` in MATLAB (VERIFIED).
- Python side (`train_matchnet_loso.py`) processes these arrays, trimming to the minimum length between EEG and audio envelopes (VERIFIED).

## 7. Event / trigger structure
- Events are loaded from `data.event.eeg.value`.
- `expinfo.attend_mf` defines whether the subject attended Male (1) or Female (2) (VERIFIED in `preproc_data.m`).

## 8. Label semantics
- **CRITICAL FINDING (VERIFIED):** The label (1 or 2) corresponds to the **speaker gender**, NOT the left/right ear or spatial location.
- **Audio Assignment (VERIFIED):** `preproc_data.m` explicitly forces `wavA` to be the attended speaker.
  - `story_names = [expinfo.wavfile_male expinfo.wavfile_female];`
  - `wavA` is set using index `data.event.eeg.value`.
  - Therefore, `wavA` is *always* attended. 
- During MatchNet training (`train_matchnet_loso.py`), `wavA` is loaded as `env_a` and the network is trained with a binary label indicating whether to match `z_E` with `z_A` or `z_B`.

## 9. Audio file structure
- Stereo/Mono raw `.wav` files of Danish stories.
- Audio is processed into envelopes via `extract_gammatone_envelopes.py` and saved as `gammatone_envelopes.pkl` (VERIFIED).

## 10. EEG â†” audio mapping
- Handled by `audio_mapping.json`.
- **CRITICAL CLARIFICATION (KAGGLE vs LOCAL):** The `audio_mapping.json` file is *not* stored inside the massive Kaggle dataset along with the `.mat` files. Instead, it is a lightweight JSON file tracked directly inside the Git repository (`data/audio_mapping.json`). 
- This file explicitly maps each trial (e.g. `S1` -> `trial_0`) to exactly two external `.wav` files.
  - `wavA`: Contains the filename of the *attended* audio (e.g., `marianne_story3_trial_1.wav`).
  - `wavB`: Contains the filename of the *unattended* audio (e.g., `aske_story4_trial_1.wav`).
- `train_matchnet_loso.py` loads this Git-tracked JSON to find the filenames, and then looks up the corresponding processed audio tensors.
- Trials are truncated to `min(eeg_len, wav_len)` (VERIFIED).

## 11. Audio preprocessing / Gammatone pipeline
- **Python side (`extract_gammatone_envelopes.py`):**
  1. 28-band Gammatone (50 Hz to 8000 Hz, ERB spaced) using `scipy.signal.gammatone`.
  2. Absolute envelope extraction with power compression (`** 0.6`).
  3. Low-pass filtered at 8 Hz.
  4. Resampled to 64 Hz.
  5. Yields a `(28, T)` tensor (VERIFIED).
- **NOTE:** `preproc_data.m` originally used `^0.3` compression and then averaged all frequency channels. The Python pipeline (`extract_gammatone_envelopes.py`) preserves the 28 channels for MatchNet.

## 12. Cached dataset format
- Audio envelopes are cached in a single `.pkl` dictionary mapping filenames to `(28, T)` numpy arrays (VERIFIED).
- EEG is read dynamically from `.mat` files in `train_matchnet_loso.py`.

## 13. Exact tensors entering the existing MatchNet
- **EEG (`eeg`):** `(B, 8, T)` (VERIFIED from `train_matchnet_loso.py` channel subsetting).
- **Audio A (`env_a`):** `(B, 28, T)` (VERIFIED).
- **Audio B (`env_b`):** `(B, 28, T)` (VERIFIED).
- **Normalization:** Per-channel Z-score normalization (`normalize_array`) is applied dynamically before windowing (VERIFIED).

## 14. Existing windowing / evaluation procedure
- Handled in `train_matchnet_loso.py` (INFERRED FROM CODE). Windows of 5s (training) and 10s (evaluation) are used.
- Overlapping windows are extracted with a 2s hop length.

## 15. Data integrity findings
- Handled dynamically during data loading (e.g., mismatched lengths truncated via `min()`).
- Further verification required via the `inspect_dtu_dataset.py` Kaggle script.

## 16. Known ambiguities / unresolved questions
- Are there any subjects missing from the `.mat` cache? (To be verified by Kaggle script).
- Does `audio_mapping.json` perfectly cover all 60 trials for all 18 subjects? (To be verified).

## 17. Reference files used
- `EEG_Training_New/preproc_data.m`
- `EEG_Training_New/models/matchnet.py`
- `EEG_Training_New/training/train_matchnet_loso.py`
- `EEG_Training_New/data/extract_gammatone_envelopes.py`
- `EEG_Training_New/docs/DATASETS.md`

## 18. Final verified data flow diagram
```text
RAW EEG (.mat)                   RAW AUDIO (.wav)
      |                                  |
   (MATLAB)                           (Python)
      |                                  |
  preproc_data.m                 extract_gammatone_envelopes.py
  - Downsample 64Hz              - 28-Band Gammatone
  - EOG Denoise                  - Power Compression (^0.6)
  - Split Trials                 - Lowpass 8Hz
      |                          - Downsample 64Hz
      V                                  |
 S{n}_data_preproc.mat                   V
      |                       gammatone_envelopes.pkl
      |                                  |
      +----------------+-----------------+
                       |
                   (Python)
             train_matchnet_loso.py
                       |
               audio_mapping.json
             (Maps Trial -> wavA, wavB)
                       |
               Z-Score Normalization
                       |
                  MatchNet (B, 8, T) + (B, 28, T)
```


# Block 1 — Input & Preprocessing

## 1. Exact Channel Mapping & Ordering
- **VERIFIED FACT**: The BioSemi64 + EXG setup yields 66 channels in the .mat file. Python zero-based indexing maps 0 to Fp1, 1 to AF7, ..., 65 to EXG2.
- We will maintain **two explicitly separate configurations**:
  - **A) Historical MatchNet reproduction (VERIFIED FACT)**: [13, 46, 43, 23, 50, 0, 52, 14] which equals [C5, FCz, FC6, P9, C6, Fp1, TP8, T7]. This bizarre configuration was used in the previous baseline and MUST be preserved separately for an apples-to-apples baseline comparison.
  - **B) New wearable architecture (VERIFIED FACT)**: [0, 33, 6, 41, 22, 59, 15, 52] which equals the intended bilateral set [Fp1, Fp2, F7, F8, P7, P8, TP7, TP8].

## 2. Sampling Rate
- **VERIFIED FACT**: The raw EEG is downsampled from 512 Hz to **64 Hz** in MATLAB. The input to the Python model is strictly 64 Hz.

## 3. Existing Preprocessing & Filtering
- **VERIFIED FACT**:
  - **MATLAB**: 50 Hz line-noise removal, downsampling (512->64), 0.1 Hz high-pass, EOG Regression (VEOG/HEOG), Common Average Reference (CAR).
  - **Python**: Bandpass filtering (1.0 Hz - 6.0 Hz by default).
- **DECISION**: Freeze Python preprocessing at **1–6 Hz** for the first architecture experiment. Expanding the frequency band will be treated as a separate controlled experiment later.

## 4. Normalization & Leakage Limitations
- **VERIFIED FACT**: Normalization happens in Python. It is **Per-Channel, Per-Trial Z-score Normalization** (similarly per-band, per-trial for audio).
- **LEAKAGE / DEPLOYMENT LIMITATION**: 
  - Statistics are computed independently per trial. No training-subject statistics are transferred to the held-out subject (strict LOSO is maintained).
  - **HOWEVER**, full-trial statistics use information from the entire 50-second trial. Therefore, a 1-second window at t=5s is normalized using information from t=6s to t=50s. This is **not causal/online**.
- **DECISION**: We freeze this per-trial protocol initially for controlled baseline comparison, but explicitly mark causal/online normalization as a future necessary experiment for real-time deployment.

## 5. Existing Windowing
- **VERIFIED FACT**: The 50s trial is sliced into overlapping windows strictly bounded within a trial.
- **ASSUMPTION / CLARIFICATION**: A 1-second window with a 1-second stride yields 50 windows. This is the **proposed architecture-development input** ONLY.
- **DECISION**: Do not claim this 1s/1s stride as the final evaluation protocol. The historical 69.02% benchmark MUST retain its exact original evaluation protocol for the final comparison.

## 6. Label/Metadata Attached
- **VERIFIED FACT**: The dataloader yields (X, Y_A, Y_B). Y_A is strictly the attended audio envelope, and Y_B is strictly the unattended audio envelope. The Siamese/Contrastive training objective intrinsically targets Y_A.

## 7. Open Questions
- None for Block 1. Block 1 is now rigorously defined and frozen for the first set of architecture experiments.


# Block 2 — EEG Temporal Feature Extraction

## 1. The Existing EEGNet Baseline (VERIFIED FROM PROJECT)
- The historical MatchNet baseline uses an EEGNet temporal encoder with a massive initial **k=64** (1.0s) temporal convolution, padded with 32 zeros on each side.
- This is followed by a spatial depthwise convolution and a **k=16** (0.25s) depthwise separable temporal refinement.
- **Limitation of the Baseline:** A fixed 64-sample kernel applies a very wide, fixed-resolution temporal window. While this is highly effective for long offline windows, it limits the network's ability to selectively focus on short, high-frequency morphological features (like onset slopes) independently of the broad 1-second trend. It is also structurally rigid if we intend to support ultra-low-latency short windows (e.g., 0.1s / 6 samples).

## 2. Multi-Window Analysis & Short-Window Feasibility (ARCHITECTURAL CONSTRAINT)
The proposed MSCA architecture must support the following evaluation windows:
- 0.1 s → ~6 samples
- 0.25 s → 16 samples
- 0.5 s → 32 samples
- 1.0 s → 64 samples
- 2.0 s, 5.0 s, 10.0 s → >128 samples

**Behavior of MSCA Kernels (k=3, 7, 15, 31) at short windows:**
- For a 1s window (64 samples), all kernels are fully valid local operators.
- For a 0.25s window (16 samples), `k=31` is physically larger than the entire input sequence. A convolution would either crash or require >50% zero-padding to simply execute, turning it into a massive global padding-artifact generator rather than a local feature extractor.
- For a 0.1s window (6 samples), `k=7, 15, 31` are all mathematically invalid as local convolutions without extreme padding.

**DECISION:** The temporal block cannot be blindly hardcoded to `k=31` without a dynamic handling strategy if it is to support 0.1–0.25s windows. If multi-scale is used, large kernels must be dynamically bypassed, masked, or adaptively weighted when the input window is shorter than the kernel size.

## 3. Evaluation of Candidate Kernels (SIGNAL-PROCESSING INFERENCE)
Even though the signal is bandpass filtered at 1-6Hz (where a full 6Hz cycle is ~11 samples):
- **k=3 (46.9 ms) & k=7 (109.4 ms)**: These kernels do not capture a full oscillatory cycle. However, they are highly effective at learning local slopes, onset/offset trajectories, and phase-shifts.
- **k=15 (234.4 ms)**: Captures ~1.5 cycles of the upper band (6 Hz). Functions as a strong mid-range morphology extractor.
- **k=31 (484.4 ms)**: Captures slow delta/theta oscillations and wide context.
- **Padding Comparison:** The baseline EEGNet uses `k=64` (padding 32). Therefore, the previous argument that `k=31` (padding 15) contains "too much synthetic padding" is rejected. Padding is standard and historically validated in this pipeline.

## 4. Multi-Scale Architecture Candidates (ARCHITECTURAL HYPOTHESIS)
- **A) Single k=15**: Simple, parameter-efficient. **Fails** to capture wide 1-second context or isolated high-resolution slopes.
- **B) Two-scale (k=7 + k=15)**: Better, but still misses the wide context that baseline EEGNet captures with k=64.
- **C) MSCA Multi-Scale (k=3 + k=7 + k=15 + k=31)**: Evaluates the signal simultaneously at 4 distinct resolutions.
  - **Expected Benefit:** Allows the network to learn both sharp local onsets (k=3) and wide slow-wave trends (k=31) simultaneously, potentially overcoming EEGNet's fixed k=64 limitation. This is crucial since historical DTU data shows longer temporal context directly improves AAD (from 57% at 2s to 77% at 30s).
  - **Parameter Risk:** 4 parallel dense convolutions would overfit the 18 subjects.
  - **Solution:** Depthwise/separable multi-scale temporal processing.
  - **Short-Window Risk:** Fails mathematically at 0.1s windows without dynamic masking/bypassing.

## 5. Recommended Temporal Block & Falsifiable Experiment
- **RECOMMENDED DESIGN:** A **Depthwise Multi-Scale Temporal Module (k=3, 7, 15, 31)**.
- **Why:** It directly addresses the fixed-resolution limitation of the EEGNet baseline by extracting multi-resolution morphology, while depthwise operations keep parameters low enough to prevent overfitting.
- **Falsifiable Experiment:** 
  1. Train Baseline MatchNet (EEGNet encoder, k=64 -> k=16).
  2. Train Modified MatchNet (New MSCA Depthwise Temporal encoder, k=[3,7,15,31] concatenated).
  3. Compare LOSO AUROC on the 1-second to 10-second evaluation windows. If MSCA performs worse, the multi-scale hypothesis is rejected, and we revert to single-scale.
