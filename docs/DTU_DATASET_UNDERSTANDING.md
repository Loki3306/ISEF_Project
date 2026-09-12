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

## 1. Temporal Information in 1-Second (64-Sample) Window
- **VERIFIED FROM PROJECT/DATA**: The input is 64 samples (1.0 seconds) at 64 Hz, strictly bandpass filtered at 1–6 Hz.
- **INFERENCE (Physics/Signal Processing)**: A 1 Hz wave has a period of 64 samples (1000 ms). A 6 Hz wave has a period of ~10.7 samples (~167 ms). Because the signal is bounded at 1-6 Hz, it contains NO high-frequency transients. The signal is highly smoothed. Any kernel smaller than 10 samples cannot capture a full oscillation of even the highest frequency in the band.

## 2. Evaluation of Candidate Kernels (at 64 Hz)
- **k = 3 (46.9 ms)**: Captures local gradients/slopes. Sub-cycle for 1-6 Hz. (INFERENCE)
- **k = 7 (109.4 ms)**: Sub-cycle for 1-6 Hz. Captures half-waves. (INFERENCE)
- **k = 15 (234.4 ms)**: Captures at least one full cycle of the 6 Hz upper bound. Good mid-range feature extractor. (INFERENCE)
- **k = 31 (484.4 ms)**: Captures nearly 50% of the entire 1-second window. (VERIFIED FROM PROJECT/DATA)
  - **Failure Mode**: If padded (e.g. `padding="same"`), it requires 15 zeros on each side, meaning ~47% of the edges are synthetic padding, causing massive edge artifacts. If unpadded, the sequence shrinks from 64 to 34, destroying temporal resolution for later fusion blocks. (INFERENCE)

## 3. Evaluation of the Proposed MSCA Branches (k=3, 7, 15, 31)
- **ARCHITECTURAL HYPOTHESIS / VERDICT**: The MSCA proposal of four parallel branches is drastically over-parameterized for a 1-6 Hz, 64-sample signal. The k=31 branch is too large and will cause severe edge artifacts or resolution loss. Furthermore, 4 parallel dense convolutions will likely overfit the limited 18-subject DTU dataset. We reject the 4-branch k=31 proposal for Block 2.

## 4. Architectural Candidates
- **A) Single dense temporal convolution**: Parameter heavy ($C_{in} \times C_{out} \times K$).
- **B) Depthwise temporal convolution**: Applies filters per-channel independently. Massive parameter reduction ($C \times K$). Highly appropriate for small datasets (used effectively in EEGNet). (INFERENCE)
- **C) Two-scale depthwise convolution (e.g., k=7, k=15)**: Captures sub-cycle gradients and full-cycle 6Hz oscillations without catastrophic edge effects.

## 5. Temporal Downsampling
- **DECISION**: Do not downsample/stride in the temporal encoder. We only have 64 samples. We need to preserve temporal resolution for alignment with the audio envelope later. (ARCHITECTURAL HYPOTHESIS)

## 6. Recommended Temporal Block
- **RECOMMENDATION**: A single Depthwise Temporal Convolution (k=15) OR a Two-Scale Depthwise Convolution (k=7, k=15).
- **Why**: Depthwise convolution strictly controls parameter count to prevent overfitting the 18 subjects. k=15 (234ms) is perfectly sized to capture the 6 Hz cycles (167ms) without crossing the 50% window threshold that causes catastrophic padding artifacts. It is the absolute smallest, most interpretable mechanism to extract temporal morphology before moving to spatial blocks.
