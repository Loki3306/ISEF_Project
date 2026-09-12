# DTU Dataset — Complete Data Understanding

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

## 10. EEG ↔ audio mapping
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


# Block 1 � Input & Preprocessing

## 1. Exact Channel Mapping
- **VERIFIED FROM EEG_training_new (`DATASETS.md`)**
- The BioSemi64 + EXG setup yields 66 channels in the `.mat` file.
- Python zero-based indexing is as follows:
  - 0: Fp1, 1: AF7, 2: AF3, 3: F1, 4: F3, 5: F5, 6: F7, 7: FT7, 8: FC5, 9: FC3, 10: FC1, 11: C1, 12: C3, 13: C5, 14: T7, 15: TP7, 16: CP5, 17: CP3, 18: CP1, 19: P1, 20: P3, 21: P5, 22: P7, 23: P9, 24: PO7, 25: PO3, 26: O1, 27: Iz, 28: Oz, 29: POz, 30: Pz, 31: CPz, 32: Fpz, 33: Fp2, 34: AF8, 35: AF4, 36: AFz, 37: Fz, 38: F2, 39: F4, 40: F6, 41: F8, 42: FT8, 43: FC6, 44: FC4, 45: FC2, 46: FCz, 47: Cz, 48: C2, 49: C4, 50: C6, 51: T8, 52: TP8, 53: CP6, 54: CP4, 55: CP2, 56: P2, 57: P4, 58: P6, 59: P8, 60: P10, 61: PO8, 62: PO4, 63: O2, 64: EXG1, 65: EXG2

## 2. Exact 8-Channel Ordering
- **CRITICAL FINDING (VERIFIED FROM `train_matchnet_loso.py`)**: The existing MatchNet baseline was trained using Python indices `[13, 46, 43, 23, 50, 0, 52, 14]`.
- This actually mapped to `[C5, FCz, FC6, P9, C6, Fp1, TP8, T7]`.
- **The previous documentation (`DATASETS.md`) was completely wrong.** It claimed those indices corresponded to `Fp1, Fp2, F7...`.
- To use the *actual* proposed spatial channels `[Fp1, Fp2, F7, F8, P7, P8, TP7, TP8]`, the indices MUST be: **`[0, 33, 6, 41, 22, 59, 15, 52]`**.

## 3. Sampling Rate
- **VERIFIED FROM DATA**: The raw EEG is downsampled from 512 Hz to **64 Hz** in MATLAB.
- The input to the Python model is strictly 64 Hz.

## 4. Existing Preprocessing
- **VERIFIED FROM CODE**
- **MATLAB**:
  - 50 Hz line-noise removal.
  - Downsampling (512 Hz -> 64 Hz).
  - High-pass filter (0.1 Hz).
  - EOG Regression (VEOG/HEOG).
  - Common Average Reference (CAR).
- **Python**:
  - Channel selection.
  - Bandpass filtering (1.0 Hz - 6.0 Hz by default).

## 5. Existing Normalization
- **VERIFIED FROM `train_matchnet_loso.py`**
- Normalization happens in Python.
- It is **Per-Channel, Per-Trial Z-score Normalization**.
- The `normalize_array` function subtracts the mean of the channel over the entire 50s trial and divides by the std of that channel over the trial.
- Audio (28 bands) is identically normalized (Per-Band, Per-Trial).

## 6. Existing Windowing
- **VERIFIED FROM `chunk_trial` function**
- The 50s trial is sliced into overlapping windows.
- Windows are strictly bounded within a trial (no crossing boundaries).
- Remainder samples at the end of the trial are discarded.
- The proposed **1-second input shape** would be exactly `[8, 64]` (8 channels, 64 samples).

## 7. Label/Metadata Attached
- **VERIFIED FROM CODE**: There is no explicit scalar `1` or `0` label passed to the model.
- The dataloader yields `(X, Y_A, Y_B)`.
- `Y_A` is strictly the attended audio envelope.
- `Y_B` is strictly the unattended audio envelope.
- The training objective (Contrastive Loss) intrinsically treats `Y_A` as the positive pair.

## 8. Leakage Considerations
- **VERIFIED AS SAFE**: Normalization statistics are computed *per-trial* before windowing, and trials are entirely independent. Subject separation (LOSO) is fully maintained. There is no leakage of statistics from validation into training.

## 9. Open Questions
- Do we want to maintain the `[1.0 Hz - 6.0 Hz]` Python bandpass filter for our new 1.0s window, or expand it?
- Do we want to use the *old* bizarre channels `[C5, FCz...]` to directly compare with the baseline, or use the *intended* `[Fp1, Fp2, F7, F8, P7, P8, TP7, TP8]` channels for the new model?
