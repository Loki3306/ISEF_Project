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
