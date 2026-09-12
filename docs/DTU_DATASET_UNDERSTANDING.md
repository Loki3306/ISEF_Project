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
- `train_matchnet_loso.py` loads the subject/trial key (e.g. `S1` -> `trial_0`) to find `wavA` and `wavB` filenames, which are then used to index into the `.pkl` cache (VERIFIED).
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
