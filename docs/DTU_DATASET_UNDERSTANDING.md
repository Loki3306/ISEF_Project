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


# Block 2 — EEG Temporal Feature Extraction (Final Verification)

## 1. The Existing EEGNet Baseline
- **VERIFIED FACT:** The historical baseline uses an EEGNet temporal encoder with an initial temporal convolution of **k=64** (padded with 32 zeros on each side). This is followed by a spatial block, and then a **k=16** depthwise separable temporal refinement (padded with 8 zeros).
- **Temporal Information Captured:** A k=64 kernel does *not* mean it cannot represent short features. Because neural networks learn arbitrary filter weights, a k=64 filter is perfectly capable of learning sharp, high-frequency wavelets (e.g., Gabor-like filters with a narrow envelope) embedded within a larger context. 
- **Limitation to Fix:** While k=64 can theoretically learn sharp wavelets, forcing all temporal filters to span 1 second creates a rigid receptive field that may struggle to decouple ultra-local morphology from broad background oscillations dynamically. 

## 2. Multi-Scale Justification (k=3, 7, 15, 31)
What distinct information could these branches add?
- **k=3 (46.9 ms):** (INFERENCE) Captures immediate local slopes, first-derivatives, and micro-onset boundaries. AAD relevance: Can isolate sharp cortical responses to acoustic transients without integrating surrounding slow-wave noise.
- **k=7 (109.4 ms):** (INFERENCE) Captures wider local curvature and half-wave morphology. AAD relevance: Represents phonemic-level cortical locking.
- **k=15 (234.4 ms):** (INFERENCE) Captures full syllabic-rate oscillations (e.g., 4-6 Hz). AAD relevance: Matches the fundamental cortical envelope-tracking rhythm.
- **k=31 (484.4 ms):** (INFERENCE) Captures slow delta oscillations and wider context. AAD relevance: Provides a phase-reference frame for the faster temporal scales.
- **Is it complementary?** (HYPOTHESIS): Yes, explicit multi-scale processing forces the network to independently represent these distinct scales, whereas a single k=64 kernel might entangle them.

## 3. Parameterization (Dense vs. Depthwise)
Assume the input is processed independently over 1 channel (as in EEGNet's first layer) or uses $C_{in}=8, C_{out}=8$.
- **A. Existing EEGNet Temporal (k=64 + k=16):** 
  - `Conv2d(1, 8, k=64)` = 512 params. 
  - `Conv2d(16, 16, k=16, groups=16)` = 256 params. 
  - Total Temporal = **768 parameters**.
- **B. Single k=15:** 
  - Dense ($C_{in}=8, C_{out}=8$): $8 \times 8 \times 15 = 960$. 
  - Depthwise: $8 \times 15 = 120$.
- **C. k=7 + k=15:** 
  - Dense: $960 + 448 = 1,408$. 
  - Depthwise: $120 + 56 = 176$.
- **D. k=3 + k=7 + k=15:** 
  - Dense: $1408 + 192 = 1,600$. 
  - Depthwise: $176 + 24 = 200$.
- **E. k=3 + k=7 + k=15 + k=31:** 
  - Dense: $1600 + 1984 = 3,584$. 
  - Depthwise: $200 + 248 = 448$.
- **Conclusion (VERIFIED):** A 4-branch depthwise multi-scale module (448 params) is actually *smaller* than the baseline EEGNet temporal pathway (768 params). Parameter explosion is entirely avoided if we use depthwise convolutions.

## 4. Window Compatibility (0.1s to 10s)
At 64 Hz: 0.1s (6 samples), 0.25s (16 samples), 0.5s (32 samples), 1s (64 samples), 2s (128 samples), 5s (320 samples), 10s (640 samples).
- **Short-Window Failure:** A fixed `k=31` convolution mathematically fails at 0.1s and 0.25s without absurd padding. A fixed `k=15` fails at 0.1s.
- **Architectural Options for Variable Windows:**
  - *A) Fixed multi-scale:* Forces extreme padding at short windows (Artifact-prone).
  - *B) Conditional/Bypass:* Explicitly masks or drops the k=15 and k=31 branches when the input length is too short.
  - *C) Scale-adaptive/dilated:* Uses small kernels with dynamic dilation factors.
  - *D) Separate architectures:* Train a different model per window length.
- **Decision (HYPOTHESIS):** Option B (Conditional/Bypass) is the most robust and biologically interpretable. If the signal is too short to estimate a 0.5-second trend, the network should dynamically rely solely on the surviving k=3 and k=7 branches, mimicking how humans perform ultra-fast heuristic auditory processing vs. delayed integrated processing.

## 5. Architectural Alignment (Later Blocks)
- **Output:** The temporal block will output a tensor of shape `[Batch, Scales*Filters, Time]`. 
- **Compatibility:** This tensor preserves the exact temporal resolution (no downsampling). It seamlessly feeds the Frequency block (which can perform spectral attention across the concatenated scales) and the Spatial block (which can mix channels across all extracted temporal scales). 

## 6. Experiment Design (Ablation Sequence)
To prove that multi-scale processing actually drives the AAD improvement, we must run this exact ablation (keeping spatial/attention/loss strictly identical):
- **E0 (Baseline):** Replace MSCA temporal block with EEGNet temporal block (k=64 -> k=16).
- **E1 (1-Scale):** Single depthwise k=15.
- **E2 (2-Scale):** Depthwise k=[7, 15].
- **E3 (3-Scale):** Depthwise k=[3, 7, 15].
- **E4 (4-Scale):** Depthwise k=[3, 7, 15, 31].
*Justification requirement: E(N) must statistically outperform E(N-1) across the 1-10s windows to justify the addition of the larger scale.*

## 7. Final Decision Summary
- **VERIFIED FACTS:** The existing baseline uses large kernels (k=64, k=16) padded heavily. A depthwise 4-branch multi-scale block uses *fewer* temporal parameters (448) than the baseline (768).
- **INFERENCES:** Small kernels (<10) capture local slopes/derivatives. Large kernels (>30) capture broad phase trends.
- **HYPOTHESES:** Explicitly separating these scales forces the network to learn decoupled features (sharp vs smooth), which is better for AAD than a single k=64 kernel that might entangle them.
- **RECOMMENDED TEMPORAL DESIGN:** A **Depthwise Multi-Scale Temporal Module** (`k=3, 7, 15, 31`) with **Dynamic Conditional Bypassing** for ultra-short windows (0.1-0.25s).
- **WHY IT SHOULD IMPROVE AAD:** It solves the rigidity of the baseline by preserving sharp transients while simultaneously capturing long-range context, supporting both low-latency extraction and high-accuracy long-window aggregation.
- **WHAT EXPERIMENT WOULD DISPROVE IT:** The E0-E4 ablation sequence. If E4 <= E0, the multi-scale temporal separation hypothesis is false.


# Block 3 — EEG Frequency Representation

## 1. Current Baseline Frequency Content (VERIFIED FACT)
- **Raw EEG frequency content:** Originally recorded at 512 Hz, capable of capturing up to 256 Hz.
- **MATLAB preprocessing:** High-pass filtered at 0.1 Hz and downsampled to 64 Hz (retains 0.1–32 Hz).
- **Python preprocessing:** Bandpass filtered rigidly to **1–6 Hz**.
- **EEGNet Temporal Convolutions:** The unconstrained k=64 and k=16 temporal convolutions act as data-driven FIR filters. They implicitly learn frequency-selective representations (phase and amplitude) optimized for the task within whatever frequencies remain in the input.

## 2. Core Question (SIGNAL-PROCESSING INFERENCE)
**Does the new architecture need an explicit frequency representation?**
An unconstrained Temporal Convolution (Block 2) is mathematically a finite impulse response (FIR) filter. It already learns to isolate specific frequencies. An explicit frequency representation is only required if the temporal convolutions struggle to learn clean frequency boundaries from the limited 18-subject dataset, in which case parameterized filters (like SincConv) can heavily regularize the learning process.

## 3. Candidate Evaluation
- **A) No explicit frequency branch (F0)**
  - *Mechanism:* Relies entirely on Block 2 to learn spectral-temporal features.
  - *Parameters:* 0 additional. *Suitability for LOSO:* Very high.
  - *Limitation:* Temporal filters might become noisy/overfit without explicit frequency regularization.
- **B) Fixed sub-band representation (F1)**
  - *Mechanism:* Pre-computes filter-banks (e.g., Delta 1-4Hz, Theta 4-6Hz).
  - *Parameters:* 0 learned. *Limitation:* Boundaries are handcrafted and rigid.
- **C) Learnable SincConv (F2)**
  - *Mechanism:* Learns parameterized bandpass filters (low cut, high cut).
  - *Parameters:* Extremely low (~2 parameters per filter).
  - *Expected Benefit:* Enforces clean biological bandpass shapes while adapting to optimal AAD bands.
- **D) STFT/Wavelet**
  - *Mechanism:* Time-frequency maps. *Limitation:* High memory footprint, massive overkill for a 1-6 Hz bandpassed signal.

## 4. Critical Sinc Limitation (SIGNAL-PROCESSING INFERENCE)
A SincConv layer operates linearly. It **CANNOT** recover frequencies removed by the upstream 1–6 Hz filter (e.g., Alpha 8-12Hz, Beta 15-30Hz).
- **A) SincConv on 1–6 Hz:** Can only learn to subdivide the 1-6 Hz band (e.g., separating Delta 1-4Hz from Theta 4-6Hz).
- **B) SincConv on broader-band:** Would allow the network to discover AAD-relevant frequencies across the entire spectrum. *(FUTURE EXPERIMENT)*

## 5. Frequency Resolution & Multi-Window Constraints (SIGNAL-PROCESSING INFERENCE)
Can useful sub-bands be learned within 1-6 Hz? The frequency resolution limit is $\Delta f \approx 1/T$.
- **1.0s window:** $\Delta f \approx 1$ Hz. SincConv could reasonably distinguish a 1-3 Hz band from a 4-6 Hz band.
- **0.25s window:** $\Delta f \approx 4$ Hz. It is **physically impossible** to resolve sub-bands inside a 1-6 Hz signal.
- **0.1s window:** $\Delta f \approx 10$ Hz.
- **Conclusion:** SincConv on a 1-6 Hz signal only functions as a frequency discriminator for windows $\geq$ 1 second. For ultra-short windows, it provides zero frequency resolution.

## 6. Architectural Complementarity (ARCHITECTURAL HYPOTHESIS)
If the frequency branch simply applies SincConv and retains phase (without envelope extraction or spectral pooling), it is mathematically just another 1D convolution. It would essentially duplicate the function of Block 2, merely with a different initialization and regularization shape. To be truly complementary, a frequency branch must extract spectral *power* (e.g., via Hilbert envelopes or magnitude pooling), while Block 2 extracts phase-locked *morphology*.

## 7. Experiment Design (Ablation Sequence)
To test whether frequency representation is necessary, keep everything (including Block 2) identical:
- **F0:** Block 2 temporal representation only.
- **F1:** Temporal + fixed frequency representation (Delta/Theta banks).
- **F2:** Temporal + learned Sinc frequency representation (on 1-6 Hz).
- **F3:** Broader preprocessing (e.g., 0.1-32 Hz) + learned Sinc frequency representation. *(FUTURE PREPROCESSING EXPERIMENT)*

## 8. Final Recommendation & Falsification
- **RECOMMENDATION (ARCHITECTURAL HYPOTHESIS):** Use **F0 (No explicit frequency branch)** as the primary baseline for the first architecture, treating **F2 (SincConv on 1-6 Hz)** strictly as an ablation.
- **Why:** The upstream 1-6 Hz preprocessing severely limits the spectrum. The uncertainty principle dictates that frequency discrimination inside a tight 5-Hz band is physically impossible for the 0.1s - 0.5s windows we intend to support. Therefore, an explicit frequency branch risks being entirely mathematically redundant to Block 2.
- **Falsification Experiment:** Train F0 vs F2. If F2 does not significantly outperform F0 at 1s-10s windows, we conclude that an explicit frequency branch on 1-6 Hz data is redundant. We would then reserve SincConv exclusively for the **F3 (Broader Preprocessing)** future experiment.
