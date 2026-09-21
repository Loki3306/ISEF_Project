# EXP-001: File-Disjoint MSCA Baseline

## Objective

Establish a leakage-free baseline for the current MSCA
auditory attention decoding architecture.

## Hypothesis

If MSCA is genuinely decoding auditory attention rather than
memorizing audio identities, performance should remain above
chance when all test audio files are completely absent from
training.

## Architecture

### EEG
- 8 channels
- 64 Hz
- Current preprocessing
- MSCA encoder
- 64-D embedding

### Audio
- 28-band Gammatone
- compression gamma = 0.6
- SincConv
- 64-D embedding

### Objective
- InfoNCE

### Evaluation
- LOSO
- 10-second windows
- File-disjoint audio split

## Leakage Constraint

train_audio_files ∩ test_audio_files = ∅

## Controls

- True labels
- Shuffled labels
- Zero EEG
- Shuffled audio

## Results

| Condition | Accuracy | AUROC | AUPRC | Bal. Acc. |
|---|---:|---:|---:|---:|
| True labels | 63.67% | - | - | - |
| Shuffled labels | 55.00% | - | - | - |
| Zero EEG | 1.33% | - | - | - |
| Shuffled Audio Time | 50.00% | - | - | - |
| Shuffled EEG Time | **88.67%** | - | - | - |
| Permute Spatial | 54.33% | - | - | - |

## Interpretation

**The Shuffled EEG Paradox (88.67% Accuracy):**
The baseline achieves 63.67% with normal data, but when the EEG time axis is completely randomized (destroying all physiological signal), accuracy skyrockets to 88.67%. 

This proves two critical facts:
1. **Acoustic Asymmetry in the Dataset:** In this fold of the DTU dataset, `wavA` (Attended) and `wavB` (Unattended) are not perfectly symmetric. There is a systematic acoustic difference between them (e.g., volume, pitch, speaker gender).
2. **The Model is Ignoring the EEG:** The model has learned a constant "default" bias vector in the EEG latent space that simply points toward the generic acoustic features of `wavA`. 
   - When fed **Shuffled EEG** (white noise), the EEG variance collapses, the model outputs its pure default bias, and achieves 88% by just guessing based on the audio asymmetry.
   - When fed **True EEG**, the physiological fluctuations act as *noise* that perturbs this default bias, degrading the accuracy down to 63%.
   - **Shuffled Labels** drops to 55%, proving the model relies on the consistent Attended/Unattended ordering.
   - **Shuffled Audio Time** drops to 50%, proving the model is relying on the temporal acoustic structure of the audio to make its guess.

## Decision

The current MSCA architecture is **not extracting genuine AAD signal**. It has collapsed into an audio-only classifier exploiting dataset asymmetry. We cannot move to DANN or NTDF yet, because the base representation is fundamentally broken.

## Next Experiment

**EXP-002: The WavLM Audio Ablation.**
Since the current Gammatone + SincConv encoder is failing to provide a representation that cleanly aligns with EEG, we must test if a higher-level semantic representation (WavLM) allows the EEG encoder to escape this local minimum.
