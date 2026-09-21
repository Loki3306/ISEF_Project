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
| True labels | TBD | TBD | TBD | TBD |
| Shuffled labels | TBD | TBD | TBD | TBD |
| Zero EEG | TBD | TBD | TBD | TBD |
| Shuffled audio | TBD | TBD | TBD | TBD |

## Interpretation

TBD

## Decision

TBD

## Next Experiment

TBD
