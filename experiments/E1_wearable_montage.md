# E1 — Wearable Montage Ablation

## Hypothesis
Does the proposed 8-channel wearable ear-EEG montage (`[0, 33, 6, 41, 22, 59, 15, 52]`) preserve the baseline LOSO performance when substituted directly for the historical 8-channel arbitrary montage (`[13, 46, 43, 23, 50, 0, 52, 14]`), assuming all other architectural and processing variables are held perfectly constant?

## Configuration (The Single Variable Change)
- **Parent Commit:** `aeb82b99be61322d2e0055cb51593bb5d376abba` (Frozen E0)
- **Intended Change:** Only the EEG channel selection configuration has been modified in the `train_matchnet_loso.py` script.
- **Unchanged:** Model architecture (MatchNet/EEGNet), Gammatone extraction, dataset, loss, preprocessing, and strict evaluation parameters (10s windows, 18 subjects).

## Pre-Run Integrity Checks
- [x] Parent of `experiment/e1-wearable-montage` is the frozen `e0-canonical-frozen` commit.
- [x] Only the `argparse` default for `--channels` in the training script has changed.
- [x] No modifications were made to preprocessing, dataset splits, boundary tracking, or audio pipelines.

## Execution Command
```bash
python scripts/verify_baseline/training/train_matchnet_loso.py
```

## Results (To be populated)
| Subject Fold | 10s Window Accuracy (%) | Decisions Evaluated | E0 Baseline | Δ (E1 - E0) |
| :--- | :--- | :--- | :--- | :--- |
| S1 | TBD | 300 | 71.33% | TBD |
| S2 | TBD | 300 | 66.00% | TBD |
| S3 | TBD | 300 | 62.33% | TBD |
| S4 | TBD | 300 | 72.00% | TBD |
| S5 | TBD | 300 | 70.00% | TBD |
| S6 | TBD | 300 | 49.67% | TBD |
| S7 | TBD | 300 | 77.67% | TBD |
| S8 | TBD | 300 | 79.67% | TBD |
| S9 | TBD | 300 | 72.33% | TBD |
| S10 | TBD | 300 | 69.33% | TBD |
| S11 | TBD | 300 | 51.67% | TBD |
| S12 | TBD | 300 | 62.33% | TBD |
| S13 | TBD | 300 | 71.67% | TBD |
| S14 | TBD | 300 | 76.33% | TBD |
| S15 | TBD | 300 | 76.33% | TBD |
| S16 | TBD | 300 | 64.67% | TBD |
| S17 | TBD | 300 | 67.33% | TBD |
| S18 | TBD | 300 | 66.67% | TBD |
| **Average** | **TBD** | **5,400** | **68.19%** | **TBD** |

## Analysis
TBD

## Scientific Decision
TBD

## Recommended Next Experiment
TBD
