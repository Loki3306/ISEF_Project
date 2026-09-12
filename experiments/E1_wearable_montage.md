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
| S1 | 65.67% | 300 | 71.33% | -5.66% |
| S2 | 59.00% | 300 | 66.00% | -7.00% |
| S3 | 50.67% | 300 | 62.33% | -11.66% |
| S4 | 63.00% | 300 | 72.00% | -9.00% |
| S5 | 56.00% | 300 | 70.00% | -14.00% |
| S6 | 51.00% | 300 | 49.67% | +1.33% |
| S7 | 53.00% | 300 | 77.67% | -24.67% |
| S8 | 57.33% | 300 | 79.67% | -22.34% |
| S9 | 56.67% | 300 | 72.33% | -15.66% |
| S10 | 62.67% | 300 | 69.33% | -6.66% |
| S11 | 54.67% | 300 | 51.67% | +3.00% |
| S12 | 55.00% | 300 | 62.33% | -7.33% |
| S13 | 58.67% | 300 | 71.67% | -13.00% |
| S14 | 67.33% | 300 | 76.33% | -9.00% |
| S15 | 69.67% | 300 | 76.33% | -6.66% |
| S16 | 62.00% | 300 | 64.67% | -2.67% |
| S17 | 58.33% | 300 | 67.33% | -9.00% |
| S18 | 66.33% | 300 | 66.67% | -0.34% |
| **Average** | **59.28%** | **5,400** | **68.19%** | **-8.91%** |

## Analysis
The result is a catastrophic collapse. The average LOSO accuracy fell from 68.19% to 59.28% (a -8.91% absolute degradation).
- **Concentrated Degradation:** The degradation is most severe in the "easiest" subjects who previously drove the high baseline. For example, Subject 7 fell from 77.67% to 53.00% (-24.67%), and Subject 8 fell from 79.67% to 57.33% (-22.34%). 
- **The Physical Constraint:** The wearable montage systematically removes the fronto-central (`FCz`, `FC6`) and central (`C5`, `C6`) electrodes, confining sensors strictly to the periphery (forehead and around the ear). Auditory evoked potentials project maximally toward the top of the head (vertex). By amputating these channels, we deprived the spatial filter of the highest-SNR signals it relied on.
- **Artifact Vulnerability:** The peripheral electrodes in the wearable montage (`Fp1`, `Fp2`, `F7`, `F8`) are highly susceptible to ocular and muscular artifacts, which may further confuse the shallow linear spatial filters in the EEGNet encoder.

## Scientific Decision
**B. WEARABLE MONTAGE DEGRADES PERFORMANCE**
The channel change causes a severe, structural performance loss. The linear spatial filter of `EEGNet` cannot compensate for the missing fronto-central dipoles when restricted to peripheral measurements.

## Recommended Next Experiment
We must now determine if this performance loss is an absolute physical limitation of the ear-EEG form factor, or if it is an architectural limitation of the EEGNet encoder.
**Next Step (Block 4):** Design and evaluate a more powerful spatial architecture (e.g., self-attention, non-linear spatial mixing, or multi-scale convolutions) capable of dynamically recovering auditory correlations from the noisier, peripheral wearable electrodes without relying on strict, static linear spatial maps.
