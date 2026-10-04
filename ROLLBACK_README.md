# Rollback Branch: baseline_deployment_v1

This branch (`rollback/baseline_deployment_v1`) represents a stable, frozen snapshot of the CA-TCN Auditory Attention Decoding project at the end of the first deployment verification phase. If future changes degrade performance, this branch serves as the canonical rollback point.

## 1. Scientific & Engineering Goals Achieved
Validated a **Dual-Mode Operation** for the CA-TCN architecture on the 18-subject DTU dataset:
- **Mode 1 (Zero-Shot):** Universal operation for stable subjects achieving high accuracy (~80-100%) without fine-tuning. Overall LOSO (Leave-One-Subject-Out) cross-validation yielded **84.3% trial accuracy** across the population.
- **Mode 2 (Rapid Adaptation):** A 12-minute rapid personalization pipeline developed for "tough" phenotypes (e.g., S6, S11) using few-shot adaptation (12 calibration trials).

## 2. Architecture Details
- **Core Model:** `CATCNDirectDecoder`
  - Uses `CATCN_AudioEncoder` (causal, 984ms Receptive Field).
  - Uses `CATCN_EEGEncoder` (anticausal, 234ms Receptive Field).
- **Classification Head:** `CrossCorrelationClassificationHead` using multi-lag cross-correlation for final classification.
- **Input Modalities:** 8-channel near-ear EEG (montage: `near_ear_expanded`).

## 3. Key Findings & Adaptation Strategy
Through a 4-regime comparison benchmark, we discovered that **full backbone fine-tuning falls into an anti-correlation trap** for challenging subjects. 
The optimal and most stable method for few-shot adaptation is:
- **Freeze the temporal backbone.**
- **Adapt only spatial layers and Batch Normalization (`spatial_proj` and `bn_spatial`).**
- This reduces the adaptable parameter count to **640 parameters** (vs 38,272 for the full model) and yielded substantial performance improvements (e.g., +18.8 pp gain on subject S11).

## 4. Key Files in this Snapshot
- `scripts/verify_baseline/training/train_catcn_loso.py`: The LOSO benchmark script, which includes real-time stdout line-buffering to allow streaming logs during Kaggle notebook execution.
- `scripts/verify_baseline/training/train_catcn_adaptation.py`: The newly implemented adaptation script that executes the 12-trial calibration pipeline and compares the 4 adaptation regimes (zero-shot, frozen-temporal, full-finetune, etc.).
- `scripts/verify_baseline/models/catcn.py`: Core architecture definitions.

## 5. Usage
To rollback to this exact state, checkout this branch:
```bash
git checkout rollback/baseline_deployment_v1
```
All model weights, scripts, and hyperparameter configurations are tuned for the 4-regime adaptation benchmark and stable real-time stdout streaming in Kaggle.
