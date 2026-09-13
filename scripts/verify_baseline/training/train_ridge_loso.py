"""
Ridge Stimulus Reconstruction — LOSO and Within-Subject AAD Baseline
====================================================================
Reuses the existing ridge_aad.py functions exactly as written.
No new architecture. No neural model. No audio leakage.

Decision rule (per window):
  1. Fit linear decoder: W = Ridge(EEG_lagged, envelope_attended)
  2. Reconstruct:        r = EEG_lagged @ W
  3. Compare:           corr(r, env_A) vs corr(r, env_B)
  4. Choose higher correlation as attended.

Modes:
  --mode loso         train on N-1 subjects, test on 1 (18 folds)
  --mode within       8-fold CV within each subject (60 trials)

Window sizes tested: 1 2 5 10 20 40 seconds

Usage (Kaggle):
    !cd ISEF_Project && python scripts/verify_baseline/training/train_ridge_loso.py --mode loso
    !cd ISEF_Project && python scripts/verify_baseline/training/train_ridge_loso.py --mode within
    !cd ISEF_Project && python scripts/verify_baseline/training/train_ridge_loso.py --mode both
"""

from __future__ import annotations

import argparse
import sys
import os
import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import (
    subject_files,
    load_subject_examples,
    iter_leave_one_subject_out,
    TrialExample,
    speech_envelope,
    lagged_eeg_matrix,
    fit_ridge,
    predict_envelope,
    feature_statistics,
    standardize_features,
)

# ─── Configuration ────────────────────────────────────────────────────────────

# DTU dataset: 66 channels = 64 scalp EEG + 2 EXG (mastoid) electrodes.
# We exclude EXG1 (index 64) and EXG2 (index 65) for now.
SCALP_CHANNELS = list(range(64))

FS = 64                  # Sampling rate (already downsampled in preprocessed MAT files)
LAG_MS = 250             # Maximum lag for the EEG→audio temporal filter (ms)
LAG_STEP_MS = 16         # Lag step (one sample at 64 Hz)
RIDGE_LAMBDA = 1e4       # Ridge regularisation (standard for EEG stimulus reconstruction)
LOWPASS_HZ = 8.0         # Lowpass filter for speech envelope (tracking band)
COMPRESSION = 0.3        # Power-law compression for speech envelope

# DTU: label=1 means subject attended to wavA; label=2 means subject attended to wavB.
LABEL_MAPPING = {1: "A", 2: "B"}

WINDOW_SIZES_S = [1, 2, 5, 10, 20, 40]    # Decision window sizes in seconds
WITHIN_SUBJECT_FOLDS = 8                   # For within-subject cross-validation

OUTPUT_DIR = Path("ridge_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Helper functions ──────────────────────────────────────────────────────────

def filter_eeg_channels(eeg: np.ndarray, channels: list[int]) -> np.ndarray:
    """Select scalp channels. eeg is [n_samples, n_channels]."""
    return eeg[:, channels]


def get_both_envelopes(example: TrialExample) -> tuple[np.ndarray, np.ndarray]:
    """Return (env_A, env_B) speech envelopes from wavA and wavB in the MAT file.
    This uses the raw waveform stored in the MAT file, NOT the gammatone pkl.
    env_A and env_B are the envelopes of the two competing audio streams.
    """
    env_a = speech_envelope(
        example.wav_a,
        lowpass_hz=LOWPASS_HZ,
        compression=COMPRESSION,
        fs=FS,
        normalize=True,
    )
    env_b = speech_envelope(
        example.wav_b,
        lowpass_hz=LOWPASS_HZ,
        compression=COMPRESSION,
        fs=FS,
        normalize=True,
    )
    return env_a, env_b


def attended_envelope(example: TrialExample) -> np.ndarray:
    """Return the envelope of the attended audio stream (the training target)."""
    env_a, env_b = get_both_envelopes(example)
    if example.label == 1:
        return env_a   # attending to speaker A
    elif example.label == 2:
        return env_b   # attending to speaker B
    else:
        raise ValueError(f"Unexpected label: {example.label}")


def eval_windows(
    eeg: np.ndarray,
    env_a: np.ndarray,
    env_b: np.ndarray,
    label: int,
    weights: np.ndarray,
    window_sec: int,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[int, int]:
    """
    Slide non-overlapping decision windows over one trial.
    Returns (n_correct, n_windows).
    """
    win_samples = window_sec * FS
    n = eeg.shape[0]
    n_correct = 0
    n_windows = 0

    start = 0
    while start + win_samples <= n:
        end = start + win_samples

        eeg_w  = eeg[start:end, :]
        env_a_w = env_a[start:end]
        env_b_w = env_b[start:end]

        # Reconstruct envelope from EEG
        pred = predict_envelope(
            eeg_w,
            weights,
            lag_ms=LAG_MS,
            lag_step_ms=LAG_STEP_MS,
            fs=FS,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        # Trim to common length (lagged_eeg_matrix may shorten slightly)
        min_len = min(len(pred), len(env_a_w), len(env_b_w))
        pred    = pred[:min_len]
        env_a_w = env_a_w[:min_len]
        env_b_w = env_b_w[:min_len]

        corr_a, _ = pearsonr(pred, env_a_w)
        corr_b, _ = pearsonr(pred, env_b_w)

        # Decision: the higher-correlation stream is predicted as attended
        if label == 1:
            # Ground truth: attending to A
            if corr_a > corr_b:
                n_correct += 1
        elif label == 2:
            # Ground truth: attending to B
            if corr_b > corr_a:
                n_correct += 1

        n_windows += 1
        start += win_samples

    return n_correct, n_windows


# ─── LOSO mode ────────────────────────────────────────────────────────────────

def run_loso() -> list[dict]:
    print(f"\n{'='*70}")
    print("RIDGE AAD — LEAVE-ONE-SUBJECT-OUT (LOSO)")
    print(f"{'='*70}")
    print(f"  Channels:    {len(SCALP_CHANNELS)} scalp (EXG excluded)")
    print(f"  Lags:        up to {LAG_MS}ms (step {LAG_STEP_MS}ms)")
    print(f"  Lambda:      {RIDGE_LAMBDA}")
    print(f"  Envelope:    Hilbert + LP {LOWPASS_HZ}Hz + compression {COMPRESSION}")
    print(f"  Windows:     {WINDOW_SIZES_S}s")
    print()

    paths = subject_files()
    all_examples = {str(p): load_subject_examples(p) for p in paths}

    rows = []
    # Per-window-size accumulators across all held-out subjects
    totals = {w: {"n_correct": 0, "n_total": 0} for w in WINDOW_SIZES_S}

    for held_out_path, train_paths in iter_leave_one_subject_out(paths):
        held_out_id = held_out_path.stem
        train_exs: list[TrialExample] = []
        for p in train_paths:
            train_exs.extend(all_examples[str(p)])

        test_exs = all_examples[str(held_out_path)]

        print(f"  Held-out: {held_out_id} | Train subjects: {len(train_paths)} | Train trials: {len(train_exs)}")

        # ── Step 1: Compute feature statistics on training set ──────────────
        train_exs_chan = [
            TrialExample(
                subject=ex.subject,
                trial_index=ex.trial_index,
                eeg=filter_eeg_channels(ex.eeg, SCALP_CHANNELS),
                wav_a=ex.wav_a,
                wav_b=ex.wav_b,
                label=ex.label,
            )
            for ex in train_exs
        ]

        feat_mean, feat_std = feature_statistics(
            train_exs_chan,
            lag_ms=LAG_MS,
            lag_step_ms=LAG_STEP_MS,
            fs=FS,
        )

        # ── Step 2: Fit cross-subject Ridge decoder ─────────────────────────
        weights = fit_ridge(
            train_exs_chan,
            LABEL_MAPPING,
            lag_ms=LAG_MS,
            lag_step_ms=LAG_STEP_MS,
            fs=FS,
            ridge_lambda=RIDGE_LAMBDA,
            feature_mean=feat_mean,
            feature_std=feat_std,
        )

        # ── Step 3: Evaluate on held-out subject ────────────────────────────
        for ws in WINDOW_SIZES_S:
            n_correct = 0
            n_total = 0
            for ex in test_exs:
                eeg_ch = filter_eeg_channels(ex.eeg, SCALP_CHANNELS)
                env_a, env_b = get_both_envelopes(ex)
                # Trim to min length
                min_len = min(eeg_ch.shape[0], len(env_a), len(env_b))
                eeg_ch = eeg_ch[:min_len]
                env_a  = env_a[:min_len]
                env_b  = env_b[:min_len]

                nc, nt = eval_windows(
                    eeg_ch, env_a, env_b, ex.label, weights, ws,
                    feature_mean=feat_mean, feature_std=feat_std,
                )
                n_correct += nc
                n_total   += nt

            acc = 100.0 * n_correct / max(n_total, 1)
            totals[ws]["n_correct"] += n_correct
            totals[ws]["n_total"]   += n_total
            rows.append({
                "mode": "LOSO",
                "subject": held_out_id,
                "window_s": ws,
                "n_correct": n_correct,
                "n_total": n_total,
                "accuracy_pct": round(acc, 2),
            })
            print(f"    [{ws:>2}s] Acc: {acc:.1f}%  ({n_correct}/{n_total})")

        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LOSO SUMMARY")
    print(f"{'='*70}")
    print(f"{'Window':>8} | {'Accuracy':>10} | {'Windows':>8}")
    print("-" * 35)
    for ws in WINDOW_SIZES_S:
        nc = totals[ws]["n_correct"]
        nt = totals[ws]["n_total"]
        acc = 100.0 * nc / max(nt, 1)
        print(f"{ws:>6}s   | {acc:>9.1f}%  | {nt:>8}")
    print()

    return rows


# ─── Within-Subject mode ──────────────────────────────────────────────────────

def run_within() -> list[dict]:
    print(f"\n{'='*70}")
    print(f"RIDGE AAD — WITHIN-SUBJECT ({WITHIN_SUBJECT_FOLDS}-FOLD CV PER SUBJECT)")
    print(f"{'='*70}")

    paths = subject_files()
    all_examples = {str(p): load_subject_examples(p) for p in paths}

    rows = []
    totals = {w: {"n_correct": 0, "n_total": 0} for w in WINDOW_SIZES_S}

    for path in paths:
        subject_id = path.stem
        exs = all_examples[str(path)]
        n = len(exs)
        fold_size = n // WITHIN_SUBJECT_FOLDS

        rng = np.random.RandomState(42)
        indices = rng.permutation(n)

        for ws in WINDOW_SIZES_S:
            n_correct_total = 0
            n_total_total   = 0

            for fold in range(WITHIN_SUBJECT_FOLDS):
                test_idx  = indices[fold * fold_size : (fold + 1) * fold_size]
                train_idx = np.concatenate([
                    indices[:fold * fold_size],
                    indices[(fold + 1) * fold_size:],
                ])

                train_exs = [exs[i] for i in train_idx]
                test_exs  = [exs[i] for i in test_idx]

                train_chan = [
                    TrialExample(
                        subject=ex.subject,
                        trial_index=ex.trial_index,
                        eeg=filter_eeg_channels(ex.eeg, SCALP_CHANNELS),
                        wav_a=ex.wav_a,
                        wav_b=ex.wav_b,
                        label=ex.label,
                    )
                    for ex in train_exs
                ]

                feat_mean, feat_std = feature_statistics(
                    train_chan,
                    lag_ms=LAG_MS,
                    lag_step_ms=LAG_STEP_MS,
                    fs=FS,
                )
                weights = fit_ridge(
                    train_chan,
                    LABEL_MAPPING,
                    lag_ms=LAG_MS,
                    lag_step_ms=LAG_STEP_MS,
                    fs=FS,
                    ridge_lambda=RIDGE_LAMBDA,
                    feature_mean=feat_mean,
                    feature_std=feat_std,
                )

                for ex in test_exs:
                    eeg_ch = filter_eeg_channels(ex.eeg, SCALP_CHANNELS)
                    env_a, env_b = get_both_envelopes(ex)
                    min_len = min(eeg_ch.shape[0], len(env_a), len(env_b))
                    eeg_ch = eeg_ch[:min_len]
                    env_a  = env_a[:min_len]
                    env_b  = env_b[:min_len]

                    nc, nt = eval_windows(
                        eeg_ch, env_a, env_b, ex.label, weights, ws,
                        feature_mean=feat_mean, feature_std=feat_std,
                    )
                    n_correct_total += nc
                    n_total_total   += nt

            acc = 100.0 * n_correct_total / max(n_total_total, 1)
            totals[ws]["n_correct"] += n_correct_total
            totals[ws]["n_total"]   += n_total_total
            rows.append({
                "mode": "within",
                "subject": subject_id,
                "window_s": ws,
                "n_correct": n_correct_total,
                "n_total": n_total_total,
                "accuracy_pct": round(acc, 2),
            })

        # Print per-subject across all windows
        subj_rows = [r for r in rows if r["subject"] == subject_id and r["mode"] == "within"]
        accs = "  ".join(f"{r['window_s']}s={r['accuracy_pct']:.1f}%" for r in subj_rows)
        print(f"  {subject_id}: {accs}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"WITHIN-SUBJECT SUMMARY (mean across all {len(paths)} subjects)")
    print(f"{'='*70}")
    print(f"{'Window':>8} | {'Accuracy':>10} | {'Windows':>8}")
    print("-" * 35)
    for ws in WINDOW_SIZES_S:
        nc = totals[ws]["n_correct"]
        nt = totals[ws]["n_total"]
        acc = 100.0 * nc / max(nt, 1)
        print(f"{ws:>6}s   | {acc:>9.1f}%  | {nt:>8}")
    print()

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ridge AAD baseline — LOSO and within-subject")
    parser.add_argument("--mode", choices=["loso", "within", "both"], default="both")
    args = parser.parse_args()

    all_rows: list[dict] = []

    if args.mode in ("loso", "both"):
        rows = run_loso()
        all_rows.extend(rows)

    if args.mode in ("within", "both"):
        rows = run_within()
        all_rows.extend(rows)

    if all_rows:
        csv_path = OUTPUT_DIR / "ridge_aad_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
