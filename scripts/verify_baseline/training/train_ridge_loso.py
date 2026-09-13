"""
Ridge Stimulus Reconstruction — LOSO and Within-Subject AAD Baseline
====================================================================
CRITICAL FIX: The DTU preprocessed MAT files store wavA/wavB as
PRE-EXTRACTED speech envelopes at 64 Hz (same rate as EEG), NOT raw
audio waveforms. Calling Hilbert() on an already-extracted envelope
produces garbage. We use wavA/wavB DIRECTLY as the regression targets,
applying only z-score normalization.

Decision rule (per window):
  1. Fit Ridge decoder: W = Ridge(EEG_lagged → attended_envelope)
  2. Reconstruct:       r = EEG_lagged @ W
  3. Compare:          corr(r, env_A) vs corr(r, env_B)
  4. Choose higher correlation as attended.

Usage (Kaggle):
    # Smoke test first (2 min):
    !python train_ridge_loso.py --mode within --pilot

    # Full run:
    !python train_ridge_loso.py --mode both
"""

from __future__ import annotations

import argparse
import sys
import csv
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import (
    subject_files,
    load_subject_examples,
    iter_leave_one_subject_out,
    TrialExample,
    normalize_eeg,
    lagged_eeg_matrix,
    _lag_samples_from_ms,
    standardize_features,
    feature_statistics,
    predict_envelope,
)

# ─── Configuration ────────────────────────────────────────────────────────────

# DTU: 66 channels = 64 scalp EEG + EXG1 (idx 64) + EXG2 (idx 65). Exclude EXG.
SCALP_CHANNELS = list(range(64))

FS          = 64     # Hz (EEG and envelopes, both already at 64 Hz in MAT file)
LAG_MS      = 250    # max lag for TRF decoder (ms)
LAG_STEP_MS = 16     # lag step (1 sample at 64 Hz)
RIDGE_LAMBDA = 1e4   # regularisation

WINDOW_SIZES_S    = [1, 2, 5, 10, 20, 40]
WITHIN_FOLDS      = 8

OUTPUT_DIR = Path("ridge_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Core: use MAT envelopes directly (NO Hilbert) ────────────────────────────

def normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    return x / (x.std() + 1e-12)


def get_envelopes(ex: TrialExample) -> tuple[np.ndarray, np.ndarray]:
    """Return (env_A, env_B) directly from the MAT-stored envelopes.
    DTU preprocessing already extracted the broadband speech envelope at 64 Hz.
    We must NOT apply Hilbert again — just normalize.
    """
    return normalize_1d(ex.wav_a), normalize_1d(ex.wav_b)


def attended_env(ex: TrialExample) -> np.ndarray:
    """Return the attended envelope.
    CONFIRMED DTU convention (audit_eeg_signal.py):
      label=1 = 'attend left' = wavB
      label=2 = 'attend right' = wavA
    """
    env_a, env_b = get_envelopes(ex)
    if ex.label == 1:
        return env_b   # attend wavB
    elif ex.label == 2:
        return env_a   # attend wavA
    raise ValueError(f"Unexpected label: {ex.label}")


# ─── Ridge fitting ─────────────────────────────────────────────────────────────

def fit_ridge_direct(
    examples: list[TrialExample],
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    """Fit linear decoder: EEG_lagged → attended_envelope (from MAT directly)."""
    n_lags = len(_lag_samples_from_ms(lag_ms=LAG_MS, lag_step_ms=LAG_STEP_MS, fs=FS))
    n_feat = examples[0].eeg.shape[1] * n_lags

    XtX = np.zeros((n_feat, n_feat), dtype=float)
    Xty = np.zeros(n_feat, dtype=float)

    for ex in examples:
        X = lagged_eeg_matrix(ex.eeg, lag_ms=LAG_MS, lag_step_ms=LAG_STEP_MS, fs=FS)
        if feature_mean is not None and feature_std is not None:
            X = standardize_features(X, feature_mean, feature_std)
        y = attended_env(ex)
        n = min(X.shape[0], len(y))
        X, y = X[:n], y[:n]
        XtX += X.T @ X
        Xty += X.T @ y

    return np.linalg.solve(XtX + ridge_lambda * np.eye(n_feat), Xty)


def feature_stats_direct(examples: list[TrialExample]) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature mean/std for standardisation across the training set."""
    return feature_statistics(examples, lag_ms=LAG_MS, lag_step_ms=LAG_STEP_MS, fs=FS)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def select_channels(ex: TrialExample, channels: list[int]) -> TrialExample:
    return TrialExample(
        subject=ex.subject,
        trial_index=ex.trial_index,
        eeg=ex.eeg[:, channels],
        wav_a=ex.wav_a,
        wav_b=ex.wav_b,
        label=ex.label,
    )


def eval_windows(
    ex: TrialExample,
    weights: np.ndarray,
    window_sec: int,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[int, int]:
    """Non-overlapping window evaluation. Returns (n_correct, n_windows)."""
    env_a, env_b = get_envelopes(ex)
    eeg = ex.eeg
    n = min(eeg.shape[0], len(env_a), len(env_b))

    win = window_sec * FS
    n_correct = n_windows = 0
    start = 0

    while start + win <= n:
        end = start + win
        pred = predict_envelope(
            eeg[start:end],
            weights,
            lag_ms=LAG_MS, lag_step_ms=LAG_STEP_MS, fs=FS,
            feature_mean=feature_mean, feature_std=feature_std,
        )
        L = min(len(pred), win)
        pred   = pred[:L]
        ea_w   = env_a[start:start + L]
        eb_w   = env_b[start:start + L]

        ca, _ = pearsonr(pred, ea_w)
        cb, _ = pearsonr(pred, eb_w)

        # label=1: attend wavB → correct if cb > ca
        # label=2: attend wavA → correct if ca > cb
        if ex.label == 1 and cb > ca:   # attend wavB
            n_correct += 1
        elif ex.label == 2 and ca > cb:  # attend wavA
            n_correct += 1

        n_windows += 1
        start += win

    return n_correct, n_windows


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def run_loso(subjects_filter=None, pilot=False):
    print(f"\n{'='*70}")
    print("RIDGE AAD — LOSO  (envelopes used directly, no Hilbert)")
    print(f"{'='*70}")

    paths = subject_files()
    if subjects_filter:
        paths = [p for p in paths if p.stem in subjects_filter]
    all_exs = {str(p): load_subject_examples(p) for p in paths}

    folds = list(iter_leave_one_subject_out(paths))
    if pilot:
        folds = folds[:1]
        print(f"  PILOT: 1 fold only ({folds[0][0].stem})")

    rows = []
    totals = {w: [0, 0] for w in WINDOW_SIZES_S}  # [n_correct, n_total]

    for held_path, train_paths in folds:
        hid = held_path.stem
        train_exs = [select_channels(ex, SCALP_CHANNELS)
                     for p in train_paths for ex in all_exs[str(p)]]
        test_exs  = [select_channels(ex, SCALP_CHANNELS)
                     for ex in all_exs[str(held_path)]]

        print(f"\n  Held-out: {hid} | Train: {len(train_paths)} subjects, {len(train_exs)} trials")

        fm, fs = feature_stats_direct(train_exs)
        weights = fit_ridge_direct(train_exs, feature_mean=fm, feature_std=fs)

        for ws in WINDOW_SIZES_S:
            nc = nt = 0
            for ex in test_exs:
                c, t = eval_windows(ex, weights, ws, feature_mean=fm, feature_std=fs)
                nc += c; nt += t
            acc = 100 * nc / max(nt, 1)
            totals[ws][0] += nc; totals[ws][1] += nt
            rows.append({"mode":"LOSO","subject":hid,"window_s":ws,"n_correct":nc,"n_total":nt,"accuracy_pct":round(acc,2)})
            print(f"    [{ws:>2}s] {acc:.1f}%  ({nc}/{nt})")

    print(f"\n{'='*70}\nLOSO SUMMARY\n{'='*70}")
    print(f"{'Window':>8} | {'Accuracy':>10} | Windows")
    for ws in WINDOW_SIZES_S:
        nc, nt = totals[ws]
        print(f"{ws:>6}s   | {100*nc/max(nt,1):>9.1f}%  | {nt}")
    return rows


# ─── Within-subject ───────────────────────────────────────────────────────────

def run_within(subjects_filter=None, pilot=False):
    print(f"\n{'='*70}")
    print(f"RIDGE AAD — WITHIN-SUBJECT  ({WITHIN_FOLDS}-fold CV, envelopes direct)")
    print(f"{'='*70}")

    paths = subject_files()
    if subjects_filter:
        paths = [p for p in paths if p.stem in subjects_filter]
    if pilot:
        paths = paths[:1]
        print(f"  PILOT: 1 subject only ({paths[0].stem})")
    all_exs = {str(p): load_subject_examples(p) for p in paths}

    rows = []
    totals = {w: [0, 0] for w in WINDOW_SIZES_S}

    for path in paths:
        sid = path.stem
        exs = [select_channels(ex, SCALP_CHANNELS) for ex in all_exs[str(path)]]
        n = len(exs)
        fold_size = n // WITHIN_FOLDS
        idx = np.random.RandomState(42).permutation(n)

        ws_results = {ws: [0, 0] for ws in WINDOW_SIZES_S}

        for fold in range(WITHIN_FOLDS):
            test_idx  = idx[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
            tr = [exs[i] for i in train_idx]
            te = [exs[i] for i in test_idx]

            fm, fs = feature_stats_direct(tr)
            weights = fit_ridge_direct(tr, feature_mean=fm, feature_std=fs)

            for ws in WINDOW_SIZES_S:
                for ex in te:
                    c, t = eval_windows(ex, weights, ws, feature_mean=fm, feature_std=fs)
                    ws_results[ws][0] += c; ws_results[ws][1] += t

        accs = "  ".join(f"{ws}s={100*ws_results[ws][0]/max(ws_results[ws][1],1):.1f}%" for ws in WINDOW_SIZES_S)
        print(f"  {sid}: {accs}")

        for ws in WINDOW_SIZES_S:
            nc, nt = ws_results[ws]
            totals[ws][0] += nc; totals[ws][1] += nt
            rows.append({"mode":"within","subject":sid,"window_s":ws,
                         "n_correct":nc,"n_total":nt,"accuracy_pct":round(100*nc/max(nt,1),2)})

    print(f"\n{'='*70}\nWITHIN-SUBJECT SUMMARY\n{'='*70}")
    print(f"{'Window':>8} | {'Accuracy':>10} | Windows")
    for ws in WINDOW_SIZES_S:
        nc, nt = totals[ws]
        print(f"{ws:>6}s   | {100*nc/max(nt,1):>9.1f}%  | {nt}")
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["loso","within","both"], default="both")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--pilot", action="store_true",
                   help="Smoke-test: 1 fold/subject only")
    args = p.parse_args()

    rows = []
    if args.mode in ("loso","both"):
        rows += run_loso(args.subjects, args.pilot)
    if args.mode in ("within","both"):
        rows += run_within(args.subjects, args.pilot)

    if rows:
        out = OUTPUT_DIR / "ridge_aad_results.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\nResults → {out}")

if __name__ == "__main__":
    main()
