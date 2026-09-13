"""
Ridge Stimulus Reconstruction — LOSO and Within-Subject AAD Baseline
====================================================================
KEY FIXES applied here:

1. LABEL CONVENTION (confirmed by audit_eeg_signal.py):
   label=1 = 'attend left' = wavB
   label=2 = 'attend right' = wavA

2. FUTURE-EEG LAGS (backward model):
   The TRF backward model predicts audio(t) from FUTURE EEG:
     audio(t) ≈ Σ_τ g(τ) × EEG(t + τ),  τ ∈ [50ms, 250ms]
   We implement this by time-reversing the EEG, applying the standard
   past-lag matrix, then time-reversing back. This is the standard
   mTRF approach.

3. LOWPASS FILTERING (1-8 Hz tracking band):
   EEG and speech envelope are both lowpass-filtered to 8 Hz before
   Ridge. The AAD signal lives in the delta/theta band (1-8 Hz).
   Broadband EEG (1-32 Hz) drowns the signal in high-frequency noise.

Usage (Kaggle):
    # Smoke test (2 min):
    !python train_ridge_loso.py --mode within --pilot

    # Full run:
    !python train_ridge_loso.py --mode both
"""

from __future__ import annotations

import argparse, sys, csv
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import (
    subject_files, load_subject_examples, iter_leave_one_subject_out,
    TrialExample, _lag_samples_from_ms, standardize_features, feature_statistics,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SCALP_CHANNELS = list(range(64))    # exclude EXG1 (64), EXG2 (65)
FS             = 64                  # Hz
LAG_MS         = 250                 # future-EEG lag range (ms)
LAG_STEP_MS    = 16                  # 1 sample at 64 Hz
RIDGE_LAMBDA   = 1e4                 # Ridge regularisation
LOWPASS_HZ     = 8.0                 # tracking band cutoff (Hz)

WINDOW_SIZES_S = [1, 2, 5, 10, 20, 40]
WITHIN_FOLDS   = 8

OUTPUT_DIR = Path("ridge_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Filtering ────────────────────────────────────────────────────────────────

def _lp_sos(cutoff: float, fs: int, order: int = 4):
    return butter(order, cutoff / (fs / 2.0), btype="low", output="sos")

_LP_SOS = _lp_sos(LOWPASS_HZ, FS)

def lowpass(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Apply lowpass filter along the specified axis."""
    return sosfiltfilt(_LP_SOS, x, axis=axis)


def normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    return x / (x.std() + 1e-12)


# ─── Envelope helpers (DTU: wavA/wavB are pre-extracted envelopes) ─────────────

def get_envelopes(ex: TrialExample) -> tuple[np.ndarray, np.ndarray]:
    """Return (env_A, env_B) — lowpass filtered and normalized.
    wavA/wavB are pre-extracted amplitude envelopes at 64 Hz in the MAT file.
    """
    ea = normalize_1d(lowpass(ex.wav_a))
    eb = normalize_1d(lowpass(ex.wav_b))
    return ea, eb


def attended_env(ex: TrialExample) -> np.ndarray:
    """Return the attended envelope.
    CRITICAL FIX (DATASETS_REFERENCE.md):
      In the preprocessed DTU .mat files, wavA is ALWAYS the attended stream.
      wavB is ALWAYS the unattended stream.
      Event labels (1 or 2) indicate speaker gender, NOT attention!
    """
    ea, _ = get_envelopes(ex)
    return ea


# ─── Future-lag EEG matrix (backward model) ────────────────────────────────────

def future_lagged_eeg(eeg: np.ndarray) -> np.ndarray:
    """Build the lagged feature matrix using FUTURE EEG lags.

    Backward model: audio(t) ≈ Σ_τ g(τ) × EEG(t + τ)
    Implemented by: reverse-time EEG → past-lag matrix → reverse back.

    Returns X with shape [n_samples, n_channels * n_lags].
    At row t: [eeg(t), eeg(t+1), ..., eeg(t+L)] (future EEG).
    """
    # 1. Lowpass filter EEG in the tracking band
    eeg_lp = lowpass(eeg, axis=0)    # [n_samples, n_channels]

    # 2. Normalize each channel
    eeg_lp = eeg_lp - eeg_lp.mean(axis=0, keepdims=True)
    scale = eeg_lp.std(axis=0, keepdims=True) + 1e-12
    eeg_lp = eeg_lp / scale

    # 3. Time-reverse → past lags in reversed time = future lags in original time
    eeg_rev = eeg_lp[::-1, :]   # [n_samples, n_channels]

    # 4. Build past-lagged matrix on the reversed signal
    lag_offsets = _lag_samples_from_ms(lag_ms=LAG_MS, lag_step_ms=LAG_STEP_MS, fs=FS)
    n, c = eeg_rev.shape
    blocks = []
    for lag in lag_offsets:
        if lag == 0:
            blocks.append(eeg_rev)
        else:
            shifted = np.vstack([np.zeros((lag, c), dtype=float), eeg_rev[:n - lag]])
            blocks.append(shifted)

    X_rev = np.concatenate(blocks, axis=1)   # [n_samples, n_ch * n_lags]

    # 5. Time-reverse back → future lags in original time
    return X_rev[::-1, :]


def n_features(n_channels: int) -> int:
    n_lags = len(_lag_samples_from_ms(lag_ms=LAG_MS, lag_step_ms=LAG_STEP_MS, fs=FS))
    return n_channels * n_lags


# ─── Ridge fitting ─────────────────────────────────────────────────────────────

def fit_ridge(
    examples: list[TrialExample],
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> np.ndarray:
    """Fit Ridge: EEG_future_lagged → attended_envelope."""
    print(f"      [Ridge] Fitting model over {len(examples)} trials (incremental XtX)...")
    
    # We must build XtX incrementally to avoid OOM (1020 trials = 28GB of RAM if concatenated!)
    n_feat = n_features(examples[0].eeg.shape[1])
    XtX = np.zeros((n_feat, n_feat), dtype=np.float64)
    Xty = np.zeros(n_feat, dtype=np.float64)

    for i, ex in enumerate(examples):
        if i % 100 == 0 and i > 0:
            print(f"      [Ridge] Processed {i}/{len(examples)} trials...")
            
        X_trial = future_lagged_eeg(ex.eeg)
        y_trial = attended_env(ex)
        n = min(X_trial.shape[0], len(y_trial))
        
        X = X_trial[:n]
        y = y_trial[:n]
        
        XtX += X.T @ X
        Xty += X.T @ y

    print("      [Ridge] Solving linear system...")
    return np.linalg.solve(XtX + ridge_lambda * np.eye(n_feat), Xty)


def predict_envelope(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reconstruct envelope from pre-computed feature matrix using the trained weights."""
    return X @ weights


# ─── Evaluation ───────────────────────────────────────────────────────────────

def select_channels(ex: TrialExample, channels: list[int]) -> TrialExample:
    return TrialExample(
        subject=ex.subject, trial_index=ex.trial_index,
        eeg=ex.eeg[:, channels], wav_a=ex.wav_a, wav_b=ex.wav_b, label=ex.label,
    )


def eval_windows(
    ex: TrialExample,
    weights: np.ndarray,
    window_sec: int,
) -> tuple[int, int]:
    """Non-overlapping window evaluation. Returns (n_correct, n_windows)."""
    env_a, env_b = get_envelopes(ex)
    
    # Pre-compute the feature matrix for the entire trial ONCE.
    # Doing this per-window breaks sosfiltfilt (boundary effects on tiny chunks).
    X_full = future_lagged_eeg(ex.eeg)
    
    n = min(X_full.shape[0], len(env_a), len(env_b))

    win = window_sec * FS
    n_correct = n_windows = 0
    start = 0

    while start + win <= n:
        end = start + win
        pred = predict_envelope(X_full[start:end], weights)
        L = min(len(pred), win)
        pred  = pred[:L]
        ea_w  = env_a[start:start + L]
        eb_w  = env_b[start:start + L]

        # Handle NaNs that might occur if the EEG chunk was perfectly flat
        if np.isnan(pred).any():
            start += win
            continue

        ca, _ = pearsonr(pred, ea_w)
        cb, _ = pearsonr(pred, eb_w)

        # wavA is ALWAYS attended, wavB is ALWAYS unattended
        if ca > cb:
            n_correct += 1

        n_windows += 1

        start += win

    return n_correct, n_windows


# ─── LOSO ─────────────────────────────────────────────────────────────────────

def run_loso(subjects_filter=None, pilot=False):
    print(f"\n{'='*70}")
    print(f"RIDGE AAD — LOSO  [{LOWPASS_HZ}Hz LP, future lags, label: 1=wavB 2=wavA]")
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
    totals = {w: [0, 0] for w in WINDOW_SIZES_S}

    for held_path, train_paths in folds:
        hid = held_path.stem
        train_exs = [select_channels(ex, SCALP_CHANNELS)
                     for p in train_paths for ex in all_exs[str(p)]]
        test_exs  = [select_channels(ex, SCALP_CHANNELS)
                     for ex in all_exs[str(held_path)]]

        print(f"\n  Held-out: {hid} | Train: {len(train_paths)} subj, {len(train_exs)} trials")

        weights = fit_ridge(train_exs)

        for ws in WINDOW_SIZES_S:
            nc = nt = 0
            for ex in test_exs:
                c, t = eval_windows(ex, weights, ws)
                nc += c; nt += t
            acc = 100 * nc / max(nt, 1)
            totals[ws][0] += nc; totals[ws][1] += nt
            rows.append({"mode":"LOSO","subject":hid,"window_s":ws,
                         "n_correct":nc,"n_total":nt,"accuracy_pct":round(acc, 2)})
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
    print(f"RIDGE AAD — WITHIN-SUBJECT  [{WITHIN_FOLDS}-fold CV, {LOWPASS_HZ}Hz LP, future lags]")
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

            weights = fit_ridge(tr)

            for ws in WINDOW_SIZES_S:
                for ex in te:
                    c, t = eval_windows(ex, weights, ws)
                    ws_results[ws][0] += c; ws_results[ws][1] += t

        accs = "  ".join(
            f"{ws}s={100*ws_results[ws][0]/max(ws_results[ws][1],1):.1f}%"
            for ws in WINDOW_SIZES_S
        )
        print(f"  {sid}: {accs}")

        for ws in WINDOW_SIZES_S:
            nc, nt = ws_results[ws]
            totals[ws][0] += nc; totals[ws][1] += nt
            rows.append({"mode":"within","subject":sid,"window_s":ws,
                         "n_correct":nc,"n_total":nt,
                         "accuracy_pct":round(100*nc/max(nt,1),2)})

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
