"""
Phase 0E — Temporal Alignment Audit (Highest Priority)
Sweeps EEG-audio correlation across physiological lags to verify
that the attended speech envelope is more strongly represented in
EEG than the unattended, and at the correct temporal offset.

Outputs:
  audit_results/temporal_alignment.csv  — per-trial, per-lag results
  audit_results/temporal_alignment_summary.csv — per-lag aggregate

Usage (Kaggle):
    !cd ISEF_Project && python scripts/verify_baseline/audit/audit_temporal_alignment.py --subjects S5_data_preproc S6_data_preproc
"""

from __future__ import annotations
import sys
import csv
import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import subject_files, load_subject_examples
from analysis._common import load_subject_data, fsample_values

OUTPUT_DIR = Path("audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)

EEG_FS = 64  # Downsampled EEG sampling rate (Hz)
AUDIO_FS = 64  # Audio envelope sampling rate

LAG_RANGE_MS = np.arange(-500, 1001, 50)  # -500ms to +1000ms in 50ms steps

MAPPING_CANDIDATES = [
    Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json"),
    REPO_ROOT / "scripts" / "verify_baseline" / "data" / "audio_mapping.json",
]

TEMPORAL_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]  # Use first 8 channels (temporal); override if needed


def find_envelope_file():
    direct = Path("/kaggle/input/datasets/lokeshgile/new-dtu-gammatones/gammatone_envelopes (1).pkl")
    if direct.exists():
        return direct
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob("*gammatone*.pkl"))
        if candidates:
            return candidates[0]
    return None


def load_mapping():
    for p in MAPPING_CANDIDATES:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def butter_bandpass(data, low, high, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


def pearson_corr_lag(eeg_ch: np.ndarray, envelope: np.ndarray, lag_samples: int) -> float:
    """
    Computes Pearson correlation between EEG channel and shifted envelope.
    Positive lag: EEG follows audio (physiologically expected).
    """
    n = len(eeg_ch)
    if lag_samples >= 0:
        # EEG(t) vs Audio(t - lag): shift audio forward
        e = eeg_ch[lag_samples:]
        a = envelope[:n - lag_samples]
    else:
        # Negative lag: audio follows EEG (anti-causal)
        lag_abs = -lag_samples
        e = eeg_ch[:n - lag_abs]
        a = envelope[lag_abs:]

    if len(e) < 10:
        return float("nan")

    e = e - e.mean()
    a = a - a.mean()
    denom = (np.std(e) * np.std(a) + 1e-12) * len(e)
    return float(np.dot(e, a) / denom)


def downsample_eeg(eeg_raw: np.ndarray, fs_raw: int, fs_target: int = EEG_FS) -> np.ndarray:
    """Downsample EEG from fs_raw to fs_target via decimation."""
    factor = fs_raw // fs_target
    return eeg_raw[:, ::factor] if eeg_raw.ndim == 2 else eeg_raw[::factor]


def audit_temporal_alignment(subjects_to_run=None):
    paths = subject_files()
    if subjects_to_run:
        paths = [p for p in paths if p.stem in subjects_to_run]

    mapping = load_mapping()
    env_path = find_envelope_file()

    print(f"\n{'='*70}")
    print("PHASE 0E — TEMPORAL ALIGNMENT AUDIT")
    print(f"{'='*70}")
    print(f"  Subjects: {[p.stem for p in paths]}")
    print(f"  Lag range: {LAG_RANGE_MS[0]}ms to {LAG_RANGE_MS[-1]}ms in {LAG_RANGE_MS[1]-LAG_RANGE_MS[0]}ms steps")
    print(f"  Envelope file: {env_path}")

    if env_path is None:
        print("ERROR: Envelope file not found. Cannot run temporal alignment audit.")
        return
    if mapping is None:
        print("ERROR: audio_mapping.json not found. Cannot run temporal alignment audit.")
        return

    with open(env_path, "rb") as f:
        envelopes = pickle.load(f)

    rows = []  # per-trial, per-lag
    lag_agg = {lag: {"delta_r": [], "r_att": [], "r_unatt": []} for lag in LAG_RANGE_MS}

    for path in paths:
        subject_id = path.stem
        subj_key = subject_id.split("_")[0]
        data = load_subject_data(path)
        fs_vals = fsample_values(data)
        fs_eeg_raw = fs_vals["eeg"]
        examples = load_subject_examples(path)

        print(f"\n  Processing: {subject_id} (fs_raw={fs_eeg_raw} Hz)")

        for trial_idx, ex in enumerate(examples):
            eeg_raw = ex.eeg  # [n_channels, n_samples_raw]
            label = ex.label  # 1 = attend A, 2 = attend B

            trial_key = f"trial_{trial_idx}"
            if subj_key not in mapping or trial_key not in mapping[subj_key]:
                continue

            fname_a = mapping[subj_key][trial_key].get("wavA", {}).get("filename", None)
            fname_b = mapping[subj_key][trial_key].get("wavB", {}).get("filename", None)
            if fname_a not in envelopes or fname_b not in envelopes:
                continue

            env_a = envelopes[fname_a]  # [n_bands, n_samples_audio]
            env_b = envelopes[fname_b]

            # Use broadband envelope: mean across bands
            env_a_broad = env_a.mean(axis=0) if env_a.ndim == 2 else env_a
            env_b_broad = env_b.mean(axis=0) if env_b.ndim == 2 else env_b

            # Low-pass envelope at 8 Hz (envelope tracking frequency band)
            env_a_lp = butter_lowpass(env_a_broad, cutoff=8.0, fs=AUDIO_FS)
            env_b_lp = butter_lowpass(env_b_broad, cutoff=8.0, fs=AUDIO_FS)

            # Downsample EEG and bandpass 1-8 Hz
            eeg_ds = downsample_eeg(eeg_raw, fs_raw=fs_eeg_raw, fs_target=EEG_FS)
            try:
                eeg_bp = butter_bandpass(eeg_ds, low=1.0, high=8.0, fs=EEG_FS)
            except Exception:
                eeg_bp = eeg_ds

            # Trim to minimum length
            min_len = min(eeg_bp.shape[1], len(env_a_lp), len(env_b_lp))
            eeg_bp = eeg_bp[:, :min_len]
            env_a_lp = env_a_lp[:min_len]
            env_b_lp = env_b_lp[:min_len]

            # Attended / unattended based on label
            if label == 1:
                env_att = env_a_lp
                env_unatt = env_b_lp
            else:  # label == 2
                env_att = env_b_lp
                env_unatt = env_a_lp

            # Use temporal channels only (or all channels, averaged)
            ch_indices = [i for i in TEMPORAL_CHANNELS if i < eeg_bp.shape[0]]
            eeg_temporal = eeg_bp[ch_indices, :].mean(axis=0)

            for lag_ms in LAG_RANGE_MS:
                lag_samples = int(round(lag_ms * EEG_FS / 1000.0))
                r_att = pearson_corr_lag(eeg_temporal, env_att, lag_samples)
                r_unatt = pearson_corr_lag(eeg_temporal, env_unatt, lag_samples)
                delta_r = r_att - r_unatt if not np.isnan(r_att) and not np.isnan(r_unatt) else float("nan")

                rows.append({
                    "subject": subject_id,
                    "trial_idx": trial_idx,
                    "label": label,
                    "lag_ms": int(lag_ms),
                    "r_attended": round(r_att, 5),
                    "r_unattended": round(r_unatt, 5),
                    "delta_r": round(delta_r, 5),
                })

                if not np.isnan(delta_r):
                    lag_agg[lag_ms]["delta_r"].append(delta_r)
                    lag_agg[lag_ms]["r_att"].append(r_att)
                    lag_agg[lag_ms]["r_unatt"].append(r_unatt)

    # Save per-trial CSV
    if rows:
        csv_path = OUTPUT_DIR / "temporal_alignment.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  -> Per-trial alignment CSV: {csv_path}")

    # Compute and print aggregate lag profile
    summary_rows = []
    print(f"\n{'='*70}")
    print("AGGREGATE LAG PROFILE")
    print(f"{'='*70}")
    print(f"{'lag_ms':>8} | {'mean_r_att':>12} | {'mean_r_unatt':>14} | {'mean_delta_r':>14} | n_trials")
    print("-" * 65)

    best_lag = None
    best_delta = -999.0
    for lag_ms in LAG_RANGE_MS:
        vals = lag_agg[lag_ms]
        if not vals["delta_r"]:
            continue
        m_att = np.mean(vals["r_att"])
        m_unatt = np.mean(vals["r_unatt"])
        m_delta = np.mean(vals["delta_r"])
        n = len(vals["delta_r"])

        print(f"{int(lag_ms):>8} | {m_att:>12.5f} | {m_unatt:>14.5f} | {m_delta:>14.5f} | {n}")
        summary_rows.append({
            "lag_ms": int(lag_ms),
            "mean_r_attended": round(float(m_att), 5),
            "mean_r_unattended": round(float(m_unatt), 5),
            "mean_delta_r": round(float(m_delta), 5),
            "n_trials": n,
        })
        if m_delta > best_delta:
            best_delta = m_delta
            best_lag = lag_ms

    if summary_rows:
        summ_path = OUTPUT_DIR / "temporal_alignment_summary.csv"
        with open(summ_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  -> Summary CSV: {summ_path}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  Peak Δr = {best_delta:.5f} at lag = {best_lag}ms")
    print()
    if best_lag is not None and 0 < best_lag <= 300 and best_delta > 0.005:
        verdict = "PASS"
        print(f"  FINDING: Attended speech correlation peaks at {best_lag}ms lag")
        print(f"  This is physiologically plausible (auditory cortex response latency ~50-200ms)")
        print(f"  INTERPRETATION: EEG/audio temporal alignment is likely correct")
    elif best_lag is not None and best_delta > 0.001:
        verdict = "MARGINAL"
        print(f"  FINDING: Weak positive Δr at {best_lag}ms — signal is present but weak")
        print(f"  INTERPRETATION: Signal exists but may be noisy or preprocessing dampens it")
    elif best_delta <= 0:
        verdict = "FAIL"
        print(f"  FINDING: Δr is not positive at any lag")
        print(f"  INTERPRETATION: No evidence of attended speech tracking in EEG")
        print(f"  ACTION: Investigate alignment, reference, and filter choices before proceeding")
    else:
        verdict = "UNCLEAR"

    print(f"\nPHASE 0E VERDICT: {verdict}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0E: Temporal alignment audit")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs to audit (default: all)")
    args = parser.parse_args()
    audit_temporal_alignment(args.subjects)
