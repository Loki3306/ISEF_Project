"""
Diagnostic: Is there any EEG-speech correlation signal in the DTU data?

Sweeps both wavA-from-MAT and gammatone-from-pkl as the envelope source.
For each, computes the mean cross-correlation between EEG channels and
the attended/unattended envelope across lags -500ms to +500ms.

If BOTH show Δr ≈ 0 at all lags → data is broken.
If gammatone shows signal but wavA doesn't → wavA is the wrong representation.
If wavA shows signal but Ridge is at chance → lag direction or lambda is wrong.

Usage (Kaggle):
    !cd /kaggle/working/ISEF_Project && python scripts/verify_baseline/audit/audit_eeg_signal.py
"""

from __future__ import annotations
import sys, json, pickle, numpy as np
from pathlib import Path
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import subject_files, load_subject_examples

DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/datasets/lokeshgile/new-dtu-gammatones/gammatone_envelopes (1).pkl"),
]
MAPPING_CANDIDATES = [
    Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json"),
    REPO_ROOT / "scripts" / "verify_baseline" / "data" / "audio_mapping.json",
]

FS = 64
LAG_MS_RANGE = np.arange(-500, 501, 50)   # -500ms to +500ms
TEST_SUBJECTS = ["S1_data_preproc", "S5_data_preproc"]
TEMPORAL_CHANNELS = list(range(64))        # all scalp channels


def find_env_pkl():
    for p in DATA_DIR_CANDIDATES:
        if p.exists(): return p
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*gammatone*.pkl"):
            return p
    return None


def load_mapping():
    for p in MAPPING_CANDIDATES:
        if p.exists():
            with open(p) as f: return json.load(f)
    return None


def pearson_lag(eeg_ch: np.ndarray, env: np.ndarray, lag_ms: float) -> float:
    lag = int(round(lag_ms * FS / 1000.0))
    n = len(eeg_ch)
    if lag >= 0:                             # audio leads EEG (causal)
        e = eeg_ch[lag:]
        a = env[:n - lag]
    else:                                    # EEG leads audio (anti-causal)
        lag_abs = -lag
        e = eeg_ch[:n - lag_abs]
        a = env[lag_abs:]
    if len(e) < 10: return float("nan")
    r, _ = pearsonr(e - e.mean(), a - a.mean())
    return float(r)


def normalize(x): 
    x = np.asarray(x, dtype=float).ravel()
    return (x - x.mean()) / (x.std() + 1e-12)


def run():
    paths = subject_files()
    paths = [p for p in paths if p.stem in TEST_SUBJECTS]
    mapping = load_mapping()
    env_pkl_path = find_env_pkl()

    if env_pkl_path:
        print(f"Gammatone pkl: {env_pkl_path}")
        with open(env_pkl_path, "rb") as f:
            envelopes = pickle.load(f)
    else:
        envelopes = None
        print("WARNING: Gammatone pkl not found — skipping gammatone test.")

    print(f"\n{'='*70}")
    print("EEG-SPEECH SIGNAL DIAGNOSTIC")
    print(f"Subjects: {TEST_SUBJECTS}")
    print(f"Lags: {LAG_MS_RANGE[0]}ms to {LAG_MS_RANGE[-1]}ms")
    print(f"{'='*70}\n")

    # Accumulators: lag → list of (r_att - r_unatt) per trial per channel
    lag_delta_wav   = {l: [] for l in LAG_MS_RANGE}
    lag_delta_gamma = {l: [] for l in LAG_MS_RANGE}

    for path in paths:
        sid = path.stem
        subj_key = sid.split("_")[0]
        exs = load_subject_examples(path)

        print(f"Subject: {sid} ({len(exs)} trials)")

        for ex in exs:
            eeg = ex.eeg  # [n_samples, 66]

            # ── Source 1: wavA/wavB directly from MAT ──
            env_a_wav = normalize(ex.wav_a)
            env_b_wav = normalize(ex.wav_b)

            # ── Source 2: gammatone pkl (broadband mean) ──
            env_a_gamma = env_b_gamma = None
            if envelopes and mapping:
                trial_key = f"trial_{ex.trial_index}"
                if subj_key in mapping and trial_key in mapping[subj_key]:
                    fa = mapping[subj_key][trial_key].get("wavA", {}).get("filename")
                    fb = mapping[subj_key][trial_key].get("wavB", {}).get("filename")
                    if fa in envelopes and fb in envelopes:
                        # Mean across 28 gammatone bands → broadband envelope
                        env_a_gamma = normalize(envelopes[fa].mean(axis=0))
                        env_b_gamma = normalize(envelopes[fb].mean(axis=0))

            # Attended / unattended assignment
            if ex.label == 1:
                att_wav, unatt_wav = env_a_wav, env_b_wav
                att_gamma, unatt_gamma = env_a_gamma, env_b_gamma
            else:
                att_wav, unatt_wav = env_b_wav, env_a_wav
                att_gamma, unatt_gamma = env_b_gamma, env_a_gamma

            n = eeg.shape[0]
            min_len = min(n, len(att_wav))

            # Average over temporal channels (best channels for AAD tracking)
            eeg_mean = eeg[:min_len, TEMPORAL_CHANNELS].mean(axis=1)

            for lag_ms in LAG_MS_RANGE:
                r_att = pearson_lag(eeg_mean, att_wav[:min_len], lag_ms)
                r_un  = pearson_lag(eeg_mean, unatt_wav[:min_len], lag_ms)
                if not (np.isnan(r_att) or np.isnan(r_un)):
                    lag_delta_wav[lag_ms].append(r_att - r_un)

                if att_gamma is not None:
                    min_g = min(min_len, len(att_gamma))
                    r_att_g = pearson_lag(eeg_mean, att_gamma[:min_g], lag_ms)
                    r_un_g  = pearson_lag(eeg_mean, unatt_gamma[:min_g], lag_ms)
                    if not (np.isnan(r_att_g) or np.isnan(r_un_g)):
                        lag_delta_gamma[lag_ms].append(r_att_g - r_un_g)

    # ── Print results ──
    print(f"\n{'lag_ms':>8} | {'Δr (wavA-MAT)':>14} | {'Δr (gammatone)':>16}")
    print("-" * 45)
    best_wav = best_gamma = (-999, 0)
    for lag_ms in LAG_MS_RANGE:
        dw = np.mean(lag_delta_wav[lag_ms]) if lag_delta_wav[lag_ms] else float("nan")
        dg = np.mean(lag_delta_gamma[lag_ms]) if lag_delta_gamma[lag_ms] else float("nan")
        print(f"{int(lag_ms):>8} | {dw:>14.5f} | {dg:>16.5f}")
        if not np.isnan(dw) and dw > best_wav[0]: best_wav = (dw, lag_ms)
        if not np.isnan(dg) and dg > best_gamma[0]: best_gamma = (dg, lag_ms)

    print(f"\nBest Δr (wavA-MAT):    {best_wav[0]:.5f} at lag {best_wav[1]}ms")
    print(f"Best Δr (gammatone):   {best_gamma[0]:.5f} at lag {best_gamma[1]}ms")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    for name, (best_dr, _) in [("wavA-MAT", best_wav), ("gammatone", best_gamma)]:
        if best_dr > 0.02:
            print(f"  {name}: SIGNAL PRESENT (Δr={best_dr:.4f})")
        elif best_dr > 0.005:
            print(f"  {name}: WEAK SIGNAL (Δr={best_dr:.4f})")
        else:
            print(f"  {name}: NO SIGNAL (Δr={best_dr:.4f}) — investigate data!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
