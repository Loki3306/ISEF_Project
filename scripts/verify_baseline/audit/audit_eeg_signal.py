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
    print(f"{'='*70}\n")

    # Accumulators — normal AND inverted labels
    lag_delta_wav_norm = {l: [] for l in LAG_MS_RANGE}
    lag_delta_wav_inv  = {l: [] for l in LAG_MS_RANGE}
    lag_delta_gam_norm = {l: [] for l in LAG_MS_RANGE}

    wav_a_vals, wav_b_vals = [], []   # for statistics check

    for path in paths:
        sid = path.stem
        subj_key = sid.split("_")[0]
        exs = load_subject_examples(path)
        print(f"Subject: {sid} ({len(exs)} trials)")

        for ex in exs:
            eeg = ex.eeg   # [n_samples, 66]
            env_a = normalize(ex.wav_a)
            env_b = normalize(ex.wav_b)
            wav_a_vals.append(ex.wav_a.ravel())
            wav_b_vals.append(ex.wav_b.ravel())

            # gammatone broadband
            ga = gb = None
            if envelopes and mapping:
                tk = f"trial_{ex.trial_index}"
                if subj_key in mapping and tk in mapping[subj_key]:
                    fa = mapping[subj_key][tk].get("wavA", {}).get("filename")
                    fb = mapping[subj_key][tk].get("wavB", {}).get("filename")
                    if fa in envelopes and fb in envelopes:
                        ga = normalize(envelopes[fa].mean(axis=0))
                        gb = normalize(envelopes[fb].mean(axis=0))

            # NORMAL label convention: label=1 → attend wavA
            if ex.label == 1:
                att, unatt = env_a, env_b
                gatt, gunatt = ga, gb
            else:
                att, unatt = env_b, env_a
                gatt, gunatt = gb, ga

            # INVERTED label convention: label=1 → attend wavB
            if ex.label == 1:
                att_inv, unatt_inv = env_b, env_a
            else:
                att_inv, unatt_inv = env_a, env_b

            n = min(eeg.shape[0], len(att))
            # Mean over all scalp channels (simple average)
            eeg_mean = eeg[:n, :64].mean(axis=1)

            for lag_ms in LAG_MS_RANGE:
                rn  = pearson_lag(eeg_mean, att[:n], lag_ms)
                run = pearson_lag(eeg_mean, unatt[:n], lag_ms)
                ri  = pearson_lag(eeg_mean, att_inv[:n], lag_ms)
                rin = pearson_lag(eeg_mean, unatt_inv[:n], lag_ms)

                if not (np.isnan(rn) or np.isnan(run)):
                    lag_delta_wav_norm[lag_ms].append(rn - run)
                if not (np.isnan(ri) or np.isnan(rin)):
                    lag_delta_wav_inv[lag_ms].append(ri - rin)

                if gatt is not None:
                    ng = min(n, len(gatt))
                    rg  = pearson_lag(eeg_mean, gatt[:ng], lag_ms)
                    rgu = pearson_lag(eeg_mean, gunatt[:ng], lag_ms)
                    if not (np.isnan(rg) or np.isnan(rgu)):
                        lag_delta_gam_norm[lag_ms].append(rg - rgu)

    # ── wavA/wavB statistics ──────────────────────────────────────────────────
    all_wav_a = np.concatenate(wav_a_vals)
    all_wav_b = np.concatenate(wav_b_vals)
    print(f"\n── wavA stats (raw):  mean={all_wav_a.mean():.4f}  std={all_wav_a.std():.4f}  "
          f"min={all_wav_a.min():.4f}  max={all_wav_a.max():.4f}")
    print(f"── wavB stats (raw):  mean={all_wav_b.mean():.4f}  std={all_wav_b.std():.4f}  "
          f"min={all_wav_b.min():.4f}  max={all_wav_b.max():.4f}")
    corr_ab, _ = pearsonr(all_wav_a[:10000], all_wav_b[:10000])
    print(f"── Corr(wavA, wavB) (first 10k samples):  r={corr_ab:.4f}")
    print()

    # ── Correlation table ─────────────────────────────────────────────────────
    print(f"{'lag_ms':>8} | {'Δr NORMAL':>12} | {'Δr INVERTED':>13} | {'Δr GAMMA':>10}")
    print("-" * 55)
    best_norm  = (-999, 0)
    best_inv   = (-999, 0)
    best_gam   = (-999, 0)
    for lag_ms in LAG_MS_RANGE:
        dn  = np.mean(lag_delta_wav_norm[lag_ms]) if lag_delta_wav_norm[lag_ms] else float("nan")
        di  = np.mean(lag_delta_wav_inv[lag_ms])  if lag_delta_wav_inv[lag_ms]  else float("nan")
        dg  = np.mean(lag_delta_gam_norm[lag_ms]) if lag_delta_gam_norm[lag_ms] else float("nan")
        print(f"{int(lag_ms):>8} | {dn:>12.5f} | {di:>13.5f} | {dg:>10.5f}")
        if not np.isnan(dn) and dn > best_norm[0]: best_norm = (dn, lag_ms)
        if not np.isnan(di) and di > best_inv[0]:  best_inv  = (di, lag_ms)
        if not np.isnan(dg) and dg > best_gam[0]:  best_gam  = (dg, lag_ms)

    print(f"\nBest Δr (NORMAL labels):    {best_norm[0]:.5f}  at {best_norm[1]}ms")
    print(f"Best Δr (INVERTED labels):  {best_inv[0]:.5f}  at {best_inv[1]}ms")
    print(f"Best Δr (gammatone):        {best_gam[0]:.5f}  at {best_gam[1]}ms")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if best_inv[0] > 0.005 and best_inv[0] > best_norm[0]:
        print("  *** LABEL INVERSION CONFIRMED ***")
        print(f"  Inverted labels give Δr={best_inv[0]:.4f} vs Normal Δr={best_norm[0]:.4f}")
        print("  FIX: swap att/unatt convention — label=1 means attend wavB, not wavA")
    elif best_norm[0] > 0.005:
        print(f"  Signal present with NORMAL labels (Δr={best_norm[0]:.4f})")
        print("  Labels are correct. Check Ridge bandwidth / preprocessing.")
    else:
        print(f"  No signal in either direction. Best Δr={max(best_norm[0],best_inv[0]):.5f}")
        print("  Data may be misaligned or EEG may be missing the AAD frequency band.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
