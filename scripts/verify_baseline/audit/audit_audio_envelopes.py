"""
Phase 0D — Audio Envelope Audit
Verifies sampling rate, duration, number of bands, compression, and
temporal length matching between EEG trials and audio envelopes.

Usage (Kaggle):
    !cd ISEF_Project && python scripts/verify_baseline/audit/audit_audio_envelopes.py
"""

from __future__ import annotations
import sys
import pickle
import json
import csv
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import subject_files, load_subject_examples
from analysis._common import load_subject_data, fsample_values

OUTPUT_DIR = Path("audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)

EEG_FS = 64  # Expected downsampled EEG sampling rate

ENVELOPE_CANDIDATES = [
    Path("/kaggle/input/datasets/lokeshgile/new-dtu-gammatones/gammatone_envelopes (1).pkl"),
    Path("/kaggle/input").rglob("*gammatone*.pkl") if Path("/kaggle/input").exists() else iter([]),
]

MAPPING_CANDIDATES = [
    Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json"),
    REPO_ROOT / "scripts" / "verify_baseline" / "data" / "audio_mapping.json",
]


def find_envelope_file():
    # Direct candidates
    direct = Path("/kaggle/input/datasets/lokeshgile/new-dtu-gammatones/gammatone_envelopes (1).pkl")
    if direct.exists():
        return direct
    # Search
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob("*gammatone*.pkl"))
        if candidates:
            return candidates[0]
        candidates = list(Path("/kaggle/input").rglob("*.pkl"))
        if candidates:
            return candidates[0]
    return None


def load_mapping():
    for p in MAPPING_CANDIDATES:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def audit_audio_envelopes():
    print(f"\n{'='*70}")
    print("PHASE 0D — AUDIO ENVELOPE AUDIT")
    print(f"{'='*70}")

    env_path = find_envelope_file()
    if env_path is None:
        print("ERROR: Cannot find gammatone envelope .pkl file.")
        print("  Expected location: /kaggle/input/...gammatone...pkl")
        return

    print(f"  Envelope file: {env_path}")
    with open(env_path, "rb") as f:
        envelopes = pickle.load(f)

    # Inspect envelope structure
    keys = list(envelopes.keys())
    print(f"  Total entries in envelope dict: {len(keys)}")
    print(f"  Example keys (first 3): {keys[:3]}")

    # Inspect a few entries
    example_key = keys[0]
    example_env = envelopes[example_key]
    print(f"\n  Example entry: '{example_key}'")
    print(f"    Shape:  {example_env.shape}  (expected: [n_bands, n_samples])")
    print(f"    dtype:  {example_env.dtype}")
    print(f"    mean:   {example_env.mean():.4f}")
    print(f"    std:    {example_env.std():.4f}")
    print(f"    min:    {example_env.min():.4f}")
    print(f"    max:    {example_env.max():.4f}")

    if example_env.ndim == 2:
        n_bands, n_samples = example_env.shape
        duration_s = n_samples / EEG_FS
        print(f"    n_bands:     {n_bands}  (expect 28)")
        print(f"    n_samples:   {n_samples}")
        print(f"    duration:    {duration_s:.1f}s @ {EEG_FS} Hz")
    else:
        print(f"  WARNING: Unexpected shape ndim={example_env.ndim}")

    # Check power-law compression heuristic
    # If compression was applied (e.g., x^0.3), values should be in a compressed range
    raw_range = example_env.max() - example_env.min()
    print(f"\n  Value range: {raw_range:.4f}")
    if example_env.min() >= 0:
        print(f"  Values are non-negative (consistent with envelope after abs() or Hilbert)")
    else:
        print(f"  Values have negative entries — normalization/subtraction was applied")

    # Now cross-check envelope duration against EEG trial duration
    mapping = load_mapping()
    paths = subject_files()

    if not mapping or not paths:
        print("\n  WARNING: Cannot perform EEG-audio duration matching (missing mapping or subject files).")
        return

    print(f"\n--- EEG vs Audio Duration Matching ---")
    mismatch_count = 0
    match_count = 0
    truncation_sizes = []
    rows = []

    for path in paths[:4]:  # Pilot: first 4 subjects
        subject_id = path.stem
        subj_key = subject_id.split("_")[0]
        data = load_subject_data(path)
        fs_vals = fsample_values(data)
        fs_eeg_raw = fs_vals["eeg"]  # Raw fs (likely 512 Hz)
        n_trials = data.eeg.shape[1]

        examples = load_subject_examples(path)

        for trial_idx, ex in enumerate(examples):
            eeg_raw = ex.eeg  # Shape: [n_samples, n_channels] from load_subject_examples
            n_samples_raw = eeg_raw.shape[0]  # time axis is axis=0
            eeg_duration_at_64hz = int(n_samples_raw * EEG_FS / fs_eeg_raw)

            trial_key = f"trial_{trial_idx}"
            if subj_key not in mapping or trial_key not in mapping[subj_key]:
                continue

            fname_a = mapping[subj_key][trial_key].get("wavA", {}).get("filename", None)
            if fname_a is None or fname_a not in envelopes:
                continue

            env = envelopes[fname_a]
            audio_n_samples = env.shape[1] if env.ndim == 2 else env.shape[0]

            diff = abs(eeg_duration_at_64hz - audio_n_samples)
            if diff > 1:
                mismatch_count += 1
                truncation_sizes.append(diff)
            else:
                match_count += 1

            rows.append({
                "subject": subject_id,
                "trial_idx": trial_idx,
                "eeg_raw_samples": n_samples_raw,
                "fs_eeg_raw": fs_eeg_raw,
                "eeg_samples_at_64hz": eeg_duration_at_64hz,
                "audio_samples": audio_n_samples,
                "diff_samples": diff,
                "status": "MATCH" if diff <= 1 else f"MISMATCH({diff})",
            })

    if rows:
        csv_path = OUTPUT_DIR / "audio_duration_audit.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Duration matches:    {match_count}")
        print(f"  Duration mismatches: {mismatch_count}")
        if truncation_sizes:
            print(f"  Mismatch sizes:      min={min(truncation_sizes)} max={max(truncation_sizes)} mean={np.mean(truncation_sizes):.1f} samples")
        print(f"  -> Duration audit CSV saved to: {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    all_ok = (mismatch_count == 0) if rows else None
    verdict = "PASS" if all_ok else ("FAIL" if all_ok is False else "PARTIAL (pilot only)")
    print(f"PHASE 0D VERDICT: {verdict}")
    if mismatch_count > 0:
        print(f"  {mismatch_count} trials have EEG/audio duration mismatch > 1 sample.")
        print(f"  Our current code uses min_len truncation which silently discards this difference.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    audit_audio_envelopes()
