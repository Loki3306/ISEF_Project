"""
Phase 0C — Channel Structure Verification
Identifies all channels in the DTU data, checks sampling rates,
and documents the reference scheme without modifying the pipeline.

Usage (Kaggle):
    !cd ISEF_Project && python scripts/verify_baseline/audit/audit_channels.py
"""

from __future__ import annotations
import sys
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import subject_files
from analysis._common import load_subject_data, channel_labels, fsample_values

OUTPUT_DIR = Path("audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def audit_channels():
    paths = subject_files()

    print(f"\n{'='*70}")
    print("PHASE 0C — CHANNEL STRUCTURE VERIFICATION")
    print(f"{'='*70}")

    rows = []
    all_ch_counts = set()
    all_ch_names = {}  # subject -> list of channel names
    fs_summary = []

    for path in paths:
        subject_id = path.stem
        data = load_subject_data(path)

        # EEG shape: [1, n_trials] with each trial being [n_samples, n_channels]
        n_trials = data.eeg.shape[1]
        eeg_trial0 = data.eeg[0, 0]
        n_samples, n_channels = eeg_trial0.shape

        # Sample rates
        fs_vals = fsample_values(data)
        fs_eeg = fs_vals["eeg"]

        # Channel names
        try:
            ch_names = channel_labels(data)
        except Exception as e:
            ch_names = [f"ch_{i}" for i in range(n_channels)]
            print(f"  WARNING: Could not read channel labels for {subject_id}: {e}")

        all_ch_counts.add(n_channels)
        all_ch_names[subject_id] = ch_names
        fs_summary.append(fs_eeg)

        rows.append({
            "subject": subject_id,
            "n_channels": n_channels,
            "n_trials": n_trials,
            "n_samples_trial0": n_samples,
            "duration_s_trial0": n_samples / fs_eeg,
            "fs_eeg": fs_eeg,
            "fs_wavA": fs_vals["wavA"],
            "fs_wavB": fs_vals["wavB"],
            "ch_first_5": " | ".join(ch_names[:5]) if ch_names else "N/A",
            "ch_last_5": " | ".join(ch_names[-5:]) if len(ch_names) >= 5 else "N/A",
        })

        print(f"\nSubject: {subject_id}")
        print(f"  Channels: {n_channels} | Trials: {n_trials} | fs_eeg: {fs_eeg} Hz | fs_wavA: {fs_vals['wavA']} Hz")
        print(f"  Trial duration (trial 0): {n_samples} samples = {n_samples/fs_eeg:.1f}s")
        if ch_names:
            print(f"  First 5 channels:  {ch_names[:5]}")
            print(f"  Last 5 channels:   {ch_names[-5:]}")

    # Save CSV
    csv_path = OUTPUT_DIR / "channel_audit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Cross-subject consistency check
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Unique channel counts across subjects: {sorted(all_ch_counts)}")
    print(f"  Unique EEG sampling rates: {sorted(set(fs_summary))}")

    if len(all_ch_counts) == 1:
        n_ch = list(all_ch_counts)[0]
        print(f"\n  All subjects have {n_ch} channels — CONSISTENT")
        if n_ch == 66:
            print(f"\n  FINDING: 66 channels = 64 scalp EEG + 2 mastoid electrodes")
            print(f"  The DTU dataset includes mastoid reference electrodes as data channels.")
            print(f"  ACTION REQUIRED (Phase 2.5): Determine correct re-referencing strategy.")
            print(f"  Do NOT assume mastoid channels must be dropped. Check what each paper does.")
        elif n_ch == 64:
            print(f"  FINDING: 64 channels — likely already referenced (mastoids removed).")
    else:
        print(f"\n  WARNING: Inconsistent channel count across subjects!")

    # Check if channel names are consistent
    first_subj = list(all_ch_names.keys())[0]
    reference_names = all_ch_names[first_subj]
    for subj, names in all_ch_names.items():
        if names != reference_names:
            print(f"  WARNING: Channel names differ between {first_subj} and {subj}")
            break
    else:
        print(f"  Channel ordering is consistent across all subjects — CONSISTENT")

    # Mastoid identification
    print(f"\n--- Mastoid / Reference Channel Analysis ---")
    reference_ch_names = all_ch_names.get(first_subj, [])
    mastoid_candidates = [
        (i, name) for i, name in enumerate(reference_ch_names)
        if any(kw in name.upper() for kw in ["MAST", "M1", "M2", "A1", "A2", "TP9", "TP10", "REF"])
    ]
    if mastoid_candidates:
        print(f"  Mastoid/reference channel candidates:")
        for idx, name in mastoid_candidates:
            print(f"    Index {idx}: '{name}'")
    else:
        print(f"  No obvious mastoid/reference channels identified by name.")
        print(f"  Last 2 channel names: {reference_ch_names[-2:] if len(reference_ch_names) >= 2 else reference_ch_names}")

    print(f"\n  -> Channel audit CSV saved to: {csv_path}")
    print(f"\n{'='*70}")
    consistent = len(all_ch_counts) == 1
    print(f"PHASE 0C VERDICT: {'PASS' if consistent else 'FAIL — inconsistent channel count'}")
    print(f"  NOTE: Mastoid handling decision deferred to Phase 2.5 (after literature review)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    audit_channels()
