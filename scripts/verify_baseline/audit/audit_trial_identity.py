"""
Phase 0A — Trial Identity Verification
Produces a fully traceable record of every trial: subject, indices,
audio file references, label, and event sample positions.

Usage (Kaggle):
    !cd ISEF_Project && python scripts/verify_baseline/audit/audit_trial_identity.py
"""

from __future__ import annotations
import sys
import os
import json
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import subject_files, load_subject_examples
from analysis._common import load_subject_data, trial_labels, trial_event_samples, fsample_values

OUTPUT_DIR = Path("audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MAPPING_CANDIDATES = [
    Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json"),
    REPO_ROOT / "scripts" / "verify_baseline" / "data" / "audio_mapping.json",
]


def load_mapping() -> dict | None:
    for p in MAPPING_CANDIDATES:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    print("WARNING: audio_mapping.json not found. Audio filename tracing will be skipped.")
    return None


def audit_trial_identity():
    paths = subject_files()
    if not paths:
        print("ERROR: No subject files found. Check DATA_DIR.")
        return

    mapping = load_mapping()

    rows = []
    errors = []

    print(f"\n{'='*70}")
    print("PHASE 0A — TRIAL IDENTITY VERIFICATION")
    print(f"{'='*70}")
    print(f"Subjects found: {len(paths)}\n")

    seen_audio_pairs: dict[tuple, list[str]] = {}  # (fileA, fileB) -> list of "subj/trial"

    for path in paths:
        subject_id = path.stem  # e.g. "S5_data_preproc"
        subj_key = subject_id.split("_")[0]  # e.g. "S5"

        data = load_subject_data(path)
        labels = trial_labels(data)
        event_samples = trial_event_samples(data) if hasattr(data, 'event') else []
        fs_vals = fsample_values(data)
        n_trials = data.eeg.shape[1]

        print(f"Subject: {subject_id} | Trials: {n_trials} | fs_eeg={fs_vals['eeg']} | fs_wavA={fs_vals['wavA']} | fs_wavB={fs_vals['wavB']}")

        for trial_idx in range(n_trials):
            label = labels[trial_idx]
            event_sample = event_samples[trial_idx] if trial_idx < len(event_samples) else "N/A"

            # Audio file mapping
            audio_a_file = "MISSING"
            audio_b_file = "MISSING"
            mapping_found = False

            if mapping and subj_key in mapping:
                trial_key = f"trial_{trial_idx}"
                if trial_key in mapping[subj_key]:
                    audio_a_file = mapping[subj_key][trial_key].get("wavA", {}).get("filename", "MISSING")
                    audio_b_file = mapping[subj_key][trial_key].get("wavB", {}).get("filename", "MISSING")
                    mapping_found = True
                else:
                    errors.append(f"ERROR: {subject_id}/trial_{trial_idx} missing from audio_mapping.json")
            elif mapping:
                errors.append(f"ERROR: Subject key '{subj_key}' not in audio_mapping.json")

            # Track audio pair reuse
            pair_key = (audio_a_file, audio_b_file)
            tag = f"{subject_id}/trial_{trial_idx}"
            if pair_key not in seen_audio_pairs:
                seen_audio_pairs[pair_key] = []
            seen_audio_pairs[pair_key].append(tag)

            # Attended stream from label
            if label == 1:
                attended = "A"
            elif label == 2:
                attended = "B"
            else:
                attended = f"UNKNOWN({label})"
                errors.append(f"ERROR: {subject_id}/trial_{trial_idx} has unexpected label value: {label}")

            row = {
                "subject": subject_id,
                "trial_idx": trial_idx,
                "label_raw": label,
                "attended_stream": attended,
                "audio_A_file": audio_a_file,
                "audio_B_file": audio_b_file,
                "eeg_event_sample": event_sample,
                "mapping_found": mapping_found,
            }
            rows.append(row)

    # Write CSV
    csv_path = OUTPUT_DIR / "trial_identity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  -> Trial identity saved to: {csv_path}")

    # Check for repeated audio pairs ACROSS subjects (could indicate shared stimuli or mapping bug)
    repeated_pairs = {k: v for k, v in seen_audio_pairs.items() if len(v) > 1}
    cross_subject_repeats = {
        k: v for k, v in repeated_pairs.items()
        if len(set(tag.split("/")[0] for tag in v)) > 1
    }

    print(f"\n--- Label Distribution ---")
    label_counts: dict[str, int] = {}
    for row in rows:
        key = f"attended_{row['attended_stream']}"
        label_counts[key] = label_counts.get(key, 0) + 1
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v} trials ({100*v/len(rows):.1f}%)")

    print(f"\n--- Audio File Reuse ---")
    print(f"  Unique audio pairs:     {len(seen_audio_pairs)}")
    print(f"  Reused pairs (any):     {len(repeated_pairs)}")
    print(f"  Cross-subject repeats:  {len(cross_subject_repeats)}")
    if cross_subject_repeats:
        print("  NOTE: Same audio pair used for multiple subjects (may be expected if stimuli are shared):")
        for k, v in list(cross_subject_repeats.items())[:5]:
            print(f"    {k[0][:30]}... used in: {v[:3]}")

    print(f"\n--- Errors & Warnings ---")
    if errors:
        for e in errors:
            print(f"  {e}")
        # Write errors
        err_path = OUTPUT_DIR / "trial_identity_errors.txt"
        err_path.write_text("\n".join(errors))
        print(f"  -> {len(errors)} errors saved to {err_path}")
    else:
        print("  PASS: No label mapping errors found.")

    print(f"\n{'='*70}")
    verdict = "PASS" if not errors else f"FAIL ({len(errors)} errors)"
    print(f"PHASE 0A VERDICT: {verdict}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    audit_trial_identity()
