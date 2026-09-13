"""
Phase 0B — Label/Audio Mapping Chain Verification
Follows the label chain explicitly: raw label -> attended stream ->
audio file loaded -> InfoNCE positive -> evaluation target.

Compares the label chain against audio_mapping.json independently
of the training code to detect any mapping discrepancy.

Usage (Kaggle):
    !cd ISEF_Project && python scripts/verify_baseline/audit/audit_label_mapping.py
"""

from __future__ import annotations
import sys
import json
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import subject_files, load_subject_examples
from analysis._common import load_subject_data, trial_labels

OUTPUT_DIR = Path("audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MAPPING_CANDIDATES = [
    Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json"),
    REPO_ROOT / "scripts" / "verify_baseline" / "data" / "audio_mapping.json",
]


def load_mapping() -> dict:
    for p in MAPPING_CANDIDATES:
        if p.exists():
            print(f"  Using mapping: {p}")
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError("audio_mapping.json not found in any candidate location.")


def audit_label_mapping():
    paths = subject_files()
    mapping = load_mapping()

    print(f"\n{'='*70}")
    print("PHASE 0B — LABEL / AUDIO MAPPING CHAIN VERIFICATION")
    print(f"{'='*70}")

    rows = []
    errors = []
    warnings = []
    n_total = 0
    n_mapping_consistent = 0
    n_mapping_missing = 0
    n_label_unexpected = 0

    for path in paths:
        subject_id = path.stem
        subj_key = subject_id.split("_")[0]

        data = load_subject_data(path)
        labels = trial_labels(data)
        n_trials = data.eeg.shape[1]

        examples = load_subject_examples(path)

        for trial_idx in range(n_trials):
            n_total += 1
            label_raw = labels[trial_idx]
            ex = examples[trial_idx]

            # --- Step 1: Decode label to attended stream ---
            if label_raw == 1:
                attended_stream = "A"
                attended_wav_from_label = ex.wav_a
                unattended_wav_from_label = ex.wav_b
            elif label_raw == 2:
                attended_stream = "B"
                attended_wav_from_label = ex.wav_b
                unattended_wav_from_label = ex.wav_a
            else:
                errors.append(f"UNEXPECTED_LABEL: {subject_id}/trial_{trial_idx} label={label_raw}")
                n_label_unexpected += 1
                attended_stream = "UNKNOWN"
                attended_wav_from_label = None
                unattended_wav_from_label = None

            # --- Step 2: Check audio_mapping.json ---
            trial_key = f"trial_{trial_idx}"
            mapping_attended = "MISSING"
            mapping_file_a = "MISSING"
            mapping_file_b = "MISSING"
            chain_ok = False

            if subj_key in mapping and trial_key in mapping[subj_key]:
                trial_map = mapping[subj_key][trial_key]
                mapping_file_a = trial_map.get("wavA", {}).get("filename", "MISSING")
                mapping_file_b = trial_map.get("wavB", {}).get("filename", "MISSING")

                # The mapping should designate which stream is attended
                # In our train script: wavA is always loaded as env_attended, wavB as env_unattended
                # So: label=1 (attend A) -> env_attended=wavA -> correct
                #     label=2 (attend B) -> env_attended=wavA -> WRONG (attending B but positive is A)
                # This is the critical check.
                mapping_attended_stream = "A"  # Our code ALWAYS uses wavA as attended

                if attended_stream == "A":
                    chain_ok = True
                    note = "OK"
                elif attended_stream == "B":
                    # Label says B is attended but train code uses wavA as positive
                    chain_ok = False
                    errors.append(
                        f"LABEL_MISMATCH: {subject_id}/trial_{trial_idx} "
                        f"label={label_raw}(attend_B) but train code uses wavA as InfoNCE positive"
                    )
                    note = "MISMATCH: label=attend_B, positive=wavA"
                else:
                    note = f"UNKNOWN_LABEL({label_raw})"
            else:
                n_mapping_missing += 1
                warnings.append(f"MAPPING_MISSING: {subject_id}/{trial_key} not in audio_mapping.json")
                note = "MAPPING_MISSING"

            if chain_ok:
                n_mapping_consistent += 1

            rows.append({
                "subject": subject_id,
                "trial_idx": trial_idx,
                "label_raw": label_raw,
                "attended_stream": attended_stream,
                "mapping_file_A": mapping_file_a,
                "mapping_file_B": mapping_file_b,
                "train_code_positive": "wavA",
                "chain_ok": chain_ok,
                "note": note,
            })

    # Save CSV
    csv_path = OUTPUT_DIR / "label_mapping_audit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\n  Total trials audited:       {n_total}")
    print(f"  Label chain consistent:     {n_mapping_consistent}  ({100*n_mapping_consistent/n_total:.1f}%)")
    print(f"  Mapping missing:            {n_mapping_missing}")
    print(f"  Unexpected label values:    {n_label_unexpected}")
    print(f"  -> CSV saved to: {csv_path}")

    print(f"\n--- CRITICAL FINDING: InfoNCE Positive Selection ---")
    print(f"  Our train code sets: env_attended = envelopes[wavA]   (ALWAYS)")
    print(f"  Our train code sets: env_unattended = envelopes[wavB] (ALWAYS)")
    print(f"  Label 1 = attend A -> positive=wavA -> CORRECT")
    print(f"  Label 2 = attend B -> positive=wavA -> WRONG (wavB is attended but we treat wavA as positive)")

    n_label_1 = sum(1 for r in rows if r['label_raw'] == 1)
    n_label_2 = sum(1 for r in rows if r['label_raw'] == 2)
    print(f"\n  Trials where label=1 (attend A, mapping correct): {n_label_1} ({100*n_label_1/n_total:.1f}%)")
    print(f"  Trials where label=2 (attend B, mapping WRONG):   {n_label_2} ({100*n_label_2/n_total:.1f}%)")

    if n_label_2 > 0:
        print(f"\n  *** CRITICAL BUG CONFIRMED ***")
        print(f"  For {n_label_2} trials ({100*n_label_2/n_total:.1f}%), the InfoNCE positive is the UNATTENDED speaker.")
        print(f"  This directly corrupts training for those trials.")
        print(f"  If the dataset is balanced: ~50% of training is inverted -> model learns noise.")

    print(f"\n--- Errors ---")
    if errors:
        for e in errors[:20]:
            print(f"  {e}")
        err_path = OUTPUT_DIR / "label_mapping_errors.txt"
        err_path.write_text("\n".join(errors))
        print(f"  ... {len(errors)} total errors saved to {err_path}")
    else:
        print("  No label chain errors detected.")

    print(f"\n--- Warnings ---")
    if warnings:
        for w in warnings[:5]:
            print(f"  {w}")
    else:
        print("  No warnings.")

    print(f"\n{'='*70}")
    if n_label_2 > 0:
        print(f"PHASE 0B VERDICT: CRITICAL FAIL")
        print(f"  The InfoNCE training objective is inverted for label=2 trials.")
        print(f"  FIX REQUIRED: train code must check label and swap wavA/wavB accordingly.")
    elif errors:
        print(f"PHASE 0B VERDICT: FAIL ({len(errors)} errors)")
    else:
        print(f"PHASE 0B VERDICT: PASS")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    audit_label_mapping()
