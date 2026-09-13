"""
Phase 0 Master Audit Runner
Runs all Phase 0 audit scripts in sequence and summarises findings.

Usage (Kaggle):
    !cd ISEF_Project && python scripts/verify_baseline/audit/run_all_audits.py
    !cd ISEF_Project && python scripts/verify_baseline/audit/run_all_audits.py --subjects S5_data_preproc S6_data_preproc
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

# Run each audit module
from audit_trial_identity import audit_trial_identity
from audit_label_mapping import audit_label_mapping
from audit_channels import audit_channels
from audit_audio_envelopes import audit_audio_envelopes
from audit_temporal_alignment import audit_temporal_alignment

OUTPUT_DIR = Path("audit_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_all(subjects=None):
    print("\n" + "#" * 70)
    print("# PHASE 0 — COMPLETE DATA FORENSICS AUDIT")
    print("#" * 70)

    print("\n[1/5] Phase 0A: Trial Identity")
    audit_trial_identity()

    print("\n[2/5] Phase 0B: Label/Audio Mapping Chain")
    audit_label_mapping()

    print("\n[3/5] Phase 0C: Channel Structure")
    audit_channels()

    print("\n[4/5] Phase 0D: Audio Envelope Structure")
    audit_audio_envelopes()

    print("\n[5/5] Phase 0E: Temporal Alignment (pilot subjects)")
    pilot = subjects or ["S5_data_preproc", "S6_data_preproc"]
    audit_temporal_alignment(pilot)

    print("\n" + "#" * 70)
    print("# PHASE 0 COMPLETE")
    print("# Review audit_results/ for all CSV outputs and error logs.")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Run all forensic audits")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subjects for temporal alignment pilot")
    args = parser.parse_args()
    run_all(args.subjects)
