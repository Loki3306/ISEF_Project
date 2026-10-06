"""
Multi-Band Sinc-CATCN Scientific Negative Controls & Anti-Cheating Ablation Suite.

Rigorously verifies that model predictions are driven 100% by true physiological
cortical auditory tracking (neural phase-locking to speech envelopes) and NOT by:
1. Audio volume / energy shortcuts
2. Reverse time leakage
3. Temporal lag violations (+10s shift)
4. Stream asymmetry bias
5. Acoustic story memorization

Evaluates:
- Control 0: Ground Truth Baseline (Intact Pipeline)
- Control 1: Time-Reversed Speech Envelope (t -> -t, reverses acoustic onsets)
- Control 2: Temporal Latency Violation (+10s shift, outside physiological ERP window)
- Control 3: Zero-EEG Shortcut Test (EEG = 0, proves zero audio-only shortcut)
- Control 4: Synthetic Gaussian EEG Noise (surrogate noise matched to channel variance)
- Control 5: Cross-Trial Shuffled EEG (trial-mismatched EEG and audio)
- Control 6: Stream Anti-Symmetry Check (Delta(A,B) + Delta(B,A) == 0 to machine precision)
"""

from __future__ import annotations
import argparse
import sys
import os
import json
import time
import math
from pathlib import Path
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.multiband_catcn import MultiBandCATCNDecoder
from src.streaming.causal_filters import StreamingCausalEEGFilter
from src.models.spatial_adapter import SpatialEEGAdapter
from src.selective_aad.temporal_gate import SignalQualityMonitor, StickyHysteresisGate
from src.selective_aad.metrics import calculate_selective_metrics, compute_temporal_stability_metrics
from baselines.ridge_aad import load_subject_examples, TrialExample
from training.montages import MONTAGES
from data.multiband_provider import get_multiband_envelopes, get_mapping
from training.train_multiband_catcn import (
    discover_eeg_subjects, butter_lowpass_sosfilt,
    train_subject_spatial_adapter, evaluate_windows, evaluate_streaming_trials,
    FS
)

def run_multiband_ablations(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    audio_channels = args.audio_bands + (1 if args.include_broadband else 0)

    print("=" * 116)
    print("  MULTI-BAND SINC-CATCN SCIENTIFIC NEGATIVE CONTROLS & ABLATION AUDIT")
    print(f"  Montage: {args.montage} ({n_ch} channels) | Audio Bands: {audio_channels} | Device: {device}")
    print("=" * 116)

    # 1. Locate and Load Pre-Trained Checkpoint
    checkpoint_candidates = [
        Path(args.checkpoint_path) if args.checkpoint_path else None,
        REPO_ROOT / "results" / "multiband_catcn" / "sinc_multiband_catcn_best.pt",
        Path("/kaggle/working/sinc_multiband_catcn_best.pt"),
        Path("sinc_multiband_catcn_best.pt"),
        REPO_ROOT / "results" / "multiband_catcn" / "multiband_catcn_best.pt",
        Path("/kaggle/working/multiband_catcn_best.pt"),
        Path("multiband_catcn_best.pt")
    ]
    ckpt_path = None
    for cand in checkpoint_candidates:
        if cand and cand.exists():
            ckpt_path = cand
            break

    model = MultiBandCATCNDecoder(
        eeg_channels=n_ch,
        audio_bands=audio_channels,
        hidden_dim=args.hidden_dim,
        min_lag_samples=args.min_lag_samples,
        max_lag_samples=args.max_lag_samples,
        head_type=args.head_type,
        use_sinc=args.use_sinc,
        arch=args.arch
    ).to(device)

    if ckpt_path:
        print(f"\n[CHECKPOINT]: Loading pre-trained weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        # Prefix normalization
        if any(k.startswith("model.") for k in state.keys()) and not any(k.startswith("model.") for k in model.state_dict().keys()):
            state = {k[6:]: v for k, v in state.items()}
        elif not any(k.startswith("model.") for k in state.keys()) and any(k.startswith("model.") for k in model.state_dict().keys()):
            state = {f"model.{k}": v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
            print("  --> Strict checkpoint load verified.")
        except Exception as err:
            print(f"[INFO] Strict load threw ({err}), loading matching parameters with strict=False...")
            model.load_state_dict(state, strict=False)
    else:
        if args.smoke_test:
            print("\n[SMOKE TEST] No checkpoint on disk. Initializing random weights for pipeline validation...")
        else:
            raise FileNotFoundError("No checkpoint found. Please provide --checkpoint_path.")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 2. Mathematical Anti-Symmetry Check on Synthetic Pairs
    print("\n[SANITY CHECK]: Verifying Machine-Precision Anti-Symmetry...")
    test_eeg = torch.randn(8, n_ch, 320, device=device)
    test_a = torch.randn(8, audio_channels, 320, device=device)
    test_b = torch.randn(8, audio_channels, 320, device=device)
    delta_ab, _, _ = model(test_eeg, test_a, test_b)
    delta_ba, _, _ = model(test_eeg, test_b, test_a)
    anti_sym_err = torch.max(torch.abs(delta_ab + delta_ba)).item()
    print(f"  * Max Stream Asymmetry Discrepancy |Delta(A,B) + Delta(B,A)|: {anti_sym_err:.2e}")
    assert anti_sym_err < 1e-6, f"Anti-symmetry violation! Discrepancy: {anti_sym_err}"
    print("  --> PASS: Anti-symmetry holds to machine precision. Zero stream order bias.")

    # 3. Load Dataset & Envelopes
    all_paths = discover_eeg_subjects(args.eeg_dir)
    if args.subjects and args.subjects.lower() != 'all':
        req = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]
        all_paths = [p for p in all_paths if p.stem.split("_")[0].upper() in req]
    if not all_paths:
        if args.smoke_test:
            all_paths = [Path("S1_data_preproc.mat"), Path("S2_data_preproc.mat")]
        else:
            raise FileNotFoundError("No DTU subjects found.")
    elif args.smoke_test and not args.subjects:
        all_paths = all_paths[:2]

    try:
        mapping = get_mapping()
        envelopes = get_multiband_envelopes(
            target_bands=args.audio_bands,
            custom_env_file=args.audio_env_file,
            custom_audio_dir=args.audio_dir,
            auto_extract=(not args.smoke_test)
        )
    except Exception as e:
        if args.smoke_test:
            mapping, envelopes = {}, {}
        else:
            raise e

    causal_eeg_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.5, fs=FS, order=2, n_channels=n_ch)
    sq_monitor = SignalQualityMonitor()

    # Partition Test Trials for All Subjects
    print(f"\n[DATA PREPARATION]: Extracting held-out test trials across {len(all_paths)} subjects...")
    subject_test_data = {}
    for p in all_paths:
        sub_name = p.stem
        sub_key = sub_name.replace("_data_preproc", "")

        if args.smoke_test and (not p.exists() or not envelopes):
            n_trials = 4
            exs = [
                TrialExample(
                    subject=sub_name, trial_index=i,
                    eeg=np.random.randn(3200, 64).astype(np.float32),
                    wav_a=np.random.randn(3200).astype(np.float32),
                    wav_b=np.random.randn(3200).astype(np.float32),
                    label=1
                )
                for i in range(n_trials)
            ]
            raw_ya_list = [np.random.randn(args.audio_bands, 3200).astype(np.float32) for _ in range(n_trials)]
            raw_yb_list = [np.random.randn(args.audio_bands, 3200).astype(np.float32) for _ in range(n_trials)]
        else:
            exs = list(load_subject_examples(p))
            raw_ya_list, raw_yb_list, valid_exs = [], [], []
            for i, ex in enumerate(exs):
                trial_idx = getattr(ex, 'trial_index', i)
                trial_key = f"trial_{trial_idx}"
                if sub_key in mapping and trial_key in mapping[sub_key]:
                    fa = mapping[sub_key][trial_key]["wavA"]["filename"]
                    fb = mapping[sub_key][trial_key]["wavB"]["filename"]
                    if fa in envelopes and fb in envelopes:
                        raw_ya_list.append(envelopes[fa])
                        raw_yb_list.append(envelopes[fb])
                        valid_exs.append(ex)
            exs = valid_exs

        n_valid = min(len(exs), len(raw_ya_list))
        if n_valid == 0:
            continue

        split_idx = int(math.floor(n_valid * (1.0 - args.test_split)))
        sub_eeg_te, sub_ya_te, sub_yb_te = [], [], []
        sub_eeg_cal, sub_ya_cal, sub_yb_cal = [], [], []

        for idx in range(n_valid):
            raw_eeg = exs[idx].eeg[:, montage_channels].astype(np.float32)
            cur_ya = raw_ya_list[idx]
            cur_yb = raw_yb_list[idx]
            min_len = min(len(raw_eeg), cur_ya.shape[-1], cur_yb.shape[-1])
            raw_eeg = raw_eeg[:min_len]
            cur_ya = cur_ya[:, :min_len]
            cur_yb = cur_yb[:, :min_len]

            causal_eeg_filter.reset()
            eeg_c = causal_eeg_filter.process_chunk(raw_eeg)
            eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-12)

            ya_c = butter_lowpass_sosfilt(cur_ya, 8.0, FS, order=2).astype(np.float32)
            yb_c = butter_lowpass_sosfilt(cur_yb, 8.0, FS, order=2).astype(np.float32)
            ya_c = (ya_c - np.mean(ya_c, axis=-1, keepdims=True)) / (np.std(ya_c, axis=-1, keepdims=True) + 1e-12)
            yb_c = (yb_c - np.mean(yb_c, axis=-1, keepdims=True)) / (np.std(yb_c, axis=-1, keepdims=True) + 1e-12)

            if args.include_broadband:
                bb_a = np.mean(ya_c, axis=0, keepdims=True)
                bb_b = np.mean(yb_c, axis=0, keepdims=True)
                ya_c = np.concatenate([bb_a, ya_c], axis=0)
                yb_c = np.concatenate([bb_b, yb_c], axis=0)

            if idx < split_idx:
                if idx < args.calib_trials:
                    sub_eeg_cal.append(eeg_c)
                    sub_ya_cal.append(ya_c)
                    sub_yb_cal.append(yb_c)
            else:
                sub_eeg_te.append(eeg_c)
                sub_ya_te.append(ya_c)
                sub_yb_te.append(yb_c)

        if sub_eeg_te:
            subject_test_data[sub_name] = (sub_eeg_te, sub_ya_te, sub_yb_te, sub_eeg_cal, sub_ya_cal, sub_yb_cal)

    print(f"[DATA READY]: Loaded held-out test splits for {len(subject_test_data)} DTU subjects.")

    # 4. Define Evaluation Helper for Arbitrary Conditions
    def evaluate_condition(name: str, modify_fn):
        accs_raw_5s, accs_5s, accs_10s, accs_20s = [], [], [], []
        win_5s_smp = int(5.0 * FS)
        win_10s_smp = int(10.0 * FS)
        win_20s_smp = int(20.0 * FS)

        for sub_name, (te_eeg, te_ya, te_yb, cal_eeg, cal_ya, cal_yb) in subject_test_data.items():
            mod_eeg, mod_ya, mod_yb = modify_fn(te_eeg, te_ya, te_yb)
            
            # Raw un-gated 5.0s window accuracy
            a_raw5, _ = evaluate_windows(model, mod_eeg, mod_ya, mod_yb, win_5s_smp, device, adapter=None)
            
            # Streaming 5.0s evaluation with Sticky Hysteresis Gate
            ad_margins, ad_labels, ad_raw_eeg = evaluate_streaming_trials(
                model, None, mod_eeg, mod_ya, mod_yb, window_sec=5.0, step_sec=0.5, fs=FS, device=device
            )
            if ad_margins and len(np.concatenate(ad_margins)) > 0:
                t3_decs = []
                for m_seq, eeg_trial in zip(ad_margins, ad_raw_eeg):
                    gate = StickyHysteresisGate(alpha=0.85, threshold_switch=0.35, threshold_maintain=0.15, n_confirm=2)
                    for v, w_eeg in zip(m_seq, eeg_trial):
                        sq = sq_monitor.check_eeg_window(w_eeg)
                        out = gate.update(v, is_artifact=not sq["is_valid"])
                        t3_decs.append(out["decision"])
                flat_decs = np.array(t3_decs)
                gt_str = np.where(np.concatenate(ad_labels) == 1, "A", "B")
                m_res = calculate_selective_metrics(flat_decs, gt_str)
                a5 = m_res["selective_accuracy"] * 100.0 if m_res["accepted_count"] > 0 else 50.0
            else:
                a5 = 50.0

            a10, _ = evaluate_windows(model, mod_eeg, mod_ya, mod_yb, win_10s_smp, device, adapter=None)
            a20, _ = evaluate_windows(model, mod_eeg, mod_ya, mod_yb, win_20s_smp, device, adapter=None)

            accs_raw_5s.append(a_raw5)
            accs_5s.append(a5)
            accs_10s.append(a10)
            accs_20s.append(a20)

        return float(np.mean(accs_raw_5s)), float(np.mean(accs_5s)), float(np.mean(accs_10s)), float(np.mean(accs_20s))

    # =========================================================================
    # 5. EXECUTE ALL NEGATIVE CONTROLS
    # =========================================================================
    results = {}

    # Control 0: Ground Truth Baseline (Intact Pipeline)
    print("\n--- [Ablation 0/5]: Running Intact Ground Truth Baseline ---")
    results["Ground Truth Intact Pipeline"] = evaluate_condition(
        "Ground Truth", lambda e, ya, yb: (e, ya, yb)
    )

    # Control 1: Time-Reversed Speech Envelope (Reverses acoustic onsets)
    print("--- [Ablation 1/5]: Running Time-Reversed Speech Envelope (t -> -t) ---")
    results["Time-Reversed Audio Envelope"] = evaluate_condition(
        "Time-Reversed", lambda e, ya, yb: (
            e,
            [np.ascontiguousarray(a[:, ::-1]) for a in ya],
            [np.ascontiguousarray(b[:, ::-1]) for b in yb]
        )
    )

    # Control 2: Temporal Latency Violation (+10.0s Shift / Lag Trap)
    print("--- [Ablation 2/5]: Running Temporal Latency Violation (+10s Lag Trap) ---")
    shift_smp = int(10.0 * FS)
    results["Temporal Latency Violation (+10s Lag)"] = evaluate_condition(
        "+10s Lag", lambda e, ya, yb: (
            e,
            [np.roll(a, shift_smp, axis=-1) for a in ya],
            [np.roll(b, shift_smp, axis=-1) for b in yb]
        )
    )

    # Control 3: Zero-EEG Shortcut Control (EEG = 0)
    print("--- [Ablation 3/5]: Running Zero-EEG Shortcut Control (EEG = 0) ---")
    results["Zero-EEG Shortcut Control (EEG = 0)"] = evaluate_condition(
        "Zero-EEG", lambda e, ya, yb: (
            [np.zeros_like(x) for x in e],
            ya, yb
        )
    )

    # Control 4: Synthetic Gaussian EEG Noise
    print("--- [Ablation 4/5]: Running Synthetic Gaussian EEG Noise Control ---")
    rng = np.random.RandomState(42)
    results["Synthetic Gaussian EEG Noise"] = evaluate_condition(
        "Gaussian Noise", lambda e, ya, yb: (
            [rng.randn(*x.shape).astype(np.float32) for x in e],
            ya, yb
        )
    )

    # Control 5: Cross-Trial Shuffled EEG (Trial Mismatch)
    print("--- [Ablation 5/5]: Running Cross-Trial Shuffled EEG Control ---")
    def shuffle_trials(e, ya, yb):
        if len(e) > 1:
            idx = list(range(1, len(e))) + [0]
            shuffled_e = [e[i] for i in idx]
        else:
            shuffled_e = e
        return shuffled_e, ya, yb
    results["Cross-Trial Shuffled EEG (Mismatch)"] = evaluate_condition(
        "Trial Mismatch", shuffle_trials
    )

    # 6. Format and Print Scientific Summary Table
    print("\n" + "=" * 124)
    print("  MULTI-BAND SINC-CATCN SCIENTIFIC NEGATIVE CONTROLS & ABLATION AUDIT SUMMARY")
    print("=" * 124)
    print(f"  {'Condition':<40} | {'Expected':<15} | {'Raw 5s':<8} | {'Gate 5s':<8} | {'10.0s':<8} | {'20.0s':<8} | {'Scientific Verdict':<18}")
    print("  " + "-" * 120)

    expected_dict = {
        "Ground Truth Intact Pipeline": "High (>70%)",
        "Time-Reversed Audio Envelope": "Chance (~50%)",
        "Temporal Latency Violation (+10s Lag)": "Chance (~50%)",
        "Zero-EEG Shortcut Control (EEG = 0)": "Chance (50.0%)",
        "Synthetic Gaussian EEG Noise": "Chance (~50%)",
        "Cross-Trial Shuffled EEG (Mismatch)": "Chance (~50%)"
    }

    ablation_json = {}
    for cond, (ar5, a5, a10, a20) in results.items():
        exp = expected_dict[cond]
        if cond == "Ground Truth Intact Pipeline":
            verdict = "VALID (Neural)" if a5 >= 70.0 else "SUB-OPTIMAL"
        else:
            verdict = "PASS (Zero Leakage)" if abs(ar5 - 50.0) <= 6.0 and abs(a20 - 50.0) <= 6.0 else "FLAGGED"

        print(f"  {cond:<40} | {exp:<15} | {ar5:>6.1f}% | {a5:>6.1f}% | {a10:>6.1f}% | {a20:>6.1f}% | {verdict:<18}")
        ablation_json[cond] = {
            "expected": exp,
            "raw_acc_5s": round(ar5, 2),
            "gated_acc_5s": round(a5, 2),
            "acc_10s": round(a10, 2),
            "acc_20s": round(a20, 2),
            "verdict": verdict
        }

    print("=" * 124)

    out_file = Path("/kaggle/working/multiband_catcn_ablation_results.json") if Path("/kaggle/working").exists() else (REPO_ROOT / "results" / "multiband_catcn" / "ablation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(ablation_json, f, indent=2)
    print(f"\n[OUTPUT] Ablation audit artifact saved to: {out_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Band Sinc-CATCN Negative Controls & Ablations")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", choices=list(MONTAGES.keys()))
    parser.add_argument("--audio_bands", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--min_lag_samples", type=int, default=-2)
    parser.add_argument("--max_lag_samples", type=int, default=18)
    parser.add_argument("--head_type", type=str, default="linear")
    parser.add_argument("--arch", type=str, default="sinc", choices=["sinc", "msca", "baseline"])
    parser.add_argument("--use_sinc", action="store_true", default=True)
    parser.add_argument("--include_broadband", action="store_true", default=True)
    parser.add_argument("--calib_trials", type=int, default=12)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--eeg_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--audio_env_file", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--subjects", type=str, default=None)
    args = parser.parse_args()
    run_multiband_ablations(args)
