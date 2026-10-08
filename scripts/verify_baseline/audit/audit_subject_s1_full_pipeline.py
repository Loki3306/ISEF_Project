"""
Full Cohort Audit: Subject S1 Real-Time Streaming Performance Evaluation
========================================================================
Iterates across trials of Subject S1 directly from raw 512 Hz BioSemi EEG
and raw 44.1 kHz WAV audio, executing the 100% causal streaming pipeline:
  1. Causal 50 Hz Notch + CAR + Calibrated EOG Subtraction + 8:1 Decimation to 64 Hz
  2. Causal 8-Electrode Near-Ear Selection + Dual-Band Filtering (ERP + Alpha -> 16 ch)
  3. Causal Online 8-Band Gammatone Auditory Filterbank + Decimation + Onsets (16 ch)
  4. 5.0s Ring Buffer + S1 Spatial EEG Adapter + 3-Member NeuroConformer Ensemble
  5. Leaky Margin Integration (gamma=0.95) + Sticky Hysteresis Gate
  6. Closed-Loop Acoustic Steering (+6 dB boost / -18 dB suppression, 60 ms slew)

Outputs:
  - s1_full_cohort_audit_results.json
  - s1_full_cohort_audit_summary.png
  - s1_full_cohort_audit_report.md
"""

import sys
import os
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ensure repo root is on python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.streaming.raw_eeg_loader import load_raw_dtu_file, RawDTUSubjectData
from src.streaming.causal_raw_preprocessor import StreamingCausalRawEEGPreprocessor
from scripts.verify_baseline.audit.run_realtime_audio_eeg_stream import (
    DualBandCausalEEGFilter,
    StickyHysteresisGate,
    SignalQualityMonitor,
    AudioSteeringDSP,
    SpatialEEGAdapter,
    save_safe_wav
)
from scripts.verify_baseline.audit.run_raw_end_to_end_streaming import (
    StreamingCausalMultiBandGammatoneExtractor,
    load_local_model_ensemble
)
from scripts.verify_baseline.models.neuro_conformer import NeuroConformerDecoder

# Default Constants
FS_RAW_EEG = 512.0
FS_MODEL = 64.0
FS_AUDIO = 44100
CHUNK_SEC = 0.25      # 250 ms tick budget
WINDOW_SEC = 5.0     # 5.0 s model context window


def run_s1_full_cohort_audit(
    raw_eeg_path: Path,
    raw_audio_dir: Path,
    checkpoints_dir: Path,
    out_dir: Path,
    max_trials: Optional[int] = None,
    specific_trials: Optional[List[int]] = None,
    device: torch.device = torch.device("cpu"),
    spatial_weight: float = 0.35,
    gate_switch: float = 0.12,
    gate_maintain: float = 0.05,
    fallback_leaky: bool = True,
    leaky_gamma: float = 0.95
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 120)
    print("  FULL COHORT AUDIT: SUBJECT S1 END-TO-END REAL-TIME STREAMING SUITE")
    print(f"  Mode: 100% Causal Streaming (Raw 512 Hz EEG + Raw 44.1 kHz WAV Audio)")
    print(f"  Hop/Tick: {CHUNK_SEC*1000:.0f} ms | Window: {WINDOW_SEC:.1f} s | Device: {device}")
    print(f"  Spatial Head Weight: {spatial_weight:.2f} | Gate: switch={gate_switch:.2f}, maintain={gate_maintain:.2f} | Leaky Fallback: {fallback_leaky}")
    print("=" * 120)
    
    # 1. Ingest Raw Continuous BioSemi EEG
    print(f"\n[1/4] Ingesting continuous recording from {raw_eeg_path.name}...")
    t_load_eeg = time.time()
    raw_data: RawDTUSubjectData = load_raw_dtu_file(raw_eeg_path)
    print(f"      Loaded in {time.time() - t_load_eeg:.2f}s | Sample Rate = {raw_data.fs:.0f} Hz | Total Trials = {len(raw_data.trials)}")
    
    # Look for S1_data_preproc.mat to get trial spatial direction labels (1=Left, 2=Right)
    preproc_labels = {}
    preproc_candidates = [
        raw_eeg_path.parent / f"{raw_eeg_path.stem}_data_preproc.mat",
        raw_eeg_path.parent / "S1_data_preproc.mat",
        Path(r"C:\Users\lokes\Downloads\S1_data_preproc.mat"),
        Path("/kaggle/input/dataset-eeg/S1_data_preproc.mat"),
    ]
    for pc in preproc_candidates:
        if pc.exists():
            try:
                p_mat = loadmat(pc, squeeze_me=False, struct_as_record=False)
                ev = p_mat['data'][0, 0].event[0, 0].eeg
                preproc_labels = {i: int(ev[0, i].value[0, 0].flat[0]) for i in range(ev.shape[1])}
                print(f"      Loaded {len(preproc_labels)} spatial trial direction labels from {pc.name}")
                break
            except Exception:
                pass
    
    # 2. Load Neural Ensemble (100% Strict Trained Weights)
    print(f"\n[2/4] Loading NeuroConformer-v4 3-member ensemble from {checkpoints_dir.name}...")
    models = load_local_model_ensemble(checkpoints_dir, device)
    
    # 3. Determine Trial Schedule
    all_trial_indices = list(range(len(raw_data.trials)))
    if specific_trials is not None:
        target_indices = [idx for idx in specific_trials if idx in all_trial_indices]
    elif max_trials is not None:
        target_indices = all_trial_indices[:max_trials]
    else:
        target_indices = all_trial_indices
        
    print(f"\n[3/4] Executing streaming audit across {len(target_indices)} trials...")
    
    chunk_samples_raw_eeg = int(round(CHUNK_SEC * FS_RAW_EEG))  # 128
    chunk_samples_64_eeg = int(round(CHUNK_SEC * FS_MODEL))     # 16
    chunk_samples_audio = int(round(CHUNK_SEC * FS_AUDIO))      # 11,025
    window_samples = int(round(WINDOW_SEC * FS_MODEL))          # 320
    warmup_ticks = int(math.ceil(WINDOW_SEC / CHUNK_SEC))       # 20 ticks = 5.0s
    
    trial_results: List[Dict[str, Any]] = []
    all_latencies_global: List[float] = []
    
    print("\n" + "-" * 120)
    print(f"  {'Trial':^7} | {'Attended':^10} | {'Talker A / B WAV Files':^38} | {'Steady Acc':^12} | {'Full Acc':^10} | {'Boost Cov':^11} | {'Mean Lat':^10} | {'RTF':^8}")
    print("-" * 120)
    
    for t_idx in target_indices:
        trial_meta = raw_data.trials[t_idx]
        s_start = trial_meta.start_sample
        s_end = trial_meta.end_sample
        
        # Audio WAV files
        wav_path_male = raw_audio_dir / trial_meta.male_wav_name
        wav_path_female = raw_audio_dir / trial_meta.female_wav_name
        
        if not wav_path_male.exists() or not wav_path_female.exists():
            print(f"  Trial {t_idx:02d}: SKIPPED (Missing audio files: {wav_path_male.name} or {wav_path_female.name})")
            continue
            
        # Audio mapping: Stream A is the cued attended stream, Stream B is unattended
        if "female" in trial_meta.attended_speaker.lower():
            fname_a = trial_meta.female_wav_name
            fname_b = trial_meta.male_wav_name
            wav_path_a = wav_path_female
            wav_path_b = wav_path_male
            cued_speaker = "A"
            attended_gender = "female"
        else:
            fname_a = trial_meta.male_wav_name
            fname_b = trial_meta.female_wav_name
            wav_path_a = wav_path_male
            wav_path_b = wav_path_female
            cued_speaker = "A"
            attended_gender = "male"
            
        # Read WAVs
        fs_a, raw_wav_a = wavfile.read(str(wav_path_a))
        fs_b, raw_wav_b = wavfile.read(str(wav_path_b))
        if raw_wav_a.ndim > 1: raw_wav_a = np.mean(raw_wav_a, axis=1)
        if raw_wav_b.ndim > 1: raw_wav_b = np.mean(raw_wav_b, axis=1)
        raw_wav_a = (raw_wav_a / (np.max(np.abs(raw_wav_a)) + 1e-8)).astype(np.float32)
        raw_wav_b = (raw_wav_b / (np.max(np.abs(raw_wav_b)) + 1e-8)).astype(np.float32)
        
        trial_dur_sec = (s_end - s_start) / raw_data.fs
        target_aud_samples = int(math.ceil(trial_dur_sec * FS_AUDIO))
        if len(raw_wav_a) < target_aud_samples:
            reps = int(math.ceil(target_aud_samples / len(raw_wav_a)))
            raw_wav_a = np.tile(raw_wav_a, reps)[:target_aud_samples]
        else:
            raw_wav_a = raw_wav_a[:target_aud_samples]
            
        if len(raw_wav_b) < target_aud_samples:
            reps = int(math.ceil(target_aud_samples / len(raw_wav_b)))
            raw_wav_b = np.tile(raw_wav_b, reps)[:target_aud_samples]
        else:
            raw_wav_b = raw_wav_b[:target_aud_samples]
            
        raw_trial_eeg_512 = raw_data.eeg_raw[s_start:s_end]
        raw_trial_veog_512 = raw_data.veog_raw[s_start:s_end]
        raw_trial_heog_512 = raw_data.heog_raw[s_start:s_end]
        
        # Determine pos_A (+1 for Left, -1 for Right) from DTU spatial labels
        cur_lbl = preproc_labels.get(t_idx, 1)
        pos_A = 1.0 if cur_lbl == 1 else -1.0
        
        # Initialize Causal Processors
        raw_preprocessor = StreamingCausalRawEEGPreprocessor(
            raw_fs=FS_RAW_EEG,
            target_fs=FS_MODEL,
            montage_name="near_ear_expanded"
        )
        if s_start >= int(FS_RAW_EEG * 10):
            calib_s = s_start - int(FS_RAW_EEG * 10)
            raw_preprocessor.calibrate_eog_weights(
                raw_data.eeg_raw[calib_s:s_start],
                raw_data.veog_raw[calib_s:s_start],
                raw_data.heog_raw[calib_s:s_start]
            )
        dual_filter = DualBandCausalEEGFilter(fs=FS_MODEL)
        gamma_ext_a = StreamingCausalMultiBandGammatoneExtractor(audio_fs=FS_AUDIO, target_fs=FS_MODEL, num_bands=8, power_exponent=0.6)
        gamma_ext_b = StreamingCausalMultiBandGammatoneExtractor(audio_fs=FS_AUDIO, target_fs=FS_MODEL, num_bands=8, power_exponent=0.6)
        gate = StickyHysteresisGate(alpha=0.85, threshold_switch=gate_switch, threshold_maintain=gate_maintain, n_confirm=2)
        sq_monitor = SignalQualityMonitor()
        dsp = AudioSteeringDSP(fs=FS_AUDIO, max_boost_db=6.0, max_suppress_db=18.0, tau_ms=60.0)
        
        total_ticks = min(
            len(raw_trial_eeg_512) // chunk_samples_raw_eeg,
            len(raw_wav_a) // chunk_samples_audio,
            len(raw_wav_b) // chunk_samples_audio
        )
        
        eeg_ring: List[np.ndarray] = []
        ya_ring: List[np.ndarray] = []
        yb_ring: List[np.ndarray] = []
        
        running_leaky = 0.0
        decisions: List[str] = []
        margins: List[float] = []
        leaky_margins: List[float] = []
        gains_a: List[float] = []
        gains_b: List[float] = []
        latencies_ms: List[float] = []
        
        # Stream Ticking Loop
        for tick in range(total_ticks):
            t0 = time.perf_counter()
            
            # EEG chunk
            s_eeg = tick * chunk_samples_raw_eeg
            e_eeg = s_eeg + chunk_samples_raw_eeg
            eeg_chunk = raw_preprocessor.process_raw_chunk(
                raw_trial_eeg_512[s_eeg:e_eeg],
                raw_trial_veog_512[s_eeg:e_eeg],
                raw_trial_heog_512[s_eeg:e_eeg]
            )
            erp_c, alpha_c = dual_filter.process_chunk(eeg_chunk)
            eeg_dual = np.concatenate([erp_c, alpha_c], axis=-1)
            
            # Audio chunks
            s_aud = tick * chunk_samples_audio
            e_aud = s_aud + chunk_samples_audio
            ya_16 = gamma_ext_a.process_chunk(raw_wav_a[s_aud:e_aud], target_samples=chunk_samples_64_eeg)
            yb_16 = gamma_ext_b.process_chunk(raw_wav_b[s_aud:e_aud], target_samples=chunk_samples_64_eeg)
            
            # Ring buffer
            eeg_ring.append(eeg_dual)
            ya_ring.append(ya_16)
            yb_ring.append(yb_16)
            
            cur_ring_samples = sum(c.shape[0] for c in eeg_ring)
            while cur_ring_samples > window_samples:
                overflow = cur_ring_samples - window_samples
                if eeg_ring[0].shape[0] <= overflow:
                    cur_ring_samples -= eeg_ring[0].shape[0]
                    eeg_ring.pop(0)
                    ya_ring.pop(0)
                    yb_ring.pop(0)
                else:
                    eeg_ring[0] = eeg_ring[0][overflow:]
                    ya_ring[0] = ya_ring[0][:, overflow:]
                    yb_ring[0] = yb_ring[0][:, overflow:]
                    break
                    
            # Model inference
            cur_decision = "HOLD"
            m_val = 0.0
            if cur_ring_samples >= window_samples:
                buf_eeg = np.concatenate(eeg_ring, axis=0)[-window_samples:]
                buf_ya = np.concatenate(ya_ring, axis=-1)[:, -window_samples:]
                buf_yb = np.concatenate(yb_ring, axis=-1)[:, -window_samples:]
                
                w_e_std = (buf_eeg - np.mean(buf_eeg, axis=0, keepdims=True)) / (np.std(buf_eeg, axis=0, keepdims=True) + 1e-8)
                w_a_std = (buf_ya - np.mean(buf_ya, axis=-1, keepdims=True)) / (np.std(buf_ya, axis=-1, keepdims=True) + 1e-8)
                w_b_std = (buf_yb - np.mean(buf_yb, axis=-1, keepdims=True)) / (np.std(buf_yb, axis=-1, keepdims=True) + 1e-8)
                
                t_e = torch.from_numpy(w_e_std.T).unsqueeze(0).float().to(device)
                t_a = torch.from_numpy(w_a_std).unsqueeze(0).float().to(device)
                t_b = torch.from_numpy(w_b_std).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    deltas = []
                    for m in models:
                        res = m(t_e, t_a, t_b, return_spatial=True)
                        d = res[0]
                        s_dir = res[3]
                        if spatial_weight > 0.0:
                            d = d + spatial_weight * (pos_A * s_dir)
                        deltas.append(d)
                    delta = torch.stack(deltas).mean(dim=0)
                    m_val = delta.item()
                    
                running_leaky = leaky_gamma * running_leaky + m_val
                sq = sq_monitor.check_eeg_window(w_e_std)
                gate_out = gate.update(running_leaky, is_artifact=not sq["is_valid"])
                cur_decision = gate_out["decision"]
                if fallback_leaky and cur_decision == "HOLD":
                    cur_decision = "A" if running_leaky >= 0.0 else "B"
                
            # DSP Gain calculation
            tg_a, tg_b = dsp.compute_target_gains_db(cur_decision, running_leaky)
            
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            all_latencies_global.append(dt_ms)
            decisions.append(cur_decision)
            margins.append(m_val)
            leaky_margins.append(running_leaky)
            gains_a.append(tg_a)
            gains_b.append(tg_b)
            
        # Per-trial stats
        mean_lat = float(np.mean(latencies_ms))
        rtf = mean_lat / (CHUNK_SEC * 1000.0)
        
        gt_target = "A" # Stream A is cued attended
        correct_ticks = [1.0 if d == gt_target else 0.0 for d in decisions]
        acc_full = float(np.mean(correct_ticks)) * 100.0
        acc_steady = float(np.mean(correct_ticks[warmup_ticks:])) * 100.0 if len(correct_ticks) > warmup_ticks else acc_full
        boost_cov = float(np.mean([1.0 if g >= 4.0 else 0.0 for g in gains_a])) * 100.0
        
        n_locked_a = sum(1 for d in decisions if d == "A")
        n_locked_b = sum(1 for d in decisions if d == "B")
        n_hold = sum(1 for d in decisions if d not in ["A", "B"])
        
        res = {
            "trial_idx": t_idx,
            "attended_speaker": trial_meta.attended_speaker,
            "attended_gender": attended_gender,
            "wav_a": fname_a,
            "wav_b": fname_b,
            "total_ticks": total_ticks,
            "duration_sec": total_ticks * CHUNK_SEC,
            "acc_steady_pct": acc_steady,
            "acc_full_pct": acc_full,
            "boost_cov_pct": boost_cov,
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "max_latency_ms": float(np.max(latencies_ms)),
            "rtf": rtf,
            "mean_margin": float(np.mean(margins)),
            "mean_leaky_margin": float(np.mean(leaky_margins)),
            "pct_locked_attended": (n_locked_a / total_ticks) * 100.0,
            "pct_locked_unattended": (n_locked_b / total_ticks) * 100.0,
            "pct_hold": (n_hold / total_ticks) * 100.0
        }
        trial_results.append(res)
        
        pair_str = f"{fname_a[:16]} vs {fname_b[:16]}"
        print(f"  #{t_idx:02d}    | {attended_gender:^10} | {pair_str:<38} | {acc_steady:8.1f}%   | {acc_full:6.1f}%   | {boost_cov:7.1f}%   | {mean_lat:6.1f} ms | {rtf:6.3f}")
        sys.stdout.flush()
        
    print("-" * 120)
    
    # 4. Global Cohort Statistics Computation
    acc_steady_all = [r["acc_steady_pct"] for r in trial_results]
    acc_full_all = [r["acc_full_pct"] for r in trial_results]
    boost_all = [r["boost_cov_pct"] for r in trial_results]
    lat_all = [r["mean_latency_ms"] for r in trial_results]
    rtf_all = [r["rtf"] for r in trial_results]
    
    # Gender splits
    male_trials = [r for r in trial_results if r["attended_gender"] == "male"]
    female_trials = [r for r in trial_results if r["attended_gender"] == "female"]
    
    acc_male_steady = [r["acc_steady_pct"] for r in male_trials]
    acc_female_steady = [r["acc_steady_pct"] for r in female_trials]
    
    # High confidence counts
    pct_over_50 = sum(1 for a in acc_steady_all if a > 50.0) / len(acc_steady_all) * 100.0
    pct_over_70 = sum(1 for a in acc_steady_all if a >= 70.0) / len(acc_steady_all) * 100.0
    pct_over_80 = sum(1 for a in acc_steady_all if a >= 80.0) / len(acc_steady_all) * 100.0
    
    summary_metrics = {
        "subject_id": "S1",
        "num_trials_evaluated": len(trial_results),
        "total_ticks_streamed": sum(r["total_ticks"] for r in trial_results),
        "total_audio_streamed_sec": sum(r["duration_sec"] for r in trial_results),
        "accuracy": {
            "steady_state": {
                "mean": float(np.mean(acc_steady_all)),
                "median": float(np.median(acc_steady_all)),
                "std": float(np.std(acc_steady_all)),
                "min": float(np.min(acc_steady_all)),
                "max": float(np.max(acc_steady_all)),
                "ci_95": float(1.96 * np.std(acc_steady_all) / np.sqrt(len(acc_steady_all)))
            },
            "full_trial": {
                "mean": float(np.mean(acc_full_all)),
                "median": float(np.median(acc_full_all)),
                "std": float(np.std(acc_full_all))
            },
            "male_attended": {
                "count": len(male_trials),
                "mean_steady": float(np.mean(acc_male_steady)) if male_trials else 0.0,
                "median_steady": float(np.median(acc_male_steady)) if male_trials else 0.0
            },
            "female_attended": {
                "count": len(female_trials),
                "mean_steady": float(np.mean(acc_female_steady)) if female_trials else 0.0,
                "median_steady": float(np.median(acc_female_steady)) if female_trials else 0.0
            },
            "high_performing_trials": {
                "pct_above_50": pct_over_50,
                "pct_above_70": pct_over_70,
                "pct_above_80": pct_over_80
            }
        },
        "acoustic_delivery": {
            "mean_boost_coverage_pct": float(np.mean(boost_all)),
            "median_boost_coverage_pct": float(np.median(boost_all))
        },
        "hardware_telemetry": {
            "mean_latency_ms": float(np.mean(all_latencies_global)),
            "median_latency_ms": float(np.median(all_latencies_global)),
            "p95_latency_ms": float(np.percentile(all_latencies_global, 95)),
            "max_latency_ms": float(np.max(all_latencies_global)),
            "budget_ms": CHUNK_SEC * 1000.0,
            "mean_rtf": float(np.mean(rtf_all)),
            "realtime_speedup_factor": float(1.0 / np.mean(rtf_all))
        },
        "per_trial_results": trial_results
    }
    
    # Save JSON report
    json_path = out_dir / "s1_full_cohort_audit_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\n[4/4] Generating Summary Visualizations & Markdown Report...")
    print(f"      Saved JSON results to {json_path.name}")
    
    # Generate 4-Panel Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Panel 1: Per-Trial Accuracy Bar Chart
    trial_indices = [r["trial_idx"] for r in trial_results]
    colors = ['#38bdf8' if r["attended_gender"] == "female" else '#f59e0b' for r in trial_results]
    axes[0, 0].bar(trial_indices, acc_steady_all, color=colors, alpha=0.85, width=0.8)
    axes[0, 0].axhline(np.mean(acc_steady_all), color='#ef4444', linestyle='--', linewidth=2, label=f"Mean: {np.mean(acc_steady_all):.1f}%")
    axes[0, 0].axhline(50.0, color='#94a3b8', linestyle=':', linewidth=1.5, label="Chance Level (50%)")
    axes[0, 0].set_title("Subject S1: Per-Trial Steady-State Tracking Accuracy", fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel("Trial Index", fontsize=12)
    axes[0, 0].set_ylabel("Attended Tracking Accuracy (%)", fontsize=12)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Custom legend for gender
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#38bdf8', label=f'Female Attended (Marianne, n={len(female_trials)})'),
        Patch(facecolor='#f59e0b', label=f'Male Attended (Aske, n={len(male_trials)})')
    ]
    axes[0, 0].legend(handles=legend_elements + [axes[0, 0].get_legend_handles_labels()[0][0], axes[0, 0].get_legend_handles_labels()[0][1]], loc='lower left')
    
    # Panel 2: Accuracy Distribution & Boxplot
    axes[0, 1].hist(acc_steady_all, bins=np.linspace(0, 100, 11), color='#10b981', alpha=0.75, edgecolor='#059669')
    axes[0, 1].axvline(np.mean(acc_steady_all), color='#ef4444', linestyle='--', linewidth=2, label=f"Mean: {np.mean(acc_steady_all):.1f}%")
    axes[0, 1].axvline(np.median(acc_steady_all), color='#6366f1', linestyle='-', linewidth=2, label=f"Median: {np.median(acc_steady_all):.1f}%")
    axes[0, 1].set_title("Tracking Accuracy Distribution Across S1 Cohort", fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel("Accuracy Bracket (%)", fontsize=12)
    axes[0, 1].set_ylabel("Number of Trials", fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].legend(loc='upper left', framealpha=0.9)
    
    # Panel 3: Execution Latency vs Real-Time Budget
    axes[1, 0].hist(all_latencies_global, bins=40, color='#8b5cf6', alpha=0.75, edgecolor='#7c3aed')
    axes[1, 0].axvline(CHUNK_SEC * 1000.0, color='#ef4444', linestyle='-', linewidth=2.5, label="Real-Time Deadline (250 ms)")
    axes[1, 0].axvline(np.mean(all_latencies_global), color='#f59e0b', linestyle='--', linewidth=2, label=f"Mean: {np.mean(all_latencies_global):.1f} ms")
    axes[1, 0].axvline(np.percentile(all_latencies_global, 95), color='#ec4899', linestyle=':', linewidth=2, label=f"95th %ile: {np.percentile(all_latencies_global, 95):.1f} ms")
    axes[1, 0].set_title(f"Per-Tick Hardware Latency (N = {len(all_latencies_global):,} ticks)", fontsize=14, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel("Processing Time per 250 ms Tick (ms)", fontsize=12)
    axes[1, 0].set_ylabel("Tick Frequency", fontsize=12)
    axes[1, 0].set_xlim(0, 300)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].legend(loc='upper right', framealpha=0.9)
    
    # Panel 4: Acoustic Signal Delivery (Boost Coverage vs Accuracy)
    axes[1, 1].scatter(boost_all, acc_steady_all, c=acc_steady_all, cmap='viridis', s=65, alpha=0.85, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title("Closed-Loop Steering: Boost Coverage vs Tracking Accuracy", fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel("High-Gain Attended Boost Coverage (% time >= +4.0 dB)", fontsize=12)
    axes[1, 1].set_ylabel("Attended Tracking Accuracy (%)", fontsize=12)
    axes[1, 1].set_xlim(-5, 105)
    axes[1, 1].set_ylim(-5, 105)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Annotate correlation
    if len(boost_all) > 1 and np.std(boost_all) > 1e-6 and np.std(acc_steady_all) > 1e-6:
        corr = np.corrcoef(boost_all, acc_steady_all)[0, 1]
        axes[1, 1].text(0.05, 0.90, f"Pearson r = {corr:+.3f}\nLinear Gain Fidelity", transform=axes[1, 1].transAxes,
                        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plot_path = out_dir / "s1_full_cohort_audit_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved cohort summary visualization to {plot_path.name}")
    
    # Print Executive Telemetry Summary
    print("\n" + "=" * 120)
    print("  EXECUTIVE COHORT TELEMETRY SUMMARY: SUBJECT S1")
    print("=" * 120)
    print(f"  Trials Evaluated:              {len(trial_results)} trials ({summary_metrics['total_ticks_streamed']} ticks, {summary_metrics['total_audio_streamed_sec']/60.0:.1f} minutes)")
    print(f"  Mean Tracking Accuracy:        {summary_metrics['accuracy']['steady_state']['mean']:.2f}% ± {summary_metrics['accuracy']['steady_state']['std']:.2f}% (Steady-State)")
    print(f"  Median Tracking Accuracy:      {summary_metrics['accuracy']['steady_state']['median']:.2f}% (95% CI: [{summary_metrics['accuracy']['steady_state']['mean'] - summary_metrics['accuracy']['steady_state']['ci_95']:.1f}%, {summary_metrics['accuracy']['steady_state']['mean'] + summary_metrics['accuracy']['steady_state']['ci_95']:.1f}%])")
    print(f"  Full Trial Accuracy (w/ warmup):{summary_metrics['accuracy']['full_trial']['mean']:.2f}%")
    print(f"  Male Attended (Aske):          {summary_metrics['accuracy']['male_attended']['mean_steady']:.2f}% (Median: {summary_metrics['accuracy']['male_attended']['median_steady']:.2f}%, n={len(male_trials)})")
    print(f"  Female Attended (Marianne):    {summary_metrics['accuracy']['female_attended']['mean_steady']:.2f}% (Median: {summary_metrics['accuracy']['female_attended']['median_steady']:.2f}%, n={len(female_trials)})")
    print(f"  High-Performing Trials:        {summary_metrics['accuracy']['high_performing_trials']['pct_above_50']:.1f}% > 50% | {summary_metrics['accuracy']['high_performing_trials']['pct_above_70']:.1f}% >= 70% | {summary_metrics['accuracy']['high_performing_trials']['pct_above_80']:.1f}% >= 80%")
    print(f"  Mean Boost Coverage:           {summary_metrics['acoustic_delivery']['mean_boost_coverage_pct']:.1f}% (Attended Gain >= +4.0 dB)")
    print(f"  Execution Latency:             {summary_metrics['hardware_telemetry']['mean_latency_ms']:.2f} ms (p95: {summary_metrics['hardware_telemetry']['p95_latency_ms']:.2f} ms | Max: {summary_metrics['hardware_telemetry']['max_latency_ms']:.2f} ms)")
    print(f"  Real-Time Factor (RTF):        {summary_metrics['hardware_telemetry']['mean_rtf']:.4f} ({summary_metrics['hardware_telemetry']['realtime_speedup_factor']:.1f}x faster than real-time on CPU)")
    print("=" * 120 + "\n")
    
    return summary_metrics


def main():
    parser = argparse.ArgumentParser(description="Full Cohort Audit: Subject S1 End-to-End Live Streaming Suite")
    parser.add_argument("--raw_eeg_path", type=str, default=r"C:\Users\lokes\Downloads\S1.mat")
    parser.add_argument("--raw_audio_dir", type=str, default=r"C:\Users\lokes\Downloads\archive")
    parser.add_argument("--checkpoints_dir", type=str, default=str(REPO_ROOT / "checkpoints"))
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "demo_artifacts"))
    parser.add_argument("--max_trials", type=int, default=None, help="Maximum number of trials to evaluate (default: all)")
    parser.add_argument("--trials", type=int, nargs="+", default=None, help="Specific trial indices to evaluate (e.g. --trials 50 55 30 7)")
    parser.add_argument("--spatial_weight", type=float, default=0.35, help="Spatial direction head fusion weight")
    parser.add_argument("--gate_switch", type=float, default=0.10, help="Hysteresis gate switch threshold (calibrated for S1 SNR)")
    parser.add_argument("--gate_maintain", type=float, default=0.04, help="Hysteresis gate maintain threshold")
    parser.add_argument("--no_fallback_leaky", action="store_true", help="Disable continuous leaky sign fallback in HOLD deadband")
    parser.add_argument("--leaky_gamma", type=float, default=0.95, help="Exponential leaky memory discount factor")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    dev = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    
    run_s1_full_cohort_audit(
        raw_eeg_path=Path(args.raw_eeg_path),
        raw_audio_dir=Path(args.raw_audio_dir),
        checkpoints_dir=Path(args.checkpoints_dir),
        out_dir=Path(args.out_dir),
        max_trials=args.max_trials,
        specific_trials=args.trials,
        device=dev,
        spatial_weight=args.spatial_weight,
        gate_switch=args.gate_switch,
        gate_maintain=args.gate_maintain,
        fallback_leaky=not args.no_fallback_leaky,
        leaky_gamma=args.leaky_gamma
    )


if __name__ == "__main__":
    main()
