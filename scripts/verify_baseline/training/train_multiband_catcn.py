"""
Universal Multi-Band Cochlear Gammatone + Causal ERP Cross-Attention CA-TCN Training Pipeline.
Supports training across all 18 DTU subjects with 8-subband tonotopic cochlear inputs
and causal ERP cross-attention tracking.

Designed for seamless execution on Kaggle GPU and rapid local CPU smoke verification.
"""

from __future__ import annotations
import argparse
import sys
import os
import json
import time
import math
import glob
from pathlib import Path
from copy import deepcopy
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.multiband_catcn import MultiBandCATCNDecoder
from src.streaming.causal_filters import StreamingCausalEEGFilter, DualBandCausalEEGFilter
from src.models.spatial_adapter import SpatialEEGAdapter
from src.selective_aad.temporal_gate import SignalQualityMonitor, StickyHysteresisGate, BayesianHMMGate
from src.selective_aad.metrics import calculate_selective_metrics, compute_temporal_stability_metrics
from baselines.ridge_aad import load_subject_examples, TrialExample
from training.montages import MONTAGES, DTU_CHANNELS
from data.multiband_provider import get_multiband_envelopes, get_mapping

FS = 64

# Prior single-band 5.0s CA-TCN benchmarks across all 18 DTU subjects
BASELINE_5S = {
    "S1": 66.3, "S2": 67.4, "S3": 60.2, "S4": 64.5, "S5": 65.1,
    "S6": 54.4, "S7": 76.0, "S8": 72.6, "S9": 61.1, "S10": 63.2,
    "S11": 57.3, "S12": 66.8, "S13": 69.8, "S14": 66.5, "S15": 78.5,
    "S16": 60.7, "S17": 64.6, "S18": 66.7
}

def resolve_output_path(path_str: str) -> Path:
    p = Path(path_str)
    is_kaggle = (os.name != 'nt') and Path("/kaggle").exists()
    if not is_kaggle and "kaggle" in str(p).lower():
        local_dir = REPO_ROOT / "results" / "multiband_catcn"
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir / p.name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """Causal low-pass filter for multi-band audio envelopes along the time axis."""
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    if data.ndim == 1:
        zi = signal.sosfilt_zi(sos) * (data[0] if len(data) > 0 else 0.0)
        out, _ = signal.sosfilt(sos, data, zi=zi)
        return out
    else:
        # Multi-band: shape [Bands, Time]
        out = np.zeros_like(data)
        for b in range(data.shape[0]):
            zi = signal.sosfilt_zi(sos) * (data[b, 0] if data.shape[1] > 0 else 0.0)
            out[b], _ = signal.sosfilt(sos, data[b], zi=zi)
        return out

def evaluate_windows(model, eeg_list, ya_list, yb_list, window_samples, device, adapter=None, streaming_context=False):
    """
    Evaluates non-overlapping windows across held-out test trials against ground truth attended stream A.
    - streaming_context=False: Isolated windows (cold-start zero-padded transients).
    - streaming_context=True: Continuous streaming (receptive fields maintain continuous buffer context).
    - adapter: Optional SpatialEEGAdapter for subject-specific skull impedance realignment.
    """
    total_wins = 0
    correct_wins = 0
    model.eval()
    if adapter is not None:
        adapter.eval()
    with torch.no_grad():
        for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
            t_len = min(len(eeg), ya.shape[-1], yb.shape[-1])
            if streaming_context:
                full_e = torch.from_numpy(eeg[:t_len].T.copy()).unsqueeze(0).float().to(device)
                full_a = torch.from_numpy(ya[:, :t_len].copy()).unsqueeze(0).float().to(device)
                full_b = torch.from_numpy(yb[:, :t_len].copy()).unsqueeze(0).float().to(device)
                if adapter is not None:
                    full_e = adapter(full_e)
                z_eeg = model.eeg_encoder(full_e)
                z_a = model.audio_encoder(full_a)
                z_b = model.audio_encoder(full_b)
                for s in range(0, t_len - window_samples + 1, window_samples):
                    e = s + window_samples
                    delta, (la, lb) = model.classifier_head(z_eeg[:, :, s:e], z_a[:, :, s:e], z_b[:, :, s:e])
                    val = delta.item()
                    if abs(val) < 1e-6:
                        correct_wins += 0.5
                    elif val > 0.0:
                        correct_wins += 1.0
                    total_wins += 1
            else:
                for s in range(0, t_len - window_samples + 1, window_samples):
                    e = s + window_samples
                    w_e = torch.from_numpy(eeg[s:e].T.copy()).unsqueeze(0).float().to(device)
                    w_a = torch.from_numpy(ya[:, s:e].copy()).unsqueeze(0).float().to(device)
                    w_b = torch.from_numpy(yb[:, s:e].copy()).unsqueeze(0).float().to(device)
                    if adapter is not None:
                        w_e = adapter(w_e)
                    delta, (la, lb), _ = model(w_e, w_a, w_b)
                    val = delta.item()
                    if abs(val) < 1e-6:
                        correct_wins += 0.5
                    elif val > 0.0:
                        correct_wins += 1.0
                    total_wins += 1
                    
    if total_wins == 0:
        return 50.0, 0
    acc = (correct_wins / total_wins) * 100.0
    return acc, total_wins

def train_subject_deep_adapter(
    base_model: nn.Module,
    calib_eeg: list[np.ndarray],
    calib_ya: list[np.ndarray],
    calib_yb: list[np.ndarray],
    calib_dir: list[float] | None = None,
    cal_dir: list[float] | None = None,
    win_samples: int = 320,
    hop_samples: int = 64,
    device: torch.device = torch.device("cpu"),
    epochs: int = 15,
    lr: float = 1e-3,
    deep_lr: float = 5e-4,
    l2_identity: float = 0.05,
    n_channels: int = 8,
    deep_adapt: bool = True
) -> tuple[SpatialEEGAdapter, float]:
    """
    Trains the dedicated SpatialEEGAdapter and optionally deep-adapts biological SincNet filters
    and spatial direction head parameters on calibration trials to rescue low-SNR subjects (e.g. S6, S11).
    Features:
    - Frobenius identity regularization: ||W - I_C||_F^2 (shrinkage towards unadapted baseline).
    - Symmetric dual-target augmentation: (x, ya, yb) -> 1.0, (x, yb, ya) -> 0.0.
    - SincNet filter frequency deviation penalty to preserve biological tracking bands.
    - Convex shrinkage optimization across lambda in [0.0, 1.0] to prevent over-rotation.
    - Platt margin temperature calibration for subject-specific adaptive gating.
    """
    if calib_dir is None and cal_dir is not None:
        calib_dir = cal_dir
        
    adapter = SpatialEEGAdapter(channels=n_channels).to(device)
    if not calib_eeg:
        return adapter, 1.0
        
    x_list, ya_list, yb_list, d_list = [], [], [], []
    for tr_i, (eeg, ya, yb) in enumerate(zip(calib_eeg, calib_ya, calib_yb)):
        cur_d = calib_dir[tr_i] if calib_dir is not None and tr_i < len(calib_dir) else 1.0
        min_len = min(len(eeg), ya.shape[-1], yb.shape[-1])
        s = 0
        while s + win_samples <= min_len:
            e = s + win_samples
            w_e = eeg[s:e]
            w_ya = ya[:, s:e]
            w_yb = yb[:, s:e]
            
            w_e_std = (w_e - np.mean(w_e, axis=0, keepdims=True)) / (np.std(w_e, axis=0, keepdims=True) + 1e-8)
            w_ya_std = (w_ya - np.mean(w_ya, axis=-1, keepdims=True)) / (np.std(w_ya, axis=-1, keepdims=True) + 1e-8)
            w_yb_std = (w_yb - np.mean(w_yb, axis=-1, keepdims=True)) / (np.std(w_yb, axis=-1, keepdims=True) + 1e-8)
            
            x_list.append(w_e_std.T)
            ya_list.append(w_ya_std)
            yb_list.append(w_yb_std)
            d_list.append(cur_d)
            s += hop_samples
            
    if not x_list:
        return adapter, 1.0
        
    x_arr = np.stack(x_list, axis=0)
    ya_arr = np.stack(ya_list, axis=0)
    yb_arr = np.stack(yb_list, axis=0)
    d_arr = np.array(d_list, dtype=np.float32)
    N = len(x_arr)
    
    # Symmetrized dual-target dataset
    x_aug = np.concatenate([x_arr, x_arr], axis=0)
    c1_aug = np.concatenate([ya_arr, yb_arr], axis=0)
    c2_aug = np.concatenate([yb_arr, ya_arr], axis=0)
    labels_aug = np.concatenate([np.ones(N, dtype=np.float32), np.zeros(N, dtype=np.float32)], axis=0)
    d_aug = np.concatenate([d_arr, d_arr], axis=0)
    
    ds = TensorDataset(
        torch.from_numpy(x_aug).float(),
        torch.from_numpy(c1_aug).float(),
        torch.from_numpy(c2_aug).float(),
        torch.from_numpy(labels_aug).float(),
        torch.from_numpy(d_aug).float()
    )
    loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
    
    # Freeze backbone by default
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
        
    deep_params = []
    f1_init, band_init = None, None
    if deep_adapt:
        if hasattr(base_model, "eeg_encoder") and hasattr(base_model.eeg_encoder, "sinc_net"):
            sn = base_model.eeg_encoder.sinc_net
            sn.f1_raw.requires_grad = True
            sn.band_raw.requires_grad = True
            f1_init = sn.f1_raw.detach().clone()
            band_init = sn.band_raw.detach().clone()
            deep_params.extend([sn.f1_raw, sn.band_raw])
        if hasattr(base_model, "spatial_head") and base_model.spatial_head is not None:
            for p in base_model.spatial_head.parameters():
                p.requires_grad = True
                deep_params.append(p)
                
    param_groups = [{'params': adapter.parameters(), 'lr': lr, 'weight_decay': 1e-4}]
    if deep_params:
        param_groups.append({'params': deep_params, 'lr': deep_lr, 'weight_decay': 1e-4})
    optimizer = optim.AdamW(param_groups)
    
    adapter.train()
    for _ in range(epochs):
        for bx, bya, byb, blab, bd in loader:
            bx, bya, byb, blab, bd = bx.to(device), bya.to(device), byb.to(device), blab.to(device), bd.to(device)
            optimizer.zero_grad(set_to_none=True)
            bx_adapted = adapter(bx)
            if deep_adapt and hasattr(base_model, "spatial_head") and base_model.spatial_head is not None:
                res = base_model(bx_adapted, bya, byb, return_spatial=True)
                if len(res) == 4:
                    delta, _, _, s_dir = res
                    loss_spatial = F.binary_cross_entropy_with_logits(s_dir, bd)
                else:
                    delta, _, _ = res[:3]
                    loss_spatial = 0.0
            else:
                delta, _, _ = base_model(bx_adapted, bya, byb)
                loss_spatial = 0.0
                
            loss_task = F.binary_cross_entropy_with_logits(delta, blab)
            loss_reg = l2_identity * adapter.identity_regularization_loss()
            loss_sinc = 0.0
            if f1_init is not None and band_init is not None:
                sn = base_model.eeg_encoder.sinc_net
                loss_sinc = 0.05 * ((sn.f1_raw - f1_init).pow(2).sum() + (sn.band_raw - band_init).pow(2).sum())
            loss = loss_task + loss_reg + loss_sinc + 0.15 * loss_spatial
            loss.backward()
            optimizer.step()
            
    # Convex Shrinkage Optimization on adapter projection weight
    W_trained = adapter.proj.weight.data.clone()
    eye = torch.eye(n_channels, device=device).unsqueeze(-1)
    best_lam = 0.0
    best_loss = float('inf')
    best_acc = -1.0
    candidate_lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    adapter.eval()
    with torch.no_grad():
        for lam in candidate_lams:
            adapter.proj.weight.data.copy_((1.0 - lam) * eye + lam * W_trained)
            cur_loss = 0.0
            cur_correct = 0
            n_tot = 0
            for bx, bya, byb, blab, bd in loader:
                bx, bya, byb, blab = bx.to(device), bya.to(device), byb.to(device), blab.to(device)
                d, _, _ = base_model(adapter(bx), bya, byb)
                cur_loss += F.binary_cross_entropy_with_logits(d, blab, reduction='sum').item()
                cur_correct += int(((d > 0.0) == (blab > 0.5)).sum().item())
                n_tot += bx.size(0)
            acc = cur_correct / max(1, n_tot)
            if acc > best_acc or (abs(acc - best_acc) < 1e-4 and cur_loss < best_loss):
                best_acc = acc
                best_loss = cur_loss
                best_lam = lam
                
    adapter.proj.weight.data.copy_((1.0 - best_lam) * eye + best_lam * W_trained)
    
    # Calculate subject-specific margin scale / temperature from calibration data
    cal_margins = []
    with torch.no_grad():
        for bx, bya, byb, blab, bd in loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            d, _, _ = base_model(adapter(bx), bya, byb)
            cal_margins.extend(d.cpu().numpy().tolist())
    cal_m = np.array(cal_margins)
    cal_std = float(np.std(cal_m)) if len(cal_m) > 1 else 1.0
    cal_temp = float(np.clip(cal_std, 0.6, 2.0))
    
    return adapter, cal_temp


def train_subject_spatial_adapter(
    base_model: nn.Module,
    calib_eeg: list[np.ndarray],
    calib_ya: list[np.ndarray],
    calib_yb: list[np.ndarray],
    win_samples: int,
    hop_samples: int,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    l2_identity: float = 0.05,
    n_channels: int = 8,
) -> tuple[SpatialEEGAdapter, float]:
    return train_subject_deep_adapter(
        base_model=base_model,
        calib_eeg=calib_eeg,
        calib_ya=calib_ya,
        calib_yb=calib_yb,
        calib_dir=None,
        win_samples=win_samples,
        hop_samples=hop_samples,
        device=device,
        epochs=epochs,
        lr=lr,
        l2_identity=l2_identity,
        n_channels=n_channels,
        deep_adapt=False
    )


def evaluate_streaming_trials(
    model: nn.Module,
    adapter: SpatialEEGAdapter | None,
    eeg_list: list[np.ndarray],
    ya_list: list[np.ndarray],
    yb_list: list[np.ndarray],
    dir_list: list[float] | None = None,
    spatial_weight: float = 0.0,
    window_sec: float = 5.0,
    step_sec: float = 0.5,
    fs: float = 64.0,
    device: torch.device = torch.device("cpu")
):
    """
    Extracts rolling window neural correlation margins on sequential test trials.
    Supports:
    - adapter=None (Zero-Shot) and adapter=SpatialEEGAdapter (Adapted).
    - spatial_weight: Fuses directional margin s_dir anti-symmetrically.
    Returns (trials_margins, trials_labels, trials_raw_eeg).
    """
    model.eval()
    if adapter is not None:
        adapter.eval()
        
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    trials_margins = []
    trials_labels = []
    trials_raw_eeg = []
    
    with torch.no_grad():
        for tr_idx, (eeg, ya, yb) in enumerate(zip(eeg_list, ya_list, yb_list)):
            t_len = min(len(eeg), ya.shape[-1], yb.shape[-1])
            cur_dir = dir_list[tr_idx] if dir_list is not None and tr_idx < len(dir_list) else 1.0
            pos_A = 1.0 if cur_dir > 0.5 else -1.0
            
            win_e, win_a, win_b = [], [], []
            raw_e = []
            
            curr_start = 0
            while curr_start + window_samples <= t_len:
                curr_end = curr_start + window_samples
                w_e = eeg[curr_start:curr_end]
                w_a = ya[:, curr_start:curr_end]
                w_b = yb[:, curr_start:curr_end]
                
                raw_e.append(w_e)
                w_e_std = (w_e - np.mean(w_e, axis=0, keepdims=True)) / (np.std(w_e, axis=0, keepdims=True) + 1e-8)
                w_a_std = (w_a - np.mean(w_a, axis=-1, keepdims=True)) / (np.std(w_a, axis=-1, keepdims=True) + 1e-8)
                w_b_std = (w_b - np.mean(w_b, axis=-1, keepdims=True)) / (np.std(w_b, axis=-1, keepdims=True) + 1e-8)
                
                win_e.append(w_e_std.T)
                win_a.append(w_a_std)
                win_b.append(w_b_std)
                curr_start += step_samples
                
            if win_e:
                t_e = torch.from_numpy(np.stack(win_e)).float().to(device)
                t_a = torch.from_numpy(np.stack(win_a)).float().to(device)
                t_b = torch.from_numpy(np.stack(win_b)).float().to(device)
                
                if adapter is not None:
                    t_e = adapter(t_e)
                    
                if spatial_weight > 0.0 and hasattr(model, "spatial_head") and model.spatial_head is not None:
                    res = model(t_e, t_a, t_b, return_spatial=True)
                    if len(res) == 4:
                        d, _, _, s_dir = res
                        # Machine-precision anti-symmetric directional fusion:
                        d = d + spatial_weight * (pos_A * s_dir)
                    else:
                        d, _, _ = res[:3]
                else:
                    d, _, _ = model(t_e, t_a, t_b)
                    
                m_vals = d.detach().cpu().numpy().tolist()
                trials_margins.append(np.array(m_vals, dtype=np.float64))
                trials_labels.append(np.ones(len(m_vals), dtype=np.int64))
                trials_raw_eeg.append(raw_e)
                
    return trials_margins, trials_labels, trials_raw_eeg

def discover_eeg_subjects(custom_eeg_dir: str = None) -> list[Path]:
    """Discovers all DTU S*_data_preproc.mat files with exhaustive error-tolerant search."""
    found_paths = set()
    
    # Priority 1: If custom_eeg_dir provided directly
    if custom_eeg_dir:
        p = Path(custom_eeg_dir)
        if p.is_file():
            found_paths.add(p)
        elif p.is_dir():
            try:
                for f in p.iterdir():
                    if f.suffix.lower() == ".mat":
                        found_paths.add(f)
            except Exception:
                pass
            try:
                for f in p.rglob("*.mat"):
                    found_paths.add(f)
            except Exception:
                pass
                
    # Priority 2: Standard Kaggle and local candidate paths
    standard_candidates = [
        Path("/kaggle/input/datasets/lokeshgile/dataset-eeg"),
        Path("/kaggle/input/dataset-eeg"),
        Path("/kaggle/input/Dataset_EEG"),
        Path("/kaggle/input/dataset_eeg"),
        Path("/kaggle/input/datasets/lokeshgile/dtu-eeg-raw"),
        Path("/kaggle/input/dtu-eeg-raw"),
        Path(r"C:\Users\lokes\Downloads\archive (2)\DATA_preproc"),
        REPO_ROOT / "data",
        VERIFY_ROOT / "data",
    ]
    for cand in standard_candidates:
        if cand.exists() and cand.is_dir():
            try:
                for f in cand.iterdir():
                    if f.suffix.lower() == ".mat":
                        found_paths.add(f)
            except Exception:
                pass
            try:
                for f in cand.rglob("*.mat"):
                    found_paths.add(f)
            except Exception:
                pass

    # Priority 3: Deep exhaustive search across /kaggle/input using both glob and os.walk with error suppression
    if Path("/kaggle/input").exists():
        try:
            for m in glob.glob("/kaggle/input/**/*.mat", recursive=True):
                found_paths.add(Path(m))
        except Exception:
            pass
            
        def ignore_err(e):
            return None
            
        for root, _, files in os.walk("/kaggle/input", onerror=ignore_err, followlinks=True):
            for f in files:
                if f.lower().endswith(".mat"):
                    found_paths.add(Path(root) / f)

    # Filter to actual DTU subject .mat files (contain data_preproc or start with S[0-9]+)
    valid_subjects = []
    for p in found_paths:
        name = p.name.lower()
        if "data_preproc" in name or (name.startswith("s") and len(name) > 1 and name[1:2].isdigit()):
            valid_subjects.append(p)
            
    if not valid_subjects:
        return []
        
    def sort_key(p: Path):
        s_part = p.stem.split("_")[0].upper()
        if s_part.startswith("S") and s_part[1:].isdigit():
            return (0, int(s_part[1:]))
        return (1, p.stem)
        
    # Deduplicate by filename stem so identical files from symlinks/aliases don't duplicate
    by_stem = {p.stem: p for p in valid_subjects}
    return sorted(by_stem.values(), key=sort_key)

def run_multiband_training(args):
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 16)
        args.calib_epochs = min(args.calib_epochs, 2)
        print("\n" + "=" * 96)
        print("  [SMOKE TEST MODE ENABLED]: Running rapid 1-epoch pipeline verification on CPU")
        print("=" * 96)
        
    print("=" * 96)
    print("  MULTIBAND COCHLEAR GAMMATONE + CAUSAL ERP CROSS-ATTENTION CA-TCN TRAINING")
    print(f"  Montage: {args.montage} ({n_ch} channels) | Audio Subbands: {args.audio_bands}")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 96)
    
    # 1. Discover Subjects
    if args.smoke_test and not args.eeg_dir:
        all_paths = [Path("S1_data_preproc.mat"), Path("S2_data_preproc.mat")]
        print(f"[DATA] Smoke test mode: using mock subjects {[p.name for p in all_paths]}")
    else:
        all_paths = discover_eeg_subjects(args.eeg_dir)
        print(f"[DATA] Discovered {len(all_paths)} total subject file(s) on disk: {[p.name for p in all_paths]}")
    
    if args.subjects and args.subjects.lower() != 'all':
        requested = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]
        filtered = [p for p in all_paths if p.stem.split("_")[0].upper() in requested]
        if len(filtered) > 0:
            all_paths = filtered
            print(f"[DATA] Filtered to {len(all_paths)} requested subject(s): {[p.stem for p in all_paths]}")
        else:
            print(f"[WARNING] None of the requested subjects {requested} were found in the available files on disk.")
            print(f"[INFO] Available subject files: {[p.name for p in all_paths]}")
            print(f"[INFO] Automatically proceeding with all {len(all_paths)} discovered subject file(s).")
            
    if not all_paths:
        if args.smoke_test:
            print("[SMOKE TEST] No real DTU files on disk. Synthesizing 2 mock subjects for pipeline audit...")
            all_paths = [Path("S1_data_preproc.mat"), Path("S2_data_preproc.mat")]
        else:
            raise FileNotFoundError(
                f"No DTU patient files found matching criteria. Checked custom eeg_dir: '{args.eeg_dir}' and /kaggle/input."
            )
    elif args.smoke_test and not args.subjects:
        all_paths = all_paths[:2]
        
    print(f"[DATA] Selected {len(all_paths)} DTU subjects for training: {[p.stem for p in all_paths]}")
    
    # 2. Load Mapping and 8-band Envelopes
    mapping = {}
    envelopes = {}
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
            print(f"[SMOKE TEST] Audio data not available ({e}). Synthesizing {args.audio_bands}-band envelopes.")
        else:
            raise e

    win_samples = int(args.window_sec * FS)
    if getattr(args, "dual_band", False):
        causal_eeg_filter = DualBandCausalEEGFilter(
            erp_lowcut=args.eeg_lowcut, erp_highcut=args.eeg_highcut,
            alpha_lowcut=args.alpha_lowcut, alpha_highcut=args.alpha_highcut,
            fs=FS, order=2, n_channels=n_ch
        )
        print(f"\n[DATA PREPARATION]: Extracting dual-band EEG (ERP: {args.eeg_lowcut}-{args.eeg_highcut} Hz + Alpha: {args.alpha_lowcut}-{args.alpha_highcut} Hz) and multi-band envelopes (<{args.audio_lowpass} Hz)...")
    else:
        causal_eeg_filter = StreamingCausalEEGFilter(
            lowcut=args.eeg_lowcut, highcut=args.eeg_highcut, fs=FS, order=2, n_channels=n_ch
        )
        print(f"\n[DATA PREPARATION]: Extracting causal streaming EEG ({args.eeg_lowcut}-{args.eeg_highcut} Hz) and multi-band envelopes (<{args.audio_lowpass} Hz)...")
    t_data_start = time.time()
    
    X_tr_list, YA_tr_list, YB_tr_list, DIR_tr_list = [], [], [], []
    X_va_list, YA_va_list, YB_va_list, DIR_va_list = [], [], [], []
    subject_test_data = {}
    total_train_trials = 0
    total_test_trials = 0
    
    for p in all_paths:
        sub_name = p.stem
        sub_key = sub_name.replace("_data_preproc", "")
        
        if args.smoke_test and (not p.exists() or not envelopes):
            # Synthesize 4 trials of DTU length (3200 samples = 50.0s @ 64 Hz)
            n_trials = 4
            exs = [
                TrialExample(
                    subject=sub_name,
                    trial_index=i,
                    eeg=np.random.randn(3200, 64).astype(np.float32),
                    wav_a=np.random.randn(3200).astype(np.float32),
                    wav_b=np.random.randn(3200).astype(np.float32),
                    label=1 if (i % 2 == 0) else 2
                )
                for i in range(n_trials)
            ]
            raw_ya_list = [np.random.randn(args.audio_bands, 3200).astype(np.float32) for _ in range(n_trials)]
            raw_yb_list = [np.random.randn(args.audio_bands, 3200).astype(np.float32) for _ in range(n_trials)]
        else:
            exs = list(load_subject_examples(p))
            if args.smoke_test:
                exs = exs[:4]
                
            raw_ya_list = []
            raw_yb_list = []
            valid_exs = []
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
        sub_eeg_te, sub_ya_te, sub_yb_te, sub_dir_te = [], [], [], []
        sub_eeg_cal, sub_ya_cal, sub_yb_cal, sub_dir_cal = [], [], [], []
        
        for idx in range(n_valid):
            raw_eeg = exs[idx].eeg[:, montage_channels].astype(np.float32)
            cur_ya = raw_ya_list[idx]
            cur_yb = raw_yb_list[idx]
            min_len = min(len(raw_eeg), cur_ya.shape[-1], cur_yb.shape[-1])
            raw_eeg = raw_eeg[:min_len]
            cur_ya = cur_ya[:, :min_len]
            cur_yb = cur_yb[:, :min_len]
            
            cur_label = getattr(exs[idx], 'label', 1)
            # In DTU: label 1 = Attend Left (-60 deg), label 2 = Attend Right (+60 deg)
            trial_dir = 1.0 if cur_label == 1 else 0.0
            
            # Causal EEG filtering + standardization
            causal_eeg_filter.reset()
            if getattr(args, "dual_band", False):
                eeg_erp, eeg_alpha = causal_eeg_filter.process_chunk(raw_eeg)
                eeg_erp = (eeg_erp - np.mean(eeg_erp, axis=0, keepdims=True)) / (np.std(eeg_erp, axis=0, keepdims=True) + 1e-12)
                eeg_alpha = (eeg_alpha - np.mean(eeg_alpha, axis=0, keepdims=True)) / (np.std(eeg_alpha, axis=0, keepdims=True) + 1e-12)
                eeg_c = np.concatenate([eeg_erp, eeg_alpha], axis=-1)
            else:
                eeg_c = causal_eeg_filter.process_chunk(raw_eeg)
                eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-12)
            
            # Causal Audio lowpass + standardization
            ya_c = butter_lowpass_sosfilt(cur_ya, args.audio_lowpass, FS, order=2).astype(np.float32)
            yb_c = butter_lowpass_sosfilt(cur_yb, args.audio_lowpass, FS, order=2).astype(np.float32)
            ya_c = (ya_c - np.mean(ya_c, axis=-1, keepdims=True)) / (np.std(ya_c, axis=-1, keepdims=True) + 1e-12)
            yb_c = (yb_c - np.mean(yb_c, axis=-1, keepdims=True)) / (np.std(yb_c, axis=-1, keepdims=True) + 1e-12)
            
            if getattr(args, "include_onsets", False):
                onset_a = np.maximum(0.0, np.diff(ya_c, prepend=ya_c[:, :1], axis=-1))
                onset_b = np.maximum(0.0, np.diff(yb_c, prepend=yb_c[:, :1], axis=-1))
                onset_a = (onset_a - np.mean(onset_a, axis=-1, keepdims=True)) / (np.std(onset_a, axis=-1, keepdims=True) + 1e-12)
                onset_b = (onset_b - np.mean(onset_b, axis=-1, keepdims=True)) / (np.std(onset_b, axis=-1, keepdims=True) + 1e-12)
                ya_c = np.concatenate([ya_c, onset_a], axis=0)
                yb_c = np.concatenate([yb_c, onset_b], axis=0)

            if args.include_broadband:
                bb_a = np.mean(ya_c[:args.audio_bands], axis=0, keepdims=True)
                bb_b = np.mean(yb_c[:args.audio_bands], axis=0, keepdims=True)
                ya_c = np.concatenate([bb_a, ya_c], axis=0)
                yb_c = np.concatenate([bb_b, yb_c], axis=0)
            
            # IN DTU: wavA (ya_c) is ALWAYS attended, wavB (yb_c) is ALWAYS unattended
            if idx < split_idx:
                total_train_trials += 1
                x_t = eeg_c.T # [C_eeg, T]
                is_val_trial = (idx % 10 == 0)
                
                # Calibration trials preserved for few-shot spatial adaptation (within train split)
                if idx < args.calib_trials:
                    sub_eeg_cal.append(eeg_c)
                    sub_ya_cal.append(ya_c)
                    sub_yb_cal.append(yb_c)
                    sub_dir_cal.append(trial_dir)
                
                # Chunk training trials
                hop_samples = int(args.hop_sec * FS)
                start = 0
                while start + win_samples <= min_len:
                    end = start + win_samples
                    w_x = x_t[:, start:end]
                    w_ya = ya_c[:, start:end]
                    w_yb = yb_c[:, start:end]
                    
                    if is_val_trial:
                        X_va_list.append(w_x)
                        YA_va_list.append(w_ya)
                        YB_va_list.append(w_yb)
                        DIR_va_list.append(trial_dir)
                    else:
                        X_tr_list.append(w_x)
                        YA_tr_list.append(w_ya)
                        YB_tr_list.append(w_yb)
                        DIR_tr_list.append(trial_dir)
                    start += hop_samples
            else:
                total_test_trials += 1
                sub_eeg_te.append(eeg_c)
                sub_ya_te.append(ya_c)
                sub_yb_te.append(yb_c)
                sub_dir_te.append(trial_dir)
                
        if sub_eeg_te:
            subject_test_data[sub_name] = (sub_eeg_te, sub_ya_te, sub_yb_te, sub_dir_te, sub_eeg_cal, sub_ya_cal, sub_yb_cal, sub_dir_cal)
            
    print(f"[DATA READY]: Extracted {len(X_tr_list)} train windows, {len(X_va_list)} val windows across {total_train_trials} trials in {time.time()-t_data_start:.1f}s.")
    
    if len(X_tr_list) == 0:
        raise RuntimeError("No training windows extracted. Please check dataset paths and envelopes.")
        
    X_tr = torch.from_numpy(np.stack(X_tr_list, axis=0)).float()
    YA_tr = torch.from_numpy(np.stack(YA_tr_list, axis=0)).float()
    YB_tr = torch.from_numpy(np.stack(YB_tr_list, axis=0)).float()
    DIR_tr = torch.tensor(DIR_tr_list, dtype=torch.float32)
    
    # Free memory
    del X_tr_list, YA_tr_list, YB_tr_list, DIR_tr_list
    import gc
    gc.collect()
    
    if len(X_va_list) > 0:
        X_va = torch.from_numpy(np.stack(X_va_list, axis=0)).float()
        YA_va = torch.from_numpy(np.stack(YA_va_list, axis=0)).float()
        YB_va = torch.from_numpy(np.stack(YB_va_list, axis=0)).float()
        DIR_va = torch.tensor(DIR_va_list, dtype=torch.float32)
        del X_va_list, YA_va_list, YB_va_list, DIR_va_list
        gc.collect()
    else:
        X_va = X_tr[:min(16, len(X_tr))]
        YA_va = YA_tr[:min(16, len(YA_tr))]
        YB_va = YB_tr[:min(16, len(YB_tr))]
        DIR_va = DIR_tr[:min(16, len(DIR_tr))]
        
    # GPU OPTIMIZATION: Pinned memory for async DMA transfers
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        TensorDataset(X_tr, YA_tr, YB_tr, DIR_tr),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=(len(X_tr) > args.batch_size),
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0
    )
    val_loader = DataLoader(
        TensorDataset(X_va, YA_va, YB_va, DIR_va),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0
    )
    
    # Freeze and preserve existing baseline checkpoint if present on disk
    legacy_ckpt = Path("/kaggle/working/multiband_catcn_best.pt")
    frozen_ckpt = Path("/kaggle/working/multiband_catcn_frozen_baseline.pt")
    if legacy_ckpt.exists() and not frozen_ckpt.exists():
        try:
            import shutil
            shutil.copyfile(legacy_ckpt, frozen_ckpt)
            print(f"[FREEZE]: Saved and preserved baseline checkpoint to: {frozen_ckpt}")
        except Exception as e:
            print(f"[WARNING]: Could not freeze legacy checkpoint: {e}")

    # 4. Instantiate Multi-Band CA-TCN Model
    actual_eeg_channels = X_tr.shape[1]
    audio_in_channels = YA_tr.shape[1]
    model = MultiBandCATCNDecoder(
        eeg_channels=actual_eeg_channels,
        audio_bands=audio_in_channels,
        hidden_dim=args.hidden_dim,
        min_lag_samples=args.min_lag_samples,
        max_lag_samples=args.max_lag_samples,
        head_type=args.head_type,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        use_sinc=args.use_sinc,
        arch=args.arch,
        subsample_stride=args.subsample_stride
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.arch in ["conformer", "neuroconformer"]:
        model_tag = "NeuroConformer-v4"
    elif args.arch == "msca":
        model_tag = "MSCA-CATCN-v3"
    elif args.use_sinc:
        model_tag = "Sinc-CATCN-v2"
    else:
        model_tag = "MultiBand-CATCN-Baseline"
    print(f"\n[MODEL INITIALIZED]: {model_tag} with {n_params:,} trainable parameters.")
    print(f"  Head: {args.head_type} (dropout={args.head_dropout}) | Loss: {args.loss} (margin={args.margin}, tau={args.loss_temp}) | Lags: [{args.min_lag_samples}, {args.max_lag_samples}]")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = 1 if args.arch in ["conformer", "neuroconformer"] else 0
    if warmup_epochs > 0 and args.epochs > 1:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.33, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-5)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_weights = deepcopy(model.state_dict())
    
    # Checkpoint pre-loading if provided or auto-discovered in eval_only mode
    if not args.checkpoint_path and args.eval_only:
        default_candidates = [
            Path("/kaggle/working/hybrid_neuro_conformer_best.pt"),
            Path("/kaggle/working/sinc_multiband_catcn_best.pt"),
            Path(resolve_output_path(args.output_model))
        ]
        for dc in default_candidates:
            if dc.exists():
                args.checkpoint_path = str(dc)
                print(f"[AUTO DISCOVERY]: Auto-detected pre-trained checkpoint: {dc}")
                break

    if args.checkpoint_path:
        ckpt_candidate = Path(args.checkpoint_path)
        if ckpt_candidate.exists():
            print(f"\n[CHECKPOINT]: Loading pre-trained weights from {ckpt_candidate}...")
            st = torch.load(ckpt_candidate, map_location=device, weights_only=False)
            if "model_state_dict" in st:
                st = st["model_state_dict"]
            if any(k.startswith("model.") for k in st.keys()) and not any(k.startswith("model.") for k in model.state_dict().keys()):
                st = {k[6:]: v for k, v in st.items()}
            elif not any(k.startswith("model.") for k in st.keys()) and any(k.startswith("model.") for k in model.state_dict().keys()):
                st = {f"model.{k}": v for k, v in st.items()}
            missing, unexpected = model.load_state_dict(st, strict=False)
            best_weights = deepcopy(model.state_dict())
            print(f"  --> Pre-trained checkpoint loaded successfully (missing={len(missing)}, unexpected={len(unexpected)}).")
        else:
            print(f"\n[WARNING]: Specified checkpoint {ckpt_candidate} not found on disk.")

    # 5. Training Loop
    if args.eval_only:
        print("\n" + "=" * 96)
        print("  [EVAL ONLY MODE]: Skipping backbone training, evaluating loaded checkpoint directly.")
        print("=" * 96)
    else:
        print("\n" + "=" * 96)
        print(f"  COMMENCING {model_tag.upper()} TRAINING ({args.epochs} EPOCHS)")
        print("=" * 96)
    
    epochs_no_improve = 0
    for epoch in ([] if args.eval_only else range(1, args.epochs + 1)):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        n_train_samples = 0
        n_train_batches = 0
        
        for bx, bya, byb, bdir in train_loader:
            bx = bx.to(device, non_blocking=True)
            bya = bya.to(device, non_blocking=True)
            byb = byb.to(device, non_blocking=True)
            bdir = bdir.to(device, non_blocking=True)
            
            # Subband SpecAugment (mask 1 random band with p=args.subband_mask_prob to prevent relying on single subband noise)
            if args.subband_mask_prob > 0 and np.random.rand() < args.subband_mask_prob:
                mb = np.random.randint(0, audio_in_channels)
                bya = bya.clone()
                byb = byb.clone()
                bya[:, mb, :] = 0.0
                byb[:, mb, :] = 0.0

            # Temporal Audio SpecAugment (mask a random 250ms snippet to prevent acoustic sentence memorization)
            if args.time_mask_prob > 0 and np.random.rand() < args.time_mask_prob:
                t_mask_len = int(0.25 * FS)
                t_start = np.random.randint(0, max(1, bx.size(-1) - t_mask_len))
                bya = bya.clone()
                byb = byb.clone()
                bya[:, :, t_start:t_start + t_mask_len] = 0.0
                byb[:, :, t_start:t_start + t_mask_len] = 0.0

            # Spatial Electrode SpecAugment (mask 1 random channel with p=args.channel_mask_prob to prevent hemisphere bias)
            if args.channel_mask_prob > 0 and np.random.rand() < args.channel_mask_prob:
                mc = np.random.randint(0, bx.size(1))
                bx = bx.clone()
                bx[:, mc, :] = 0.0
            
            # Symmetrized anti-biased candidate stream swapping
            swap = torch.rand(bx.size(0), device=device) > 0.5
            c1 = torch.where(swap[:, None, None], byb, bya)
            c2 = torch.where(swap[:, None, None], bya, byb)
            target_sign = torch.where(swap, -1.0, 1.0)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                if getattr(args, "spatial_loss_weight", 0.0) > 0 and hasattr(model, "spatial_head") and model.spatial_head is not None:
                    delta, (l1, l2), embeds, s_dir = model(bx, c1, c2, return_spatial=True)
                    loss_spatial = F.binary_cross_entropy_with_logits(s_dir, bdir)
                else:
                    delta, (l1, l2), embeds = model(bx, c1, c2)
                    loss_spatial = 0.0
                    
                if args.loss == 'softplus':
                    # Focal smooth logistic margin loss: downweights already well-separated windows, focuses gradient on ambiguous trials
                    p_correct = torch.sigmoid((target_sign * delta) / args.loss_temp)
                    focal_weight = torch.clamp(1.0 - p_correct, min=0.15, max=1.0)
                    loss = (focal_weight * args.loss_temp * F.softplus((args.margin - target_sign * delta) / args.loss_temp)).mean()
                else:
                    loss = torch.clamp(args.margin - target_sign * delta, min=0.0).mean()
                    
                if getattr(args, "spatial_loss_weight", 0.0) > 0 and loss_spatial != 0.0:
                    loss = loss + args.spatial_loss_weight * loss_spatial
                    
                if getattr(args, "contrastive_weight", 0.0) > 0 and embeds is not None and len(embeds) == 3:
                    ze, z1, z2 = embeds
                    eeg_emb = F.normalize(ze.mean(dim=-1), p=2, dim=-1)
                    a1_emb = F.normalize(z1.mean(dim=-1), p=2, dim=-1)
                    a2_emb = F.normalize(z2.mean(dim=-1), p=2, dim=-1)
                    att_emb = torch.where(swap[:, None], a2_emb, a1_emb)
                    unatt_emb = torch.where(swap[:, None], a1_emb, a2_emb)
                    
                    c_temp = getattr(args, "contrastive_temp", 0.1)
                    pos_sim = (eeg_emb * att_emb).sum(dim=-1, keepdim=True) / c_temp
                    unatt_sim = (eeg_emb * unatt_emb).sum(dim=-1, keepdim=True) / c_temp
                    all_cross_sim = torch.matmul(eeg_emb, att_emb.T) / c_temp
                    diag_mask = torch.eye(bx.size(0), device=device, dtype=torch.bool)
                    all_cross_sim.masked_fill_(diag_mask, float('-inf'))
                    
                    logits_nce = torch.cat([pos_sim, unatt_sim, all_cross_sim], dim=-1)
                    targets_nce = torch.zeros(bx.size(0), dtype=torch.long, device=device)
                    loss = loss + args.contrastive_weight * F.cross_entropy(logits_nce, targets_nce)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * bx.size(0)
            train_correct += int(((target_sign * delta) > 0).sum().item())
            n_train_samples += bx.size(0)
            n_train_batches += 1
            
        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train_samples)
        train_acc = (train_correct / max(1, n_train_samples)) * 100.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        n_val_samples = 0
        with torch.no_grad():
            for bx, bya, byb, bdir in val_loader:
                bx = bx.to(device, non_blocking=True)
                bya = bya.to(device, non_blocking=True)
                byb = byb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    delta, (la, lb), _ = model(bx, bya, byb)
                    if args.loss == 'softplus':
                        p_correct = torch.sigmoid(delta / args.loss_temp)
                        focal_weight = torch.clamp(1.0 - p_correct, min=0.15, max=1.0)
                        v_loss = (focal_weight * args.loss_temp * F.softplus((args.margin - delta) / args.loss_temp)).mean()
                    else:
                        v_loss = torch.clamp(args.margin - delta, min=0.0).mean()
                val_loss += v_loss.item() * bx.size(0)
                val_correct += int((delta > 0).sum().item())
                n_val_samples += bx.size(0)
                
        avg_val_loss = val_loss / max(1, n_val_samples)
        val_acc = (val_correct / max(1, n_val_samples)) * 100.0
        epoch_sec = time.time() - t_epoch_start
        
        # Checkpoint Criterion: Prioritize Validation 2AFC Accuracy over uncalibrated margin loss
        is_best = (val_acc > best_val_acc) or (abs(val_acc - best_val_acc) < 1e-4 and avg_val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_weights = deepcopy(model.state_dict())
            star_flag = f" [*BEST Acc: {val_acc:5.1f}%*]"
            epochs_no_improve = 0
        else:
            star_flag = ""
            epochs_no_improve += 1
            
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:5.1f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:5.1f}%) | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_sec:.1f}s{star_flag}")
        
        if epoch % 5 == 0 or is_best or epoch == args.epochs:
            ckpt_path = resolve_output_path(args.output_model)
            torch.save(best_weights, ckpt_path)
            
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\n[EARLY STOPPING]: Validation accuracy plateaued for {args.patience} epochs.")
            print(f"  --> Stopping early at Epoch {epoch} to prevent overfitting and save GPU compute.")
            print(f"  --> Restoring best checkpoint from Epoch {epoch - epochs_no_improve} (*BEST Val Acc: {best_val_acc:.1f}%*).")
            break
            
    # 6. Final Evaluation on Held-Out Test Split (Multi-Tier Combined Benchmark)
    gate_name = "Bayesian HMM Gate" if getattr(args, "gate_type", "sticky") == "hmm" else "Sticky Hysteresis Gate"
    gate_col = "Adapt+HMM Gate(5s)" if getattr(args, "gate_type", "sticky") == "hmm" else "Adapt+Sticky Gate(5s)"
    adapt_params = actual_eeg_channels * actual_eeg_channels
    print("\n" + "=" * 138)
    print("  GRAND COHORT BENCHMARK ON HELD-OUT TEST TRIALS (UNIFIED BEST METHODS COMBINED)")
    print(f"  Backbone: {model_tag} | Spatial: {adapt_params}-Param Adapter ({'Deep' if getattr(args, 'deep_adapt', False) else 'Linear'}) | Gate: {gate_name}")
    print("=" * 138)
    model.load_state_dict(best_weights)
    model.eval()
    
    subject_results = {}
    cohort_t1_acc = []
    cohort_t1_fsw = []
    cohort_t2_acc = []
    cohort_t2_fsw = []
    cohort_t2_gain = []
    cohort_leaky_acc = []
    cohort_leaky_fsw = []
    cohort_t3_acc = []
    cohort_t3_cov = []
    cohort_t3_fsw = []
    cohort_10s = []
    cohort_20s = []
    
    win_5s_smp = int(args.window_sec * FS)
    win_10s_smp = int(10.0 * FS)
    win_20s_smp = int(20.0 * FS)
    sq_monitor = SignalQualityMonitor()
    
    print(f"  {'Subject':<8} | {'1. Raw Zero-Shot(5s)':<18} | {'2. Raw Adapted(5s)':<22} | {'3. Leaky Contin(5s)':<18} | {gate_col:<24} | {'10.0s':<7} | {'20.0s':<7} | {'Base(5s)':<8}")
    print(f"  {'':<8} | {'Acc':<7} {'FalseSw':<9} | {'Acc':<7} {'dRaw':<6} {'FalseSw':<7} | {'Acc':<7} {'FalseSw':<9} | {'Acc':<7} {'Boost%':<7} {'FalseSw':<8} | {'Acc':<7} | {'Acc':<7} | {'Canon':<8}")
    print("  " + "-" * 134)
    
    for sub_name, (te_eeg, te_ya, te_yb, te_dir, cal_eeg, cal_ya, cal_yb, cal_dir) in subject_test_data.items():
        s_key = sub_name.split("_")[0].upper()
        base_acc = BASELINE_5S.get(s_key, 65.7)
        
        # --- Tier 1: Raw Zero-Shot 5.0s Decoder (Streaming Margins) ---
        zs_margins, zs_labels, _ = evaluate_streaming_trials(
            model, None, te_eeg, te_ya, te_yb, dir_list=te_dir, spatial_weight=0.0,
            window_sec=args.window_sec, step_sec=args.gate_step_sec, fs=FS, device=device
        )
        if zs_margins:
            flat_zs_m = np.concatenate(zs_margins)
            flat_zs_l = np.concatenate(zs_labels)
            gt_test_str = np.where(flat_zs_l == 1, "A", "B")
            t1_preds = np.where(flat_zs_m >= 0, "A", "B")
            m_t1 = calculate_selective_metrics(t1_preds, gt_test_str)
            t1_acc = m_t1["selective_accuracy"] * 100.0
            t1_fsw = float(np.mean([
                compute_temporal_stability_metrics(np.where(m >= 0, "A", "B"), np.where(l == 1, "A", "B"), args.gate_step_sec)["false_switches_per_minute"]
                for m, l in zip(zs_margins, zs_labels)
            ]))
        else:
            t1_acc, t1_fsw = 50.0, 0.0
            gt_test_str = np.array([])
            
        # --- Tier 2: Subject Adaptation (Deep or Linear) ---
        eeg_ch_dim = te_eeg[0].shape[-1]
        orig_model_state = deepcopy(model.state_dict())
        if args.adapt and cal_eeg:
            adapter, cal_temp = train_subject_deep_adapter(
                model, cal_eeg, cal_ya, cal_yb, cal_dir=cal_dir,
                win_samples=win_5s_smp, hop_samples=int(args.hop_sec * FS), device=device,
                epochs=args.calib_epochs, lr=args.calib_lr, deep_lr=getattr(args, "deep_adapt_lr", 5e-4),
                l2_identity=args.l2_identity, n_channels=eeg_ch_dim, deep_adapt=getattr(args, "deep_adapt", False)
            )
        else:
            adapter = SpatialEEGAdapter(channels=eeg_ch_dim).to(device)
            cal_temp = 1.0
            
        ad_margins, ad_labels, ad_raw_eeg = evaluate_streaming_trials(
            model, adapter, te_eeg, te_ya, te_yb, dir_list=te_dir,
            spatial_weight=getattr(args, "spatial_weight", 0.0),
            window_sec=args.window_sec, step_sec=args.gate_step_sec, fs=FS, device=device
        )
        # Strict inter-subject scientific isolation: restore model weights
        model.load_state_dict(orig_model_state)
        
        if ad_margins:
            flat_ad_m = np.concatenate(ad_margins)
            t2_preds = np.where(flat_ad_m >= 0, "A", "B")
            m_t2 = calculate_selective_metrics(t2_preds, gt_test_str)
            t2_acc = m_t2["selective_accuracy"] * 100.0
            t2_fsw = float(np.mean([
                compute_temporal_stability_metrics(np.where(m >= 0, "A", "B"), np.where(l == 1, "A", "B"), args.gate_step_sec)["false_switches_per_minute"]
                for m, l in zip(ad_margins, ad_labels)
            ]))
            raw_gain_pp = t2_acc - t1_acc
        else:
            t2_acc, t2_fsw, raw_gain_pp = t1_acc, t1_fsw, 0.0
            
        # --- Tier 3: Continuous Leaky Cumulative Decision Integration ---
        leaky_trials = []
        gamma = getattr(args, "leaky_gamma", 0.90)
        for m_seq in ad_margins:
            l_seq = np.zeros_like(m_seq)
            r_m = 0.0
            for k, val in enumerate(m_seq):
                r_m = gamma * r_m + val
                l_seq[k] = r_m
            leaky_trials.append(l_seq)
            
        if leaky_trials and len(np.concatenate(leaky_trials)) > 0:
            flat_leaky = np.concatenate(leaky_trials)
            t_leaky_preds = np.where(flat_leaky >= 0, "A", "B")
            m_leaky = calculate_selective_metrics(t_leaky_preds, gt_test_str)
            leaky_acc = m_leaky["selective_accuracy"] * 100.0
            leaky_fsw = float(np.mean([
                compute_temporal_stability_metrics(np.where(lm >= 0, "A", "B"), np.where(l == 1, "A", "B"), args.gate_step_sec)["false_switches_per_minute"]
                for lm, l in zip(leaky_trials, ad_labels)
            ]))
        else:
            leaky_acc, leaky_fsw = t2_acc, t2_fsw
            
        # --- Tier 4: Adapted + Gate (Sticky Hysteresis or Bayesian HMM) ---
        t3_dec_list = []
        t3_gains_attended = []
        for m_seq, eeg_trial in zip(ad_margins, ad_raw_eeg):
            if getattr(args, "gate_type", "sticky") == "hmm":
                gate = BayesianHMMGate(
                    mu=getattr(args, "hmm_mu", 0.35),
                    sigma=getattr(args, "hmm_sigma", 0.50),
                    switch_prior=getattr(args, "hmm_switch_prior", 0.015),
                    decision_threshold=getattr(args, "hmm_threshold", 0.70),
                    temperature=cal_temp,
                    deadband_timeout_steps=24
                )
            else:
                gate = StickyHysteresisGate(
                    alpha=args.gate_alpha,
                    threshold_switch=args.gate_switch * cal_temp,
                    threshold_maintain=args.gate_maintain * cal_temp,
                    n_confirm=2,
                    deadband_timeout_steps=24,
                    temperature=cal_temp
                )
            trial_decs = []
            trial_gains = []
            for v, w_eeg in zip(m_seq, eeg_trial):
                sq = sq_monitor.check_eeg_window(w_eeg)
                out = gate.update(v, is_artifact=not sq["is_valid"])
                trial_decs.append(out["decision"])
                trial_gains.append(out["gain_a"])
            t3_dec_list.append(np.array(trial_decs))
            t3_gains_attended.append(np.array(trial_gains))
            
        if t3_dec_list and len(np.concatenate(t3_dec_list)) > 0:
            flat_t3 = np.concatenate(t3_dec_list)
            m_t3 = calculate_selective_metrics(flat_t3, gt_test_str)
            t3_acc = m_t3["selective_accuracy"] * 100.0 if m_t3["accepted_count"] > 0 else t2_acc
            t3_cov = float(np.mean(np.concatenate(t3_gains_attended) >= 0.80) * 100.0)
            t3_hold = m_t3["abstention_rate"] * 100.0
            t3_fsw = float(np.mean([
                compute_temporal_stability_metrics(d, np.where(l == 1, "A", "B"), args.gate_step_sec)["false_switches_per_minute"]
                for d, l in zip(t3_dec_list, ad_labels)
            ]))
        else:
            t3_acc, t3_cov, t3_hold, t3_fsw = t2_acc, 0.0, 0.0, t2_fsw
            
        # --- Multi-Scale 10s & 20s Window Accuracies ---
        acc_10s, _ = evaluate_windows(model, te_eeg, te_ya, te_yb, win_10s_smp, device, adapter=adapter, streaming_context=args.streaming_context)
        acc_20s, _ = evaluate_windows(model, te_eeg, te_ya, te_yb, win_20s_smp, device, adapter=adapter, streaming_context=args.streaming_context)
        
        subject_results[sub_name] = {
            "tier1_zero_shot_5s": round(t1_acc, 2),
            "tier1_false_switches_5s": round(t1_fsw, 2),
            "tier2_adapted_5s": round(t2_acc, 2),
            "tier2_delta_raw_pp": round(raw_gain_pp, 2),
            "tier2_false_switches_5s": round(t2_fsw, 2),
            "tier3_leaky_contin_5s": round(leaky_acc, 2),
            "tier3_leaky_false_switches_5s": round(leaky_fsw, 2),
            "tier4_sticky_gated_5s": round(t3_acc, 2),
            "tier4_useful_boost_cov": round(t3_cov, 2),
            "tier4_hold_rate": round(t3_hold, 2),
            "tier4_false_switches_5s": round(t3_fsw, 2),
            "multiscale_10s": round(acc_10s, 2),
            "multiscale_20s": round(acc_20s, 2),
            "baseline_5s": round(base_acc, 2),
            "net_gain_over_baseline": round(max(t3_acc, leaky_acc) - base_acc, 2)
        }
        
        cohort_t1_acc.append(t1_acc)
        cohort_t1_fsw.append(t1_fsw)
        cohort_t2_acc.append(t2_acc)
        cohort_t2_gain.append(raw_gain_pp)
        cohort_t2_fsw.append(t2_fsw)
        cohort_leaky_acc.append(leaky_acc)
        cohort_leaky_fsw.append(leaky_fsw)
        cohort_t3_acc.append(t3_acc)
        cohort_t3_cov.append(t3_cov)
        cohort_t3_fsw.append(t3_fsw)
        cohort_10s.append(acc_10s)
        cohort_20s.append(acc_20s)
        
        print(f"  {s_key:<8} | {t1_acc:>5.1f}% {t1_fsw:>6.2f}/m | {t2_acc:>5.1f}% {raw_gain_pp:>+5.1f} {t2_fsw:>5.2f} | {leaky_acc:>5.1f}% {leaky_fsw:>6.2f}/m | {t3_acc:>5.1f}% {t3_cov:>6.1f}% {t3_fsw:>6.2f}/m | {acc_10s:>5.1f}% | {acc_20s:>5.1f}% | {base_acc:>5.1f}%")
        
    if subject_results:
        m_t1 = float(np.mean(cohort_t1_acc))
        m_t1_fsw = float(np.mean(cohort_t1_fsw))
        m_t2 = float(np.mean(cohort_t2_acc))
        m_t2_gain = float(np.mean(cohort_t2_gain))
        m_t2_fsw = float(np.mean(cohort_t2_fsw))
        m_leaky = float(np.mean(cohort_leaky_acc))
        m_leaky_fsw = float(np.mean(cohort_leaky_fsw))
        m_t3 = float(np.mean(cohort_t3_acc))
        m_t3_cov = float(np.mean(cohort_t3_cov))
        m_t3_fsw = float(np.mean(cohort_t3_fsw))
        m_10s = float(np.mean(cohort_10s))
        m_20s = float(np.mean(cohort_20s))
        
        print("  " + "-" * 134)
        print(f"  {'AVERAGE':<8} | {m_t1:>5.1f}% {m_t1_fsw:>6.2f}/m | {m_t2:>5.1f}% {m_t2_gain:>+5.1f} {m_t2_fsw:>5.2f} | {m_leaky:>5.1f}% {m_leaky_fsw:>6.2f}/m | {m_t3:>5.1f}% {m_t3_cov:>6.1f}% {m_t3_fsw:>6.2f}/m | {m_10s:>5.1f}% | {m_20s:>5.1f}% | 65.7%")
    print("=" * 138)
    
    # Save Metrics JSON
    metrics = {
        "architecture": model_tag,
        "use_sinc": args.use_sinc,
        "loss": args.loss,
        "loss_temp": args.loss_temp,
        "margin": args.margin,
        "montage": args.montage,
        "n_channels": n_ch,
        "audio_bands": audio_in_channels,
        "min_lag_samples": args.min_lag_samples,
        "max_lag_samples": args.max_lag_samples,
        "epochs": args.epochs,
        "parameters": n_params,
        "best_val_acc": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "mean_tier1_zero_shot_5s": round(float(np.mean(cohort_t1_acc)), 2) if cohort_t1_acc else 0.0,
        "mean_tier1_false_switches": round(float(np.mean(cohort_t1_fsw)), 2) if cohort_t1_fsw else 0.0,
        "mean_tier2_adapted_5s": round(float(np.mean(cohort_t2_acc)), 2) if cohort_t2_acc else 0.0,
        "mean_tier2_raw_gain_pp": round(float(np.mean(cohort_t2_gain)), 2) if cohort_t2_gain else 0.0,
        "mean_tier2_false_switches": round(float(np.mean(cohort_t2_fsw)), 2) if cohort_t2_fsw else 0.0,
        "mean_tier3_leaky_contin_5s": round(float(np.mean(cohort_leaky_acc)), 2) if cohort_leaky_acc else 0.0,
        "mean_tier3_leaky_false_switches": round(float(np.mean(cohort_leaky_fsw)), 2) if cohort_leaky_fsw else 0.0,
        "mean_tier4_sticky_gated_5s": round(float(np.mean(cohort_t3_acc)), 2) if cohort_t3_acc else 0.0,
        "mean_tier4_useful_boost_cov": round(float(np.mean(cohort_t3_cov)), 2) if cohort_t3_cov else 0.0,
        "mean_tier4_false_switches": round(float(np.mean(cohort_t3_fsw)), 2) if cohort_t3_fsw else 0.0,
        "mean_multiscale_10s": round(float(np.mean(cohort_10s)), 2) if cohort_10s else 0.0,
        "mean_multiscale_20s": round(float(np.mean(cohort_20s)), 2) if cohort_20s else 0.0,
        "subject_accuracies": subject_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    metrics_path = resolve_output_path(args.output_metrics)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OUTPUT] Model checkpoint saved to: {resolve_output_path(args.output_model)}")
    print(f"[OUTPUT] Comprehensive metrics saved to: {metrics_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Band Cochlear Gammatone + CA-TCN Training")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", choices=list(MONTAGES.keys()))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (exit if validation accuracy fails to improve for N epochs)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--audio_bands", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--min_lag_samples", type=int, default=-2, help="Minimum lag samples (-2 = -31.25 ms)")
    parser.add_argument("--max_lag_samples", type=int, default=18, help="Maximum lag samples (18 = +281.25 ms)")
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "factored"], help="Cross-correlation classification head type ('linear' unconstrained vs 'factored' low-rank)")
    parser.add_argument("--eeg_lowcut", type=float, default=1.0, help="EEG bandpass low cutoff in Hz (default: 1.0)")
    parser.add_argument("--eeg_highcut", type=float, default=6.5, help="EEG bandpass high cutoff in Hz (default: 6.5, matches SincNet passband and suppresses temporalis EMG noise)")
    parser.add_argument("--audio_lowpass", type=float, default=8.0, help="Audio envelope lowpass cutoff in Hz (default: 8.0)")
    parser.add_argument("--loss", type=str, default="softplus", choices=["softplus", "hinge"], help="Loss function (default: softplus to eliminate dead-zone)")
    parser.add_argument("--loss_temp", type=float, default=0.5, help="Temperature for softplus logistic loss")
    parser.add_argument("--margin", type=float, default=0.35, help="Separation margin between attended and unattended streams")
    parser.add_argument("--use_sinc", action="store_true", default=True, help="Enable biological SincNet filterbank (Sinc-CATCN v2)")
    parser.add_argument("--no_sinc", action="store_false", dest="use_sinc", help="Disable SincNet (revert to legacy baseline)")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--head_dropout", type=float, default=0.35, help="Dropout on cross-correlation head features to prevent memorization")
    parser.add_argument("--subband_mask_prob", type=float, default=0.30)
    parser.add_argument("--time_mask_prob", type=float, default=0.25, help="Temporal SpecAugment (mask 250ms audio chunk to prevent story memorization)")
    parser.add_argument("--channel_mask_prob", type=float, default=0.25, help="Probability of masking 1 random EEG channel during training")
    parser.add_argument("--include_broadband", action="store_true", default=True, help="Include 1D broadband envelope as Channel 0 alongside 8 Gammatone subbands")
    parser.add_argument("--no_broadband", action="store_false", dest="include_broadband", help="Disable broadband envelope inclusion")
    parser.add_argument("--adapt", action="store_true", default=True, help="Enable few-shot spatial adaptation")
    parser.add_argument("--no_adapt", action="store_false", dest="adapt", help="Disable few-shot spatial adaptation")
    parser.add_argument("--deep_adapt", action="store_true", default=False, help="Enable deep adaptation of SincNet frequency cutoffs and LayerNorms")
    parser.add_argument("--no_deep_adapt", action="store_false", dest="deep_adapt", help="Disable deep adaptation (adapt only linear spatial matrix)")
    parser.add_argument("--deep_adapt_lr", type=float, default=5e-4, help="Learning rate for SincNet and LayerNorm parameters during deep adaptation")
    parser.add_argument("--calib_trials", type=int, default=12, help="Number of calibration trials for few-shot spatial adaptation")
    parser.add_argument("--calib_epochs", type=int, default=15, help="Few-shot spatial calibration epochs for 64-parameter adapter")
    parser.add_argument("--calib_lr", type=float, default=1e-3, help="Learning rate for 64-parameter spatial adapter")
    parser.add_argument("--l2_identity", type=float, default=0.05, help="Frobenius identity shrinkage regularization weight")
    parser.add_argument("--gate_alpha", type=float, default=0.85, help="Exponential moving average factor for Sticky Hysteresis Gate")
    parser.add_argument("--gate_switch", type=float, default=0.20, help="Switching margin threshold theta_switch (default: 0.20)")
    parser.add_argument("--gate_maintain", type=float, default=0.08, help="Retention margin threshold theta_maintain (default: 0.08)")
    parser.add_argument("--gate_step_sec", type=float, default=0.5, help="Streaming step size in seconds (2 Hz control rate)")
    parser.add_argument("--dual_band", action="store_true", default=False, help="Extract dual-band EEG (1-6.5 Hz ERP + 8-13 Hz Alpha lateralization band)")
    parser.add_argument("--alpha_lowcut", type=float, default=8.0, help="Alpha bandpass low cutoff in Hz (default: 8.0)")
    parser.add_argument("--alpha_highcut", type=float, default=13.0, help="Alpha bandpass high cutoff in Hz (default: 13.0)")
    parser.add_argument("--include_onsets", action="store_true", default=False, help="Include 8-band acoustic half-wave rectified onset features")
    parser.add_argument("--spatial_loss_weight", type=float, default=0.0, help="Weight for auxiliary spatial direction BCE classification loss (default: 0.0)")
    parser.add_argument("--spatial_weight", type=float, default=0.0, help="Fusion weight for spatial direction margin during testing (default: 0.0)")
    parser.add_argument("--use_leaky_integration", action="store_true", default=True, help="Enable continuous leaky cumulative decision integration")
    parser.add_argument("--no_leaky_integration", action="store_false", dest="use_leaky_integration", help="Disable continuous leaky cumulative integration")
    parser.add_argument("--leaky_gamma", type=float, default=0.90, help="Decay factor gamma for continuous leaky cumulative decision integration (0.90 = ~3.3s half-life)")
    parser.add_argument("--gate_type", type=str, default="sticky", choices=["sticky", "hmm"], help="Decision gate type: 'sticky' (Sticky Hysteresis) vs 'hmm' (Bayesian HMM Forward Filter)")
    parser.add_argument("--hmm_mu", type=float, default=0.35, help="HMM Gaussian emission mean mu")
    parser.add_argument("--hmm_sigma", type=float, default=0.50, help="HMM Gaussian emission std sigma")
    parser.add_argument("--hmm_switch_prior", type=float, default=0.015, help="HMM state transition switch prior probability")
    parser.add_argument("--hmm_threshold", type=float, default=0.70, help="HMM decision confidence threshold")
    parser.add_argument("--contrastive_weight", type=float, default=0.0, help="InfoNCE cross-modal alignment loss weight (default: 0.0)")
    parser.add_argument("--contrastive_temp", type=float, default=0.1, help="InfoNCE temperature tau (default: 0.1)")
    parser.add_argument("--eeg_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--audio_env_file", type=str, default=None)
    parser.add_argument("--arch", type=str, default="conformer", choices=["conformer", "neuroconformer", "msca", "sinc", "baseline"], help="Model architecture: 'conformer' (v4 Dual-Stream Cross-Modal Neuro-Conformer), 'msca' (v3), 'sinc' (v2), 'baseline' (v1)")
    parser.add_argument("--subsample_stride", type=int, default=2, help="Temporal subsampling stride for Conformer attention (2 = 32Hz, 4x speedup, 1 = unstrided 64Hz)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Pre-trained checkpoint to load")
    parser.add_argument("--eval_only", action="store_true", help="Skip backbone training and execute adaptation and multi-tier benchmark directly")
    parser.add_argument("--streaming_context", action="store_true", help="Enable continuous streaming context for multi-scale 10s and 20s windows")
    parser.add_argument("--output_model", type=str, default="/kaggle/working/sinc_multiband_catcn_best.pt")
    parser.add_argument("--output_metrics", type=str, default="/kaggle/working/sinc_multiband_catcn_metrics.json")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid CPU smoke test")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects to run, or 'all'")
    args = parser.parse_args()
    run_multiband_training(args)
