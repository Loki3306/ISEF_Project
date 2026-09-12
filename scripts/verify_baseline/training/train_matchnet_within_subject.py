import argparse
import sys
import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import psutil
import gc
from pathlib import Path
from copy import deepcopy
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet, contrastive_loss, anchored_contrastive_loss
from baselines.ridge_aad import load_subject_examples, subject_files

FS = 64
DECISION_WINDOW_SEC = 10
TRAIN_WINDOW_SEC = 5
TRAIN_HOP_SEC = 2
NUM_BANDS = 28

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def get_mapping_data():
    base_dir = Path("/kaggle/input")
    
    # 1. Find audio_mapping.json
    map_files = list(base_dir.rglob("audio_mapping.json"))
    if map_files:
        map_file = map_files[0]
    else:
        map_file = REPO_ROOT / "data" / "audio_mapping.json"
        
    # 2. Find gammatone envelopes pkl
    pkl_files = list(base_dir.rglob("*gammatone*.pkl"))
    if not pkl_files:
        pkl_files = list(base_dir.rglob("*.pkl"))
        
    if pkl_files:
        env_file = pkl_files[0]
    else:
        env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        
    print(f"Using map file: {map_file}")
    print(f"Using env file: {env_file}")
    
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def prepare_dataset(examples, channels, lowcut, highcut, subject_id, mapping, envelopes):
    X, Y_A, Y_B = [], [], []
    sub_key = subject_id.replace("_data_preproc", "")
    
    for i, ex in enumerate(examples):
        eeg = ex.eeg[:, channels].T
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T 
        
        trial_key = f"trial_{i}"
        if sub_key in mapping and trial_key in mapping[sub_key]:
            fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
            fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
            env_a_full = envelopes[fname_a] 
            env_b_full = envelopes[fname_b] 
        else:
            print(f"Warning: Missing mapping for {sub_key} {trial_key}")
            continue
            
        min_len = min(x_norm.shape[1], env_a_full.shape[1])
        x_norm = x_norm[:, :min_len]
        env_a = env_a_full[:, :min_len]
        env_b = env_b_full[:, :min_len]
        
        env_a = normalize_array(env_a.T).T
        env_b = normalize_array(env_b.T).T
        
        X.append(x_norm)
        Y_A.append(env_a)
        Y_B.append(env_b)
        
    return X, Y_A, Y_B

def chunk_trial(x, ya, yb, window_sec, hop_sec):
    win_samples = int(window_sec * FS)
    hop_samples = int(hop_sec * FS)
    chunks_x, chunks_ya, chunks_yb = [], [], []
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        chunks_ya.append(ya[:, start:end])
        chunks_yb.append(yb[:, start:end])
        start += hop_samples
    return chunks_x, chunks_ya, chunks_yb

def pearson_corr(x, y, dim=1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

def evaluate_model(model, X, Y_A, Y_B, device, window_sec=10, metric="cosine", detailed_metrics=False):
    model.eval()
    window_samples = int(window_sec * FS)
    
    metrics = {
        "n_correct": 0.0,
        "n_correct_abs": 0.0,
        "n_total": 0,
        "r_correct_list": [],
        "r_incorrect_list": []
    }
    
    with torch.no_grad():
        for i in range(len(X)):
            x_np, ya_np, yb_np = X[i], Y_A[i], Y_B[i]
            start = 0
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                
                x_chunk = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                ya_chunk = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                yb_chunk = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                
                if metric == "pearson":
                    sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                    sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                else:
                    sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                    sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                if detailed_metrics:
                    metrics["r_correct_list"].append(sim_a)
                    metrics["r_incorrect_list"].append(sim_b)
                
                if sim_a > sim_b:
                    metrics["n_correct"] += 1.0
                elif sim_a == sim_b:
                    metrics["n_correct"] += 0.5
                    
                if abs(sim_a) > abs(sim_b):
                    metrics["n_correct_abs"] += 1.0
                elif abs(sim_a) == abs(sim_b):
                    metrics["n_correct_abs"] += 0.5
                    
                metrics["n_total"] += 1
                start += window_samples
                
    if detailed_metrics:
        return metrics
    return metrics["n_correct"], metrics["n_total"]

def train_matchnet_within_subject(eeg_model, channels, lowcut, highcut, batch_size=128, num_workers=2, subjects_to_run=None, loss_type="contrastive", lambda_align=0.5, align_target=0.1, smoke_test=False):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | MatchNet ({eeg_model}) Within-Subject | Channels: {channels}")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    if not all_paths:
        print("No subjects found.")
        return
        
    if smoke_test:
        smoke_subjects = ["S5_data_preproc", "S6_data_preproc", "S8_data_preproc", "S10_data_preproc"]
        all_paths = [p for p in all_paths if p.stem in smoke_subjects]
    elif subjects_to_run:
        all_paths = [p for p in all_paths if p.stem in subjects_to_run]
        
    os.makedirs(REPO_ROOT / "checkpoints", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "experiments", exist_ok=True)
    
    loso_e0_baseline = {
        "S1": 0.5833, "S10": 0.4833, "S11": 0.5500, "S12": 0.6500, 
        "S13": 0.5833, "S14": 0.5666, "S15": 0.5833, "S16": 0.5166, 
        "S17": 0.7500, "S18": 0.4666, "S2": 0.5166, "S3": 0.5333, 
        "S4": 0.5333, "S5": 0.5833, "S6": 0.4333, "S7": 0.5500, 
        "S8": 0.4500, "S9": 0.5666
    }
    
    all_subject_metrics = {}
    detailed_logs = []
    
    for subject_path in all_paths:
        subject_id = subject_path.stem
        sub_key = subject_id.replace("_data_preproc", "")
        print(f"\n{'='*50}\nEvaluating Subject: {subject_id}\n{'='*50}")
        print(f"  [Memory] Pre-subject RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")
        
        all_exs = load_subject_examples(subject_path)
        if len(all_exs) == 0:
            continue
            
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        fold_accs_abs = []
        subject_r_correct = []
        subject_r_incorrect = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_exs)):
            print(f"\n  --- Fold {fold_idx+1}/5 ---")
            
            # Use strict index tracking to prevent any temporal leakage
            train_pool_idx = list(train_idx)
            test_idx_list = list(test_idx)
            
            # Split 10% of train_pool for validation early stopping
            np.random.seed(42 + fold_idx)
            np.random.shuffle(train_pool_idx)
            val_split = max(1, int(0.1 * len(train_pool_idx)))
            
            val_idx_list = train_pool_idx[:val_split]
            train_final_idx_list = train_pool_idx[val_split:]
            
            train_exs = [all_exs[i] for i in train_final_idx_list]
            val_exs = [all_exs[i] for i in val_idx_list]
            test_exs = [all_exs[i] for i in test_idx_list]
            
            print(f"  Train trials: {len(train_exs)} | Val trials: {len(val_exs)} | Test trials: {len(test_exs)}")
            print(f"  [IDs] Train: {sorted(train_final_idx_list)}")
            print(f"  [IDs] Val:   {sorted(val_idx_list)}")
            print(f"  [IDs] Test:  {sorted(test_idx_list)}")
            
            X_tr_full, YA_tr_full, YB_tr_full = prepare_dataset(train_exs, channels, lowcut, highcut, subject_id, mapping, envelopes)
            X_va_full, YA_va_full, YB_va_full = prepare_dataset(val_exs, channels, lowcut, highcut, subject_id, mapping, envelopes)
            X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, channels, lowcut, highcut, subject_id, mapping, envelopes)
            
            # Chunk training data
            X_tr, YA_tr, YB_tr = [], [], []
            for i in range(len(X_tr_full)):
                cx, cya, cyb = chunk_trial(X_tr_full[i], YA_tr_full[i], YB_tr_full[i], TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)
                X_tr.extend(cx); YA_tr.extend(cya); YB_tr.extend(cyb)
                
            X_tr_t = torch.FloatTensor(np.stack(X_tr))
            YA_tr_t = torch.FloatTensor(np.stack(YA_tr))
            YB_tr_t = torch.FloatTensor(np.stack(YB_tr))
            
            train_dataset = TensorDataset(X_tr_t, YA_tr_t, YB_tr_t)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=True, 
                persistent_workers=(num_workers > 0)
            )
                
            model = ContrastiveMatchNet(eeg_model_type=eeg_model, eeg_channels=len(channels), audio_channels=NUM_BANDS, latent_dim=64).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler()
            
            best_val_acc = 0.0
            best_epoch = 0
            best_weights = deepcopy(model.state_dict())
            patience = 10
            epochs_no_improve = 0
            
            for epoch in range(100):
                model.train()
                train_loss, train_sa, train_sb = 0.0, 0.0, 0.0
                
                for bx, bya, byb in train_loader:
                    bx = bx.to(device, non_blocking=True)
                    bya = bya.to(device, non_blocking=True)
                    byb = byb.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        z_eeg, z_a, z_b = model(bx, bya, byb)
                        if loss_type == "anchored":
                            loss, sa, sb = anchored_contrastive_loss(z_eeg, z_a, z_b, margin=0.1, lambda_align=lambda_align, align_target=align_target)
                        else:
                            loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    train_sa += sa.item()
                    train_sb += sb.item()
                    
                nc_va, nt_va = evaluate_model(model, X_va_full, YA_va_full, YB_va_full, device, window_sec=10, metric="pearson")
                val_acc = nc_va / max(nt_va, 1)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_weights = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    break
                    
            model.load_state_dict(best_weights)
            
            # Evaluate Fold Test Set
            metrics_te = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=10, metric="pearson", detailed_metrics=True)
            test_acc = metrics_te["n_correct"] / max(metrics_te["n_total"], 1)
            test_acc_abs = metrics_te["n_correct_abs"] / max(metrics_te["n_total"], 1)
            
            fold_accs.append(test_acc)
            fold_accs_abs.append(test_acc_abs)
            subject_r_correct.extend(metrics_te["r_correct_list"])
            subject_r_incorrect.extend(metrics_te["r_incorrect_list"])
            
            print(f"  -> Fold {fold_idx+1} Best Epoch: {best_epoch+1} (Val Acc: {best_val_acc*100:.2f}%)")
            print(f"  -> Fold {fold_idx+1} Test Acc (Signed): {test_acc*100:.2f}% | (Abs): {test_acc_abs*100:.2f}%")
            
            # Log exact trial-level metadata
            detailed_logs.append({
                "subject": subject_id,
                "fold": fold_idx + 1,
                "train_trial_ids": [int(x) for x in sorted(train_final_idx_list)],
                "val_trial_ids": [int(x) for x in sorted(val_idx_list)],
                "test_trial_ids": [int(x) for x in sorted(test_idx_list)],
                "best_epoch": int(best_epoch + 1),
                "best_val_accuracy": float(best_val_acc),
                "test_accuracy": float(test_acc),
                "test_accuracy_abs": float(test_acc_abs)
            })
            
            # Cleanup memory per fold
            del X_tr, YA_tr, YB_tr, X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full
            gc.collect()
            
        subj_mean_acc = np.mean(fold_accs)
        subj_mean_abs_acc = np.mean(fold_accs_abs)
        
        r_correct_arr = np.array(subject_r_correct)
        median_r_corr = float(np.median(r_correct_arr)) if len(r_correct_arr) > 0 else 0.0
        median_abs_r_corr = float(np.median(np.abs(r_correct_arr))) if len(r_correct_arr) > 0 else 0.0
        neg_frac = float(np.mean(r_correct_arr < 0)) if len(r_correct_arr) > 0 else 0.0
        
        all_subject_metrics[subject_id] = {
            "signed_acc": subj_mean_acc,
            "abs_acc": subj_mean_abs_acc,
            "median_r_corr": median_r_corr,
            "median_abs_r_corr": median_abs_r_corr,
            "neg_frac": neg_frac,
            "r_correct_array": [float(x) for x in subject_r_correct],
            "r_incorrect_array": [float(x) for x in subject_r_incorrect]
        }
        
        print(f"\n  [RESULT] {subject_id} Metrics:")
        print(f"    - Signed Acc:     {subj_mean_acc*100:.2f}%")
        print(f"    - Abs Acc:        {subj_mean_abs_acc*100:.2f}%")
        print(f"    - Median r_corr:  {median_r_corr:.4f}")
        print(f"    - Median |r_corr|:{median_abs_r_corr:.4f}")
        print(f"    - Neg Fraction:   {neg_frac*100:.1f}%")
        
        # Save intermediate results
        out_data = {
            "summary_metrics": all_subject_metrics,
            "detailed_logs": detailed_logs
        }
        with open(PROJECT_ROOT / "experiments" / "matchnet_within_subject_forensics.json", "w") as f:
            json.dump(out_data, f, indent=4)
            
    print("\n" + "="*100)
    print(f"[MATCHNET ({eeg_model.upper()}) CORRELATION-SIGN FORENSIC ANALYSIS]")
    print("="*100)
    print(f"{'Subject':<12} | {'Signed Acc':<10} | {'Abs Acc':<10} | {'med(r_corr)':<12} | {'med(|r_corr|)':<14} | {'Neg Frac':<10}")
    print("-" * 100)
    
    overall_signed = []
    overall_abs = []
    
    for subj in sorted(all_subject_metrics.keys()):
        metrics = all_subject_metrics[subj]
        acc = metrics["signed_acc"]
        abs_acc = metrics["abs_acc"]
        med_r = metrics["median_r_corr"]
        med_abs_r = metrics["median_abs_r_corr"]
        neg_frac = metrics["neg_frac"]
        
        overall_signed.append(acc)
        overall_abs.append(abs_acc)
            
        print(f"{subj:<12} | {acc*100:>9.2f}% | {abs_acc*100:>9.2f}% | {med_r:>12.4f} | {med_abs_r:>14.4f} | {neg_frac*100:>8.1f}%")
        
    print("-" * 100)
    mean_signed = np.mean(overall_signed)
    mean_abs = np.mean(overall_abs)
    
    print(f"{'MEAN':<12} | {mean_signed*100:>9.2f}% | {mean_abs*100:>9.2f}% | {'-':>12} | {'-':>14} | {'-':>9}")
    print("="*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Contrastive MatchNet Within-Subject")
    parser.add_argument("--model", type=str, default="eegnet", choices=["eegnet", "atcnet", "eegnet_s1", "eegnet_s2"], help="Base EEG encoder")
    parser.add_argument("--channels", type=int, nargs='+', default=[0, 33, 6, 41, 22, 59, 15, 52], help="EEG channel indices to use")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader num_workers")
    parser.add_argument("--subjects", type=str, nargs='+', default=None, help="Specific subjects to run")
    parser.add_argument("--loss", type=str, default="contrastive", choices=["contrastive", "anchored"], help="Loss function to use")
    parser.add_argument("--lambda_align", type=float, default=0.5, help="Lambda for alignment penalty")
    parser.add_argument("--align_target", type=float, default=0.1, help="Positive alignment target for anchored loss")
    parser.add_argument("--smoke_test", action="store_true", help="Run only on S5, S6, S8, S10 for fast iteration")
    args = parser.parse_args()
    
    train_matchnet_within_subject(args.model, args.channels, args.lowcut, args.highcut, args.batch_size, args.num_workers, args.subjects, args.loss, args.lambda_align, args.align_target, args.smoke_test)
