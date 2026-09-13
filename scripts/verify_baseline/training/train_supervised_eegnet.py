import argparse
import sys
import os
import json
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_baseline.models.eegnet_classifier import EEGNet_Classifier
from scripts.verify_baseline.baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out

FS = 64
DECISION_WINDOW_SEC = 10
TRAIN_WINDOW_SEC = 5
TRAIN_HOP_SEC = 2

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyq = 0.5 * fs
    highcut = min(highcut, nyq - 0.1)
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def prepare_dataset_supervised(examples, lowcut, highcut):
    X = []
    Y = []
    
    for ex in examples:
        eeg = ex.eeg.T # [Channels, Time]
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T 
        
        X.append(x_norm)
        
        # DTU Dataset Label Convention
        # 1: Attended A (Left)
        # 2: Attended B (Right)
        if ex.label == 1:
            Y.append(0.0)
        elif ex.label == 2:
            Y.append(1.0)
        else:
            raise ValueError(f"Unknown label: {ex.label}")
            
    return X, Y

def chunk_trial_supervised(x, y, window_sec, hop_sec):
    win_samples = int(window_sec * FS)
    hop_samples = int(hop_sec * FS)
    
    chunks_x, chunks_y = [], []
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        chunks_y.append(y)
        start += hop_samples
        
    return chunks_x, chunks_y

def evaluate_model_supervised(model, X, Y, device, window_sec=10):
    model.eval()
    window_samples = int(window_sec * FS)
    n_correct = 0.0
    n_total = 0
    
    with torch.no_grad():
        for i in range(len(X)):
            x_np = X[i]
            y_true = Y[i]
            
            start = 0
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                
                x_chunk = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                logit = model(x_chunk)
                pred = (torch.sigmoid(logit) > 0.5).float().item()
                
                if pred == y_true:
                    n_correct += 1.0
                    
                n_total += 1
                start += window_samples
                
    return n_correct, n_total

def get_majority_accuracy(Y_train, Y_test):
    # Y_train and Y_test are lists of floats 0.0 or 1.0
    sum_y = sum(Y_train)
    majority_class = 1.0 if sum_y > (len(Y_train) / 2) else 0.0
    
    correct = sum([1 for y in Y_test if y == majority_class])
    return correct / max(len(Y_test), 1), majority_class

def train_supervised_loso(lowcut, highcut, batch_size=128, num_workers=2, subjects_to_run=None):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | 64-Channel Supervised EEGNet Classifier")
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    if subjects_to_run:
        all_paths = [p for p in all_paths if p.stem in subjects_to_run]
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    folds = list(iter_leave_one_subject_out(all_paths))
    
    os.makedirs("checkpoints_supervised", exist_ok=True)
    all_accs_dict = {}
    
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        print(f"\nEvaluating fold with held-out subject: {held_out_path.stem}")
        print(f"  [Memory] Pre-fold RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")
        
        train_exs = []
        for p in train_paths:
            train_exs.extend(subject_examples[str(p)])
            
        test_exs = subject_examples[held_out_key]
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs = train_exs[:val_split]
        train_exs = train_exs[val_split:]
        
        X_tr_full, Y_tr_full = [], []
        X_va_full, Y_va_full = [], []
        
        for p in train_paths:
            tX, tY = prepare_dataset_supervised(subject_examples[str(p)], lowcut, highcut)
            v_split_idx = int(0.1 * len(tX))
            X_va_full.extend(tX[:v_split_idx])
            Y_va_full.extend(tY[:v_split_idx])
            X_tr_full.extend(tX[v_split_idx:])
            Y_tr_full.extend(tY[v_split_idx:])

        X_te_full, Y_te_full = prepare_dataset_supervised(test_exs, lowcut, highcut)
        
        # Calculate Majority Baseline (on trials)
        maj_acc, maj_class = get_majority_accuracy(Y_tr_full, Y_te_full)
        print(f"--- BASELINE: Majority Class is {maj_class} | Test Acc: {maj_acc*100:.2f}% ---")
        
        # Chunk training data
        X_tr, Y_tr = [], []
        for i in range(len(X_tr_full)):
            cx, cy = chunk_trial_supervised(X_tr_full[i], Y_tr_full[i], TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)
            X_tr.extend(cx)
            Y_tr.extend(cy)
            
        print("Converting to PyTorch Dataset...")
        X_tr_t = torch.FloatTensor(np.stack(X_tr))
        Y_tr_t = torch.FloatTensor(np.stack(Y_tr))
        
        train_dataset = TensorDataset(X_tr_t, Y_tr_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=(num_workers > 0)
        )
            
        # Model
        num_channels = X_tr_full[0].shape[0]
        model = EEGNet_Classifier(in_channels=num_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 10
        epochs_no_improve = 0
        
        print(f"Training on {len(X_tr)} chunks ({TRAIN_WINDOW_SEC}s) | Batch Size: {batch_size} | Workers: {num_workers}...")
        
        for epoch in range(100):
            model.train()
            train_loss = 0.0
            
            for bx, by in train_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(bx)
                        loss = criterion(logits, by)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                
            nc_va, nt_va = evaluate_model_supervised(model, X_va_full, Y_va_full, device, window_sec=10)
            val_acc = nc_va / max(nt_va, 1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            num_batches = max(len(train_loader), 1)
            avg_loss = train_loss / num_batches
            
            print(f"  Epoch {epoch+1:02d}/100 | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Patience: {epochs_no_improve}/10")
                
            if epochs_no_improve >= patience:
                break
                
        # Evaluation
        model.load_state_dict(best_weights)
        print(f"  [Evaluation - 10s Window]")
        w_sec = 10
        nc_norm, nt_norm = evaluate_model_supervised(model, X_te_full, Y_te_full, device, window_sec=w_sec)
        acc_norm = nc_norm / max(nt_norm, 1)
        
        print(f"    -> Window {w_sec:2d}s | Test Acc: {acc_norm*100:.2f}% (Majority: {maj_acc*100:.2f}%)")
        
        if w_sec not in all_accs_dict:
            all_accs_dict[w_sec] = []
        all_accs_dict[w_sec].append(acc_norm)
        
        del X_tr, Y_tr, X_tr_full, Y_tr_full, X_va_full, Y_va_full, X_te_full, Y_te_full
        gc.collect()
        
    print("\n" + "="*50)
    print(f"[SUPERVISED EEGNET 64-CHANNEL CANONICAL EVALUATION (10s)]")
    print("="*50)
    for w_sec in sorted(all_accs_dict.keys()):
        final_acc_norm = np.mean(all_accs_dict[w_sec])
        print(f" Window {w_sec:2d}s | Test Acc: {final_acc_norm*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Supervised 64-Channel EEGNet")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=32.0, help="Higher default for full EEG band")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader num_workers")
    parser.add_argument("--subjects", type=str, nargs='+', default=None, help="Specific subjects to run")
    args = parser.parse_args()
    
    train_supervised_loso(args.lowcut, args.highcut, args.batch_size, args.num_workers, args.subjects)
