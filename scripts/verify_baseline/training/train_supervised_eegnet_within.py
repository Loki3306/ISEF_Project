import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import psutil
import gc
from pathlib import Path
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_baseline.models.eegnet_classifier import EEGNet_Classifier
from scripts.verify_baseline.baselines.ridge_aad import load_subject_examples, subject_files
from scripts.verify_baseline.training.train_supervised_eegnet import prepare_dataset_supervised, chunk_trial_supervised, evaluate_model_supervised, get_majority_accuracy, FS, TRAIN_WINDOW_SEC, TRAIN_HOP_SEC

def train_supervised_within(lowcut, highcut, batch_size=128, num_workers=2, subjects_to_run=None):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | 64-Channel Supervised EEGNet Classifier (WITHIN-SUBJECT)")
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    if subjects_to_run:
        all_paths = [p for p in all_paths if p.stem in subjects_to_run]
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    all_accs_dict = {}
    
    for path in all_paths:
        subject_key = str(path)
        print(f"\nEvaluating Within-Subject for: {path.stem}")
        print(f"  [Memory] RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")
        
        exs = subject_examples[subject_key]
        
        # Shuffle trials for this subject
        np.random.seed(42)
        np.random.shuffle(exs)
        
        # Split: 80% Train, 10% Val, 10% Test
        num_trials = len(exs)
        train_end = int(0.8 * num_trials)
        val_end = int(0.9 * num_trials)
        
        train_exs = exs[:train_end]
        val_exs = exs[train_end:val_end]
        test_exs = exs[val_end:]
        
        X_tr_full, Y_tr_full = prepare_dataset_supervised(train_exs, lowcut, highcut)
        X_va_full, Y_va_full = prepare_dataset_supervised(val_exs, lowcut, highcut)
        X_te_full, Y_te_full = prepare_dataset_supervised(test_exs, lowcut, highcut)
        
        # Calculate Majority Baseline (on test trials)
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
        patience = 15
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
            
            print(f"  Epoch {epoch+1:02d}/100 | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Patience: {epochs_no_improve}/{patience}")
                
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
    print(f"[SUPERVISED EEGNET 64-CHANNEL WITHIN-SUBJECT EVALUATION (10s)]")
    print("="*50)
    for w_sec in sorted(all_accs_dict.keys()):
        final_acc_norm = np.mean(all_accs_dict[w_sec])
        print(f" Window {w_sec:2d}s | Average Test Acc: {final_acc_norm*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Supervised 64-Channel EEGNet (Within-Subject)")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=32.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subjects", type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    train_supervised_within(args.lowcut, args.highcut, args.batch_size, args.num_workers, args.subjects)
