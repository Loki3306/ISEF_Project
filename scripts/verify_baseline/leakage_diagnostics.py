import os
import sys
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter
import scipy.stats as stats

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_loso import normalize_array, butter_bandpass_filter, pearson_corr

FS = 64

def get_mapping_data():
    base_dir = Path("/kaggle/input")
    map_files = list(base_dir.rglob("audio_mapping.json"))
    map_file = map_files[0] if map_files else REPO_ROOT / "data" / "audio_mapping.json"
        
    pkl_files = list(base_dir.rglob("*gammatone*.pkl"))
    if not pkl_files: pkl_files = list(base_dir.rglob("*.pkl"))
    env_file = pkl_files[0] if pkl_files else REPO_ROOT / "data" / "gammatone_envelopes.pkl"
    
    with open(map_file, 'r') as f: mapping = json.load(f)
    with open(env_file, 'rb') as f: envelopes = pickle.load(f)
    return mapping, envelopes

def run_diagnostics():
    print("============================================================")
    print("            LEAKAGE DIAGNOSTICS & CONTROLS")
    print("============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We will initialize a model. If there is a trained checkpoint for S2, we should ideally load it.
    # Since we are running on Kaggle, the user might have to run train_matchnet_loso.py first,
    # or we can evaluate the exact leakage structure mathematically without the model, 
    # but ChatGPT requested to evaluate if model predictions correlate with r(A) - r(B).
    # For now, we will compute r(A) and r(B) exactly.
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    # We use S2 as the default test subject as requested by GPT
    s2_path = next((p for p in all_paths if 'S2_data_preproc' in p.name), None)
    if not s2_path: s2_path = all_paths[0]
    sub_key = s2_path.stem.replace("_data_preproc", "")
    
    print(f"Held-out Subject (Test): {sub_key}")
    
    # 1. Calculate Training Role Probability r(f)
    # r(f) = P(file appears as A | f, training)
    train_a_counts = Counter()
    train_total_counts = Counter()
    
    for train_sub, sub_data in mapping.items():
        if train_sub == sub_key: continue
        for trial in sub_data.values():
            fa = trial['wavA']['filename']
            fb = trial['wavB']['filename']
            train_a_counts[fa] += 1
            train_total_counts[fa] += 1
            train_total_counts[fb] += 1
            
    r_f = {}
    for f in train_total_counts.keys():
        r_f[f] = train_a_counts[f] / train_total_counts[f]
        
    print("\n--- 1. ROLE FREQUENCY CORRELATION ---")
    
    examples = load_subject_examples(s2_path)
    X, Y_A, Y_B, filenames_A, filenames_B = [], [], [], [], []
    
    channels = [13, 46, 43, 23, 50, 0, 52, 14]
    
    for idx, ex in enumerate(examples):
        trial_key = f"trial_{idx}"
        if trial_key not in mapping[sub_key]: continue
        
        fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
        fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
        
        eeg = ex.eeg[:, channels].T
        eeg = butter_bandpass_filter(eeg, 1.0, 6.0, FS, axis=1)
        x_norm = normalize_array(eeg.T).T 
        
        env_a = envelopes[fname_a]
        env_b = envelopes[fname_b]
        
        min_len = min(x_norm.shape[1], env_a.shape[1], env_b.shape[1])
        X.append(x_norm[:, :min_len])
        Y_A.append(normalize_array(env_a[:, :min_len].T).T)
        Y_B.append(normalize_array(env_b[:, :min_len].T).T)
        filenames_A.append(fname_a)
        filenames_B.append(fname_b)
        
    # We will use the model to generate predictions. If untrained, it's a null model, but we 
    # expect the user to integrate this or train first. We will instantiate a random model
    # but the user should run it after training if they want the exact Pearson R.
    model = ContrastiveMatchNet(eeg_channels=8, audio_channels=28).to(device)
    model.eval()
    
    # Try to load a checkpoint if it exists
    chkpt_path = REPO_ROOT / "checkpoints" / "baseline" / "msca" / f"{sub_key}_model.pt"
    if chkpt_path.exists():
        model.load_state_dict(torch.load(chkpt_path, map_location=device))
        print(f"Loaded trained model for {sub_key}")
    else:
        print(f"WARNING: No trained model found at {chkpt_path}. Using random weights.")
    
    predictions = []
    r_diffs = []
    
    n_correct = 0
    window_samples = int(10.0 * FS)
    
    # Evaluate all chunks
    with torch.no_grad():
        for i in range(len(X)):
            x_np = X[i]
            ya_np = Y_A[i]
            yb_np = Y_B[i]
            fa = filenames_A[i]
            fb = filenames_B[i]
            
            # Historical training-role difference
            r_diff = r_f.get(fa, 0.5) - r_f.get(fb, 0.5)
            
            start = 0
            chunk_sim_a = []
            chunk_sim_b = []
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                xc = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                yac = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                ybc = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                ze, za, zb = model(xc, yac, ybc)
                sa = F.cosine_similarity(ze, za, dim=1).mean().item()
                sb = F.cosine_similarity(ze, zb, dim=1).mean().item()
                
                chunk_sim_a.append(sa)
                chunk_sim_b.append(sb)
                start += window_samples
                
            if chunk_sim_a:
                mean_sa = np.mean(chunk_sim_a)
                mean_sb = np.mean(chunk_sim_b)
                pred_margin = mean_sa - mean_sb
                
                predictions.append(pred_margin)
                r_diffs.append(r_diff)
                
                if mean_sa > mean_sb:
                    n_correct += 1
                    
    print(f"Normal Accuracy: {n_correct / len(X) * 100:.2f}%")
    
    if predictions:
        corr, pval = stats.pearsonr(r_diffs, predictions)
        print(f"Pearson Correlation between r(A)-r(B) and Decision Margin: {corr:.4f} (p={pval:.4e})")
        
        if corr > 0.3:
            print(">>> SMOKING GUN: Model decisions are heavily correlated with historical training frequencies! <<<")
            
    print("\n--- 2. TRUE EEG PERMUTATION CONTROL ---")
    # E_X -> A_Y, B_Y
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(X))
    while np.any(shuffle_indices == np.arange(len(X))):
        shuffle_indices = np.random.permutation(len(X))
        
    n_correct_eeg_perm = 0
    with torch.no_grad():
        for i in range(len(X)):
            x_np = X[shuffle_indices[i]] # Shuffled EEG
            ya_np = Y_A[i]               # Intact Audio Pair
            yb_np = Y_B[i]
            
            start = 0
            chunk_sim_a = []
            chunk_sim_b = []
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                xc = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                yac = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                ybc = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                ze, za, zb = model(xc, yac, ybc)
                sa = F.cosine_similarity(ze, za, dim=1).mean().item()
                sb = F.cosine_similarity(ze, zb, dim=1).mean().item()
                chunk_sim_a.append(sa)
                chunk_sim_b.append(sb)
                start += window_samples
                
            if chunk_sim_a:
                if np.mean(chunk_sim_a) > np.mean(chunk_sim_b):
                    n_correct_eeg_perm += 1
                    
    print(f"EEG Permutation Accuracy (Expected ~50%): {n_correct_eeg_perm / len(X) * 100:.2f}%")
    
    print("\n--- 3. AUDIO-ONLY NEURAL CLASSIFIER ---")
    # Feed Zeros to EEG, test accuracy
    n_correct_audio_only = 0
    with torch.no_grad():
        for i in range(len(X)):
            x_np = np.zeros_like(X[i]) # Audio-Only
            ya_np = Y_A[i]
            yb_np = Y_B[i]
            
            start = 0
            chunk_sim_a = []
            chunk_sim_b = []
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                xc = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                yac = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                ybc = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                ze, za, zb = model(xc, yac, ybc)
                sa = F.cosine_similarity(ze, za, dim=1).mean().item()
                sb = F.cosine_similarity(ze, zb, dim=1).mean().item()
                chunk_sim_a.append(sa)
                chunk_sim_b.append(sb)
                start += window_samples
                
            if chunk_sim_a:
                if np.mean(chunk_sim_a) > np.mean(chunk_sim_b):
                    n_correct_audio_only += 1
                    
    print(f"Audio-Only Classifier Accuracy (0 EEG) (Expected ~50% if unconfounded): {n_correct_audio_only / len(X) * 100:.2f}%")
    print("=== DIAGNOSTICS COMPLETE ===")

if __name__ == "__main__":
    run_diagnostics()
