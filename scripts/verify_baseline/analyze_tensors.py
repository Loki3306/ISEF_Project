import os
import sys
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_loso import normalize_array, butter_bandpass_filter

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

def run_tensor_audit():
    print("============================================================")
    print("              TENSOR & SYMMETRY AUDIT")
    print("============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init random baseline MSCA
    model = ContrastiveMatchNet(eeg_channels=8, audio_channels=28).to(device)
    model.eval()
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    s1_path = next((p for p in all_paths if 'S1_data_preproc' in p.name), all_paths[0])
    sub_key = s1_path.stem.replace("_data_preproc", "")
    examples = load_subject_examples(s1_path)
    
    print(f"Loaded Subject: {sub_key}")
    
    # Extract first trial data
    channels = [13, 46, 43, 23, 50, 0, 52, 14]
    ex = examples[0]
    eeg = ex.eeg[:, channels].T
    eeg = butter_bandpass_filter(eeg, 1.0, 6.0, FS, axis=1)
    x_norm = normalize_array(eeg.T).T 
    
    trial_key = "trial_0"
    fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
    fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
    
    env_a = envelopes[fname_a]
    env_b = envelopes[fname_b]
    
    # 10. Label Semantics Verification
    print("\n--- LABEL SEMANTICS VERIFICATION ---")
    print(f"Trial ID: {trial_key}")
    print(f"wavA: {fname_a} (Attended)")
    print(f"wavB: {fname_b} (Unattended)")
    print("Label conventionally used in evaluate_model: sim(EEG, A) > sim(EEG, B)")
    
    min_len = min(x_norm.shape[1], env_a.shape[1], env_b.shape[1])
    x_norm = x_norm[:, :min_len]
    env_a = env_a[:, :min_len]
    env_b = env_b[:, :min_len]
    
    # Normalization check
    env_a = normalize_array(env_a.T).T
    env_b = normalize_array(env_b.T).T
    
    # Take first 10s chunk
    chunk_len = 10 * FS
    x_chunk = torch.FloatTensor(x_norm[:, :chunk_len]).unsqueeze(0).to(device)
    ya_chunk = torch.FloatTensor(env_a[:, :chunk_len]).unsqueeze(0).to(device)
    yb_chunk = torch.FloatTensor(env_b[:, :chunk_len]).unsqueeze(0).to(device)
    
    print("\n--- 1. RAW TENSOR SYMMETRY & NaN AUDIT ---")
    print(f"EEG shape: {x_chunk.shape}")
    print(f"A shape: {ya_chunk.shape}")
    print(f"B shape: {yb_chunk.shape}")
    
    print(f"A mean/std: {ya_chunk.mean().item():.4f}, {ya_chunk.std().item():.4f}")
    print(f"B mean/std: {yb_chunk.mean().item():.4f}, {yb_chunk.std().item():.4f}")
    
    print(f"A NaN count: {torch.isnan(ya_chunk).sum().item()}")
    print(f"B NaN count: {torch.isnan(yb_chunk).sum().item()}")
    
    print(f"A Inf count: {torch.isinf(ya_chunk).sum().item()}")
    print(f"B Inf count: {torch.isinf(yb_chunk).sum().item()}")
    
    # Audio Encoder Output Symmetry
    print("\n--- 2. AUDIO ENCODER LATENT SYMMETRY ---")
    with torch.no_grad():
        za = model.audio_encoder(ya_chunk)
        zb = model.audio_encoder(yb_chunk)
        
        za_swapped = model.audio_encoder(yb_chunk)
        zb_swapped = model.audio_encoder(ya_chunk)
        
        print(f"za == zb_swapped? {torch.allclose(za, zb_swapped)}")
        print(f"zb == za_swapped? {torch.allclose(zb, za_swapped)}")
        
    print("\n--- 3. THE FUNDAMENTAL EXCHANGE SYMMETRY TEST ---")
    
    def evaluate_pair(e, a, b):
        z_e, z_a, z_b = model(e, a, b)
        
        # Exact logic from evaluate_model (cosine)
        sim_a = F.cosine_similarity(z_e, z_a, dim=1).mean()
        sim_b = F.cosine_similarity(z_e, z_b, dim=1).mean()
        return sim_a, sim_b, z_e, z_a, z_b
        
    with torch.no_grad():
        s_a, s_b, ze_orig, za_orig, zb_orig = evaluate_pair(x_chunk, ya_chunk, yb_chunk)
        s_a2, s_b2, _, _, _ = evaluate_pair(x_chunk, yb_chunk, ya_chunk)
        
        F_orig = s_a - s_b
        F_swapped = s_a2 - s_b2
        
        print(f"score_a: {s_a.item():.4f} | score_b: {s_b.item():.4f}")
        print(f"score_a_swapped: {s_a2.item():.4f} | score_b_swapped: {s_b2.item():.4f}")
        
        print(f"F(E, A, B) = {F_orig.item():.6f}")
        print(f"F(E, B, A) = {F_swapped.item():.6f}")
        
        sym_error = abs(F_orig.item() + F_swapped.item())
        print(f"Symmetry Error |F_orig + F_swapped| = {sym_error:.8f}")
        
        if not torch.allclose(F_orig, -F_swapped, atol=1e-5):
            print(">>> ERROR: SCORER IS MATHEMATICALLY ASYMMETRIC! <<<")
            
    print("\n--- 4. ZERO-EEG NaN COLLAPSE TEST ---")
    with torch.no_grad():
        x_zero = torch.zeros_like(x_chunk)
        s_a_zero, s_b_zero, ze_zero, za_zero, zb_zero = evaluate_pair(x_zero, ya_chunk, yb_chunk)
        
        print(f"ze_zero mean: {ze_zero.mean().item():.4f}, std: {ze_zero.std().item():.4f}")
        print(f"score_a_zero: {s_a_zero.item():.4f} | score_b_zero: {s_b_zero.item():.4f}")
        print(f"NaN count in ze_zero: {torch.isnan(ze_zero).sum().item()}")
        print(f"Is s_a_zero NaN?: {torch.isnan(s_a_zero).item()}")
        
        # Test Python > logic
        pred = s_a_zero > s_b_zero
        print(f"s_a_zero > s_b_zero evaluated to: {pred.item()}")
        
        # Check pearson_corr logic which uses normalization
        from training.train_matchnet_loso import pearson_corr
        p_a = pearson_corr(ze_zero, za_zero, dim=1).mean()
        p_b = pearson_corr(ze_zero, zb_zero, dim=1).mean()
        print(f"pearson_corr score_a: {p_a.item():.4f} | score_b: {p_b.item():.4f}")
        
        if torch.isnan(s_a_zero) or torch.isnan(p_a):
            print(">>> ERROR: ZERO EEG CAUSES NaN COLLAPSE! THIS EXPLAINS 1.33% ACCURACY! <<<")
            
    print("\n--- 5. MISMATCHED TRIAL SWAP TEST ---")
    # Get Trial 1 Audio
    ex1 = examples[1]
    fname_a1 = mapping[sub_key]["trial_1"]["wavA"]["filename"]
    fname_b1 = mapping[sub_key]["trial_1"]["wavB"]["filename"]
    env_a1 = normalize_array(envelopes[fname_a1][:, :chunk_len].T).T
    env_b1 = normalize_array(envelopes[fname_b1][:, :chunk_len].T).T
    
    ya_mismatch = torch.FloatTensor(env_a1).unsqueeze(0).to(device)
    yb_mismatch = torch.FloatTensor(env_b1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        s_a_m, s_b_m, _, _, _ = evaluate_pair(x_chunk, ya_mismatch, yb_mismatch)
        s_a_ms, s_b_ms, _, _, _ = evaluate_pair(x_chunk, yb_mismatch, ya_mismatch)
        
        F_mismatch = s_a_m - s_b_m
        F_mismatch_swapped = s_a_ms - s_b_ms
        
        print(f"F(E_0, A_1, B_1) = {F_mismatch.item():.6f}")
        print(f"F(E_0, B_1, A_1) = {F_mismatch_swapped.item():.6f}")
        print(f"Symmetry Error = {abs(F_mismatch.item() + F_mismatch_swapped.item()):.8f}")
        
    print("\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    run_tensor_audit()
