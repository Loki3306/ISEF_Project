import sys
import gc
import torch
import numpy as np
from pathlib import Path
import os
import subprocess

def setup_repo():
    repo_path = Path('/kaggle/working/EEG_Training_New')
    if not repo_path.exists():
        print("Cloning EEG_Training_New...")
        subprocess.run(["git", "clone", "https://github.com/Loki3306/EEG_Training_New.git", str(repo_path)], check=True)
    
    sys.path.insert(0, str(repo_path))

def patch_loso_script():
    script_path = Path('/kaggle/working/EEG_Training_New/training/train_matchnet_loso.py')
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    if "folds = list(iter_leave_one_subject_out(all_paths))[:1]" in code:
        print("Patching [:1] truncation to run full 18-subject LOSO...")
        code = code.replace("folds = list(iter_leave_one_subject_out(all_paths))[:1]", 
                            "folds = list(iter_leave_one_subject_out(all_paths))")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code)

def verify_batch_and_run():
    setup_repo()
    
    # Now we can import from EEG_Training_New
    from training.train_matchnet_loso import train_matchnet_loso, get_mapping_data, prepare_dataset, chunk_trial, FS
    from baselines.ridge_aad import load_subject_examples, subject_files

    print("--- 1. TENSOR VERIFICATION ---")
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    if not all_paths:
        print("Error: Could not find dataset paths. Please ensure dataset is mounted correctly in Kaggle.")
        return

    # Load just one subject to inspect a batch
    print(f"Loading {all_paths[0]} for batch inspection...")
    exs = load_subject_examples(all_paths[0])
    channels = [13, 46, 43, 23, 50, 0, 52, 14]
    
    X, YA, YB = prepare_dataset(exs, channels, 1.0, 6.0, all_paths[0].stem, mapping, envelopes)
    
    chunks_x, chunks_ya, chunks_yb = chunk_trial(X[0], YA[0], YB[0], window_sec=5, hop_sec=2)
    
    batch_x = torch.FloatTensor(np.stack(chunks_x))
    batch_ya = torch.FloatTensor(np.stack(chunks_ya))
    
    print(f"EEG Tensor Shape: {batch_x.shape}")
    print(f"EEG Mean: {batch_x.mean().item():.4f}, Std: {batch_x.std().item():.4f}")
    print(f"EEG Min: {batch_x.min().item():.4f}, Max: {batch_x.max().item():.4f}")
    
    print(f"Audio Tensor Shape: {batch_ya.shape}")
    print(f"Audio Mean: {batch_ya.mean().item():.4f}, Std: {batch_ya.std().item():.4f}")
    
    print("\n--- 2. EXECUTING FULL LOSO ---")
    patch_loso_script()
    
    from training.train_matchnet_loso import train_matchnet_loso as run_full_loso
    
    # Run the E0 Historical Baseline
    run_full_loso("eegnet", channels, 1.0, 6.0, batch_size=512, num_workers=2)

if __name__ == "__main__":
    verify_batch_and_run()
