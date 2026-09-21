import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.io import wavfile
import librosa
from scipy.signal import resample_poly
import math
from transformers import WavLMModel

OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "wavlm_features.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_wavlm_features(wav_path, model, target_fs=64, target_layer=6):
    """
    Extracts frozen WavLM representations for a given audio file.
    WavLM expects 16kHz audio. Its output frame rate is 50Hz (20ms stride).
    We resample the 50Hz feature sequence to 64Hz to match the EEG.
    """
    # 1. Load and resample audio to 16000 Hz (WavLM native)
    data, fs = librosa.load(wav_path, sr=16000, mono=True)
    
    # 2. Extract WavLM representations (frozen)
    inputs = torch.tensor(data).unsqueeze(0).to(DEVICE) # [1, Samples]
    with torch.no_grad():
        # output_hidden_states=True returns a tuple of all layers
        outputs = model(inputs, output_hidden_states=True)
        
    # Hidden states tuple has 13 elements (0 is embedding, 1-12 are layers)
    # Shape: [1, Frames, 768]
    hidden_states = outputs.hidden_states[target_layer].squeeze(0).cpu().numpy()
    
    # Transpose to [768, Frames]
    hidden_states = hidden_states.T
    
    # 3. Resample from 50 Hz to 64 Hz
    # WavLM frame shift is 320 samples at 16kHz = 20ms = 50 Hz
    wavlm_fs = 50
    g = math.gcd(target_fs, wavlm_fs)
    up = target_fs // g    # e.g. 64 // 2 = 32
    down = wavlm_fs // g   # e.g. 50 // 2 = 25
    
    # resample_poly operates along the last axis by default
    resampled_features = resample_poly(hidden_states, up, down, axis=-1)
    
    return resampled_features

def main(audio_dir, layer=6):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"Loading microsoft/wavlm-base on {DEVICE}...")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(DEVICE)
    model.eval()
    
    print(f"Extracting WavLM Layer {layer} features for {len(wav_files)} files...")
    
    results = {}
    for i, w in enumerate(wav_files):
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(wav_files)}] Processing...")
        try:
            feats = extract_wavlm_features(w, model, target_layer=layer)
            results[w.name] = feats
        except Exception as e:
            print(f"Failed {w.name}: {e}")
            
    with open(OUT_FILE, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Done. Saved to {OUT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio", help="Path to raw WAV files")
    parser.add_argument("--layer", type=int, default=6, help="WavLM hidden layer to extract (1-12)")
    args = parser.parse_args()
    main(Path(args.audio_dir), args.layer)
