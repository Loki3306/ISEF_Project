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
from transformers import WavLMModel, AutoFeatureExtractor

OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "wavlm_features.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_wavlm_features(wav_path, model, processor, target_fs=64, target_layers=[3, 6, 9, 12]):
    """
    Extracts frozen WavLM representations for a given audio file.
    WavLM expects 16kHz audio. Its output frame rate is 50Hz (20ms stride).
    We resample the 50Hz feature sequence to 64Hz to match the EEG.
    """
    # 1. Load and resample audio to 16000 Hz (WavLM native)
    data, fs = librosa.load(wav_path, sr=16000, mono=True)
    audio_duration = len(data) / fs
    
    # 2. Extract WavLM representations (frozen)
    inputs = processor(data, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # output_hidden_states=True returns a tuple of all layers
        outputs = model(inputs.input_values, output_hidden_states=True)
        
    # Hidden states tuple has 13 elements (0 is embedding, 1-12 are layers)
    hidden_states = []
    for l in target_layers:
        hs = outputs.hidden_states[l].squeeze(0).cpu().numpy().T # [768, Frames]
        hidden_states.append(hs)
        
    stacked_hidden = np.stack(hidden_states, axis=0) # [N_layers, 768, Frames]
    wavlm_frames = stacked_hidden.shape[-1]
    
    # 3. Resample from 50 Hz to 64 Hz
    # WavLM frame shift is 320 samples at 16kHz = 20ms = 50 Hz
    wavlm_fs = 50
    g = math.gcd(target_fs, wavlm_fs)
    up = target_fs // g    # e.g. 64 // 2 = 32
    down = wavlm_fs // g   # e.g. 50 // 2 = 25
    
    # resample_poly operates along the last axis by default
    resampled_features = resample_poly(stacked_hidden, up, down, axis=-1)
    resampled_frames = resampled_features.shape[-1]
    
    # Diagnostic print to verify lengths
    print(f"  {wav_path.name}: {audio_duration:.2f}s -> {wavlm_frames} WavLM frames -> {resampled_frames} 64Hz frames")
    
    return resampled_features

def main(audio_dir, layers=[3, 6, 9, 12]):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"Loading microsoft/wavlm-base on {DEVICE}...")
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(DEVICE)
    model.eval()
    
    print(f"Extracting WavLM Layers {layers} features for {len(wav_files)} files...")
    
    results = {}
    for i, w in enumerate(wav_files):
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(wav_files)}] Processing...")
        try:
            feats = extract_wavlm_features(w, model, processor, target_layers=layers)
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
    parser.add_argument("--layers", type=int, nargs='+', default=[3, 6, 9, 12], help="WavLM hidden layers to extract (1-12)")
    args = parser.parse_args()
    main(Path(args.audio_dir), args.layers)
