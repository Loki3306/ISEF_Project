import argparse
import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files

def extract_speaker_info(filename):
    # eeg_audio_marianne_story3_trial_1.wav -> 'marianne'
    # Actually mapping says: "marianne_story3_trial_1.wav"
    name = filename.split('_')[0].lower()
    
    # Common Danish names in DTU dataset:
    # marianne (F), aske (M), mads (M), ...
    male_names = ['aske', 'mads', 'thomas', 'jens', 'peter', 'kasper', 'mikkel', 'anders', 'christian', 'martin']
    female_names = ['marianne', 'anna', 'mette', 'kirsten', 'helle', 'susanne', 'lotte', 'line', 'camilla']
    
    if name in male_names:
        return 'M'
    elif name in female_names:
        return 'F'
    else:
        # Fallback heuristic if unknown
        return 'U'

def get_mapping_data():
    base_dir = Path("/kaggle/input")
    map_files = list(base_dir.rglob("audio_mapping.json"))
    map_file = map_files[0] if map_files else REPO_ROOT / "data" / "audio_mapping.json"
        
    pkl_files = list(base_dir.rglob("*gammatone*.pkl"))
    if not pkl_files: pkl_files = list(base_dir.rglob("*.pkl"))
    env_file = pkl_files[0] if pkl_files else REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        
    print(f"Using map file: {map_file}")
    print(f"Using env file: {env_file}")
    
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def run_forensics():
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return

    print("="*60)
    print("DATASET FORENSICS AUDIT")
    print("="*60)

    # 1. Contingency Table Stats
    male_att_count = 0
    female_att_count = 0
    male_unatt_count = 0
    female_unatt_count = 0
    
    # 2. Acoustic Stats
    rms_att = []
    rms_unatt = []
    mean_att = []
    mean_unatt = []
    std_att = []
    std_unatt = []
    
    # 3. Audio-only Classifier Features
    X_audio = []
    y_audio = []

    for p in all_paths:
        sub_key = p.stem.replace("_data_preproc", "")
        examples = load_subject_examples(p)
        
        for i, ex in enumerate(examples):
            trial_key = f"trial_{i}"
            if sub_key in mapping and trial_key in mapping[sub_key]:
                fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
                fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
                
                # wavA is Attended
                gender_a = extract_speaker_info(fname_a)
                gender_b = extract_speaker_info(fname_b)
                
                if gender_a == 'M': male_att_count += 1
                elif gender_a == 'F': female_att_count += 1
                
                if gender_b == 'M': male_unatt_count += 1
                elif gender_b == 'F': female_unatt_count += 1
                
                env_a = envelopes[fname_a] # [T, 28] or [28, T]
                env_b = envelopes[fname_b]
                
                # Gammatone envelopes in the pickle are (T, 28) typically. Wait, let's just flatten it.
                rms_a = np.sqrt(np.mean(env_a**2))
                rms_b = np.sqrt(np.mean(env_b**2))
                
                m_a = np.mean(env_a)
                m_b = np.mean(env_b)
                
                s_a = np.std(env_a)
                s_b = np.std(env_b)
                
                rms_att.append(rms_a)
                rms_unatt.append(rms_b)
                
                mean_att.append(m_a)
                mean_unatt.append(m_b)
                
                std_att.append(s_a)
                std_unatt.append(s_b)
                
                # Features for classifier:
                # We want the classifier to take [features_1, features_2] and predict if 1 is attended
                # For half the examples, we feed [A, B] -> y=1
                # For the other half, we feed [B, A] -> y=0
                feat_a = [rms_a, m_a, s_a]
                feat_b = [rms_b, m_b, s_b]
                
                if np.random.rand() > 0.5:
                    X_audio.append(feat_a + feat_b)
                    y_audio.append(1)
                else:
                    X_audio.append(feat_b + feat_a)
                    y_audio.append(0)

    # Summarize Contingency Table
    print("\n--- 1. SPEAKER GENDER X ATTENTION CONTINGENCY ---")
    print(f"Attended Speaker   -> Male: {male_att_count} | Female: {female_att_count}")
    print(f"Unattended Speaker -> Male: {male_unatt_count} | Female: {female_unatt_count}")
    
    total_att = male_att_count + female_att_count
    if total_att > 0:
        print(f"P(Male | Attended) = {male_att_count/total_att:.2f}")
    
    # Summarize Acoustic Stats
    print("\n--- 2. ACOUSTIC MARGINAL STATISTICS (Gammatone Envelope) ---")
    print(f"RMS Attended:   {np.mean(rms_att):.4f} +/- {np.std(rms_att):.4f}")
    print(f"RMS Unattended: {np.mean(rms_unatt):.4f} +/- {np.std(rms_unatt):.4f}")
    print(f"Delta RMS (Att - Unatt): {np.mean(np.array(rms_att) - np.array(rms_unatt)):.4f}")
    print(f"Mean Attended:  {np.mean(mean_att):.4f} +/- {np.std(mean_att):.4f}")
    print(f"Mean Unattended:{np.mean(mean_unatt):.4f} +/- {np.std(mean_unatt):.4f}")
    print(f"Std Attended:   {np.mean(std_att):.4f} +/- {np.std(std_att):.4f}")
    print(f"Std Unattended: {np.mean(std_unatt):.4f} +/- {np.std(std_unatt):.4f}")
    
    # Audio-only Classifier
    print("\n--- 3. AUDIO-ONLY CLASSIFIER TEST ---")
    X_audio = np.array(X_audio)
    y_audio = np.array(y_audio)
    
    scaler = StandardScaler()
    X_audio_scaled = scaler.fit_transform(X_audio)
    
    clf = LogisticRegression(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_audio_scaled, y_audio, cv=cv, scoring='accuracy')
    
    print(f"Logistic Regression (Features: RMS, Mean, Std)")
    print(f"Cross-Val Accuracy: {np.mean(scores)*100:.2f}% +/- {np.std(scores)*100:.2f}%")
    
    if np.mean(scores) > 0.60:
        print(">>> CRITICAL WARNING: Audio-only classifier is beating chance (50%) by a large margin!")
        print(">>> This confirms a massive structural bias (leakage) in the dataset.")
    else:
        print(">>> Audio-only marginals are clean. No obvious statistical bias detected.")

if __name__ == "__main__":
    run_forensics()
