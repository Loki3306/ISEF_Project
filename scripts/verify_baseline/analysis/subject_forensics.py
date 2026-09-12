import json
import numpy as np
from pathlib import Path
import sys
import os
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_baseline.baselines.ridge_aad import (
    load_all_examples,
    subject_files,
    feature_statistics,
    fit_ridge,
    predict_envelope,
    target_envelope,
    attended_stream_from_label,
    TrialExample
)

import scipy.stats

def evaluate_predictions(preds: list[np.ndarray], targets: list[np.ndarray], distractors: list[np.ndarray]):
    correct = 0
    total = len(preds)
    for p, t, d in zip(preds, targets, distractors):
        corr_t = scipy.stats.pearsonr(p, t)[0]
        corr_d = scipy.stats.pearsonr(p, d)[0]
        if corr_t > corr_d:
            correct += 1
    return correct / total if total > 0 else 0.0

def run_subject_forensics():
    print("Loading audio mapping...")
    with open(REPO_ROOT / "scripts" / "verify_baseline" / "data" / "audio_mapping.json") as f:
        audio_mapping = json.load(f)

    print("Loading all examples...")
    all_examples = load_all_examples()
    
    # E0 Canonical Montage
    channel_ids = [13, 46, 43, 23, 50, 0, 52, 14]
    
    # Subset channels
    for ex in all_examples:
        ex.eeg = ex.eeg[channel_ids, :]

    # Group by subject
    subjects = sorted(list(set(ex.subject for ex in all_examples)))
    subject_examples = {s: [ex for ex in all_examples if ex.subject == s] for s in subjects}
    
    results = {
        "subjects": subjects,
        "metrics": {},
        "decoder_transfer_matrix": np.zeros((len(subjects), len(subjects))).tolist(),
        "decoder_similarity_matrix": np.zeros((len(subjects), len(subjects))).tolist()
    }

    # 1. Subject Properties (Bandpower, etc.) & Train subject-specific models
    subject_models = {} # dict of (weights, mean, std)
    
    for s in subjects:
        print(f"Processing Subject {s}...")
        examples = subject_examples[s]
        
        # Bandpower: average variance of channels
        variances = []
        for ex in examples:
            variances.append(np.var(ex.eeg, axis=1))
        avg_var = np.mean(variances)
        
        # 5-fold CV for Oracle Accuracy
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oracle_correct = 0
        oracle_total = 0
        
        for train_idx, test_idx in kf.split(examples):
            train_ex = [examples[i] for i in train_idx]
            test_ex = [examples[i] for i in test_idx]
            
            f_mean, f_std = feature_statistics(train_ex, channel_ids=None)
            w = fit_ridge(train_ex, audio_mapping, feature_mean=f_mean, feature_std=f_std)
            
            preds = []
            targets = []
            distractors = []
            for ex in test_ex:
                p = predict_envelope(ex.eeg, w, feature_mean=f_mean, feature_std=f_std)
                t = target_envelope(ex, audio_mapping)
                d_stream = "B" if attended_stream_from_label(ex.label, audio_mapping) == "A" else "A"
                d_ex = TrialExample(ex.subject, ex.trial_index, ex.eeg, ex.wav_a, ex.wav_b, 1 if d_stream == "A" else 2)
                d = target_envelope(d_ex, audio_mapping)
                preds.append(p)
                targets.append(t)
                distractors.append(d)
                
            oracle_correct += sum(1 for p, t, d in zip(preds, targets, distractors) if scipy.stats.pearsonr(p, t)[0] > scipy.stats.pearsonr(p, d)[0])
            oracle_total += len(test_ex)
            
        oracle_acc = oracle_correct / oracle_total
        
        # Full subject model (for transfer matrix and similarity)
        full_f_mean, full_f_std = feature_statistics(examples, channel_ids=None)
        full_w = fit_ridge(examples, audio_mapping, feature_mean=full_f_mean, feature_std=full_f_std)
        subject_models[s] = (full_w, full_f_mean, full_f_std)
        
        results["metrics"][s] = {
            "bandpower": float(avg_var),
            "oracle_acc": float(oracle_acc)
        }
        
    # 2. LOSO Accuracy
    print("Computing LOSO Accuracies...")
    for s in subjects:
        train_ex = [ex for ex in all_examples if ex.subject != s]
        test_ex = subject_examples[s]
        
        f_mean, f_std = feature_statistics(train_ex, channel_ids=None)
        w = fit_ridge(train_ex, audio_mapping, feature_mean=f_mean, feature_std=f_std)
        
        preds = []
        targets = []
        distractors = []
        for ex in test_ex:
            p = predict_envelope(ex.eeg, w, feature_mean=f_mean, feature_std=f_std)
            t = target_envelope(ex, audio_mapping)
            d_stream = "B" if attended_stream_from_label(ex.label, audio_mapping) == "A" else "A"
            d_ex = TrialExample(ex.subject, ex.trial_index, ex.eeg, ex.wav_a, ex.wav_b, 1 if d_stream == "A" else 2)
            d = target_envelope(d_ex, audio_mapping)
            preds.append(p)
            targets.append(t)
            distractors.append(d)
            
        loso_acc = evaluate_predictions(preds, targets, distractors)
        results["metrics"][s]["loso_acc"] = float(loso_acc)
        results["metrics"][s]["oracle_gain"] = float(results["metrics"][s]["oracle_acc"] - loso_acc)
        print(f"Subject {s}: LOSO={loso_acc:.4f}, Oracle={results['metrics'][s]['oracle_acc']:.4f}, Gain={results['metrics'][s]['oracle_gain']:.4f}")

    # 3. Transfer Matrix & Similarity
    print("Computing Transfer Matrix and Similarity...")
    for i, s_train in enumerate(subjects):
        w_train, mean_train, std_train = subject_models[s_train]
        for j, s_test in enumerate(subjects):
            # Similarity
            w_test = subject_models[s_test][0]
            sim = np.dot(w_train, w_test) / (np.linalg.norm(w_train) * np.linalg.norm(w_test))
            results["decoder_similarity_matrix"][i][j] = float(sim)
            
            # Transfer Accuracy
            test_ex = subject_examples[s_test]
            preds = []
            targets = []
            distractors = []
            for ex in test_ex:
                p = predict_envelope(ex.eeg, w_train, feature_mean=mean_train, feature_std=std_train)
                t = target_envelope(ex, audio_mapping)
                d_stream = "B" if attended_stream_from_label(ex.label, audio_mapping) == "A" else "A"
                d_ex = TrialExample(ex.subject, ex.trial_index, ex.eeg, ex.wav_a, ex.wav_b, 1 if d_stream == "A" else 2)
                d = target_envelope(d_ex, audio_mapping)
                preds.append(p)
                targets.append(t)
                distractors.append(d)
            acc = evaluate_predictions(preds, targets, distractors)
            results["decoder_transfer_matrix"][i][j] = float(acc)

    # Save results
    out_path = REPO_ROOT / "experiments" / "forensics_data.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_subject_forensics()
