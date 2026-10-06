import json
from collections import Counter
from pathlib import Path

def test_leakage_memorization():
    base_dir = Path("scripts/verify_baseline/data")
    map_file = base_dir / "audio_mapping.json"
    
    with open(map_file, 'r') as f:
        mapping = json.load(f)
        
    subs = [f'S{i}' for i in range(1, 19)]
    
    for test_sub in subs:
        if test_sub not in mapping: continue
        
        # Calculate training frequencies
        train_a = Counter()
        train_b = Counter()
        for sub in subs:
            if sub == test_sub: continue
            if sub not in mapping: continue
            for trial in mapping[sub].values():
                train_a[trial['wavA']['filename']] += 1
                train_b[trial['wavB']['filename']] += 1
                
        # Evaluate test subject's A and B files
        test_a_score = 0
        test_b_score = 0
        
        for trial in mapping[test_sub].values():
            fa = trial['wavA']['filename']
            fb = trial['wavB']['filename']
            
            # Net positive score in training set: (Count in A) - (Count in B)
            test_a_score += (train_a[fa] - train_b[fa])
            test_b_score += (train_a[fb] - train_b[fb])
            
        print(f"{test_sub} Test Set -> Total A Net Score: {test_a_score} | Total B Net Score: {test_b_score}")

if __name__ == "__main__":
    test_leakage_memorization()
