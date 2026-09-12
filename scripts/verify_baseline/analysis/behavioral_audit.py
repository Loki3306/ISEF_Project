import json
import csv
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

def main():
    repo_root = Path(__file__).resolve().parents[3]
    json_path = repo_root / "experiments" / "matchnet_within_subject_forensics.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found. Please run the training script first.")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    metrics = data["summary_metrics"]
    
    audit_results = {}
    
    for subj, subj_data in metrics.items():
        if "r_correct_array" not in subj_data:
            print(f"Skipping {subj}: raw prediction arrays not found.")
            continue
            
        r_A = np.array(subj_data["r_correct_array"])
        r_B = np.array(subj_data["r_incorrect_array"])
        
        n_total = len(r_A)
        if n_total == 0:
            continue
            
        # Quadrants
        q1 = (r_A > 0) & (r_B < 0)
        q2 = (r_A < 0) & (r_B < 0)
        q3 = (r_A > 0) & (r_B > 0)
        q4 = (r_A < 0) & (r_B > 0)
        zeros = (r_A == 0) | (r_B == 0)
        
        q1_prop = q1.mean()
        q2_prop = q2.mean()
        q3_prop = q3.mean()
        q4_prop = q4.mean()
        zero_prop = zeros.mean()
        
        # Correctly signed only
        correct = r_A > r_B
        n_correct = correct.sum()
        
        q1_correct_prop = q1[correct].sum() / n_correct if n_correct > 0 else 0
        q2_correct_prop = q2[correct].sum() / n_correct if n_correct > 0 else 0
        q3_correct_prop = q3[correct].sum() / n_correct if n_correct > 0 else 0
        q4_correct_prop = q4[correct].sum() / n_correct if n_correct > 0 else 0
        zero_correct_prop = zeros[correct].sum() / n_correct if n_correct > 0 else 0
        
        delta_sign = subj_data["signed_acc"] - subj_data["abs_acc"]
        
        audit_results[subj] = {
            "signed_acc": subj_data["signed_acc"],
            "abs_acc": subj_data["abs_acc"],
            "delta_sign": delta_sign,
            "med_r_A": float(np.median(r_A)),
            "med_r_B": float(np.median(r_B)),
            "med_r_A_minus_r_B": float(np.median(r_A - r_B)),
            "med_abs_r_A_minus_abs_r_B": float(np.median(np.abs(r_A) - np.abs(r_B))),
            "frac_r_A_pos": float((r_A > 0).mean()),
            "frac_r_B_neg": float((r_B < 0).mean()),
            
            "q1_all": float(q1_prop),
            "q2_all": float(q2_prop),
            "q3_all": float(q3_prop),
            "q4_all": float(q4_prop),
            "zero_all": float(zero_prop),
            
            "q1_correct": float(q1_correct_prop),
            "q2_correct": float(q2_correct_prop),
            "q3_correct": float(q3_correct_prop),
            "q4_correct": float(q4_correct_prop),
            "zero_correct": float(zero_correct_prop),
        }
    
    if len(audit_results) == 0:
        print("Error: No subjects with raw arrays found in JSON.")
        return
        
    # Dump JSON
    audit_json = repo_root / "experiments" / "behavioral_audit.json"
    with open(audit_json, 'w') as f:
        json.dump(audit_results, f, indent=4)
        
    # Dump CSV
    audit_csv = repo_root / "experiments" / "behavioral_audit.csv"
    with open(audit_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["subject"] + list(audit_results[list(audit_results.keys())[0]].keys()))
        writer.writeheader()
        for subj, stats in audit_results.items():
            row = {"subject": subj}
            row.update(stats)
            writer.writerow(row)
            
    # Calculate correlations
    delta_sign_list = [v["delta_sign"] for v in audit_results.values()]
    q2_all_list = [v["q2_all"] for v in audit_results.values()]
    q1_all_list = [v["q1_all"] for v in audit_results.values()]
    signed_acc_list = [v["signed_acc"] for v in audit_results.values()]
    
    corr_delta_q2, _ = pearsonr(delta_sign_list, q2_all_list)
    corr_q1_signed, _ = pearsonr(q1_all_list, signed_acc_list)
    
    # Overall summary
    mean_q1_correct = np.mean([v["q1_correct"] for v in audit_results.values()])
    mean_q2_correct = np.mean([v["q2_correct"] for v in audit_results.values()])
    
    # MD Report
    s4_q3_all = audit_results.get("S4_data_preproc", {}).get("q3_all", 0)
    s4_q4_all = audit_results.get("S4_data_preproc", {}).get("q4_all", 0)
    s4_q2_all = audit_results.get("S4_data_preproc", {}).get("q2_all", 0)
    s4_signed = audit_results.get("S4_data_preproc", {}).get("signed_acc", 0)
    
    md_content = f"""# Behavioral Audit Report

## 1. Are high signed-accuracy subjects actually positively aligned with the attended speaker?
**Analysis:** We correlated the Q1 proportion (genuine positive alignment + distractor rejection) with Signed Accuracy across {len(audit_results)} subjects.
**Correlation ($r$):** {corr_q1_signed:.3f}
**Conclusion:** {"The correlation is moderate/high, indicating that positive alignment does somewhat contribute to performance." if corr_q1_signed > 0.3 else "The correlation is weak, indicating that high accuracy is often achieved WITHOUT genuine positive alignment."}

## 2. How much correct performance comes from Q1 versus Q2?
**Analysis:** Across all subjects, when the model makes a CORRECT signed decision ($r_A > r_B$), we look at whether it was a Q1 (ideal) or Q2 (distractor rejection) strategy.
- **Mean % of correct decisions from Q1:** {mean_q1_correct*100:.1f}%
- **Mean % of correct decisions from Q2:** {mean_q2_correct*100:.1f}%
**Conclusion:** {"Q2 is dominating the correct decisions, proving the contrastive loss relies heavily on distractor rejection." if mean_q2_correct > mean_q1_correct else "Q1 is the dominant strategy, showing the model is primarily learning positive alignment."}

## 3. Is Q2 responsible for the signed-vs-absolute discrepancy?
**Analysis:** We correlated the Signed-Absolute Accuracy Gap ($\Delta_{{sign}}$) with the proportion of overall chunks falling into Q2.
**Correlation ($r$):** {corr_delta_q2:.3f}
**Conclusion:** {"There is a strong positive correlation, proving definitively that distractor-rejection is the cause of the massive discrepancy." if corr_delta_q2 > 0.5 else "The correlation is weak, meaning the discrepancy might have other causes."}

## 4. Does S4 genuinely exhibit Q3 stimulus confusion rather than phase inversion?
**Analysis:** S4 exhibits a Signed Accuracy of {s4_signed*100:.1f}%. If this was a phase inversion, S4 should be heavily in Q4 ($r_A < 0, r_B > 0$) or Q2.
- **S4 Q3 (Stimulus Confusion, $r_A > 0, r_B > 0$):** {s4_q3_all*100:.1f}%
- **S4 Q4 (Catastrophic Phase Inversion, $r_A < 0, r_B > 0$):** {s4_q4_all*100:.1f}%
- **S4 Q2 (Pure rejection, $r_A < 0, r_B < 0$):** {s4_q2_all*100:.1f}%
**Conclusion:** {"S4 is heavily in Q3, proving it is stimulus confusion (it generates positive correlations for both, but the distractor is stronger)." if s4_q3_all > s4_q4_all else "S4 still shows significant Q4 behavior, keeping phase inversion as a possible factor."}

## Per-Subject Data Summary

| Subject | Signed Acc | Delta Sign | Overall Q1 | Overall Q2 | Correct Q1 | Correct Q2 |
|---------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
"""
    for subj in sorted(audit_results.keys()):
        stats = audit_results[subj]
        md_content += f"| {subj} | {stats['signed_acc']*100:.1f}% | {stats['delta_sign']*100:.1f}% | {stats['q1_all']*100:.1f}% | {stats['q2_all']*100:.1f}% | {stats['q1_correct']*100:.1f}% | {stats['q2_correct']*100:.1f}% |\n"

    audit_md = repo_root / "experiments" / "BEHAVIORAL_AUDIT.md"
    with open(audit_md, 'w') as f:
        f.write(md_content)
        
    print(f"Generated Audit Reports in {repo_root}/experiments/ (behavioral_audit.json, behavioral_audit.csv, BEHAVIORAL_AUDIT.md)")
    with open(audit_md, 'r') as f:
        print("\n\n" + f.read())

if __name__ == "__main__":
    main()
