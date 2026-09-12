import argparse
import os
import json
import scipy.io as sio
from pathlib import Path

def inspect_dataset(data_root, output_file):
    root_path = Path(data_root)
    
    report = {
        "dataset_root": str(root_path),
        "inventory": {
            "mat_files": [],
            "json_files": [],
            "pkl_files": [],
            "wav_files": []
        },
        "eeg_structure": {},
        "audio_mapping": {}
    }
    
    if not root_path.exists():
        print(f"Warning: Data root {data_root} does not exist.")
        report["error"] = "Data root not found."
        write_report(report, output_file)
        return
        
    # File inventory
    for path in root_path.rglob('*'):
        if path.is_file():
            size = path.stat().st_size
            entry = {"name": path.name, "path": str(path.relative_to(root_path)), "size_bytes": size}
            if path.suffix == '.mat':
                report["inventory"]["mat_files"].append(entry)
            elif path.suffix == '.json':
                report["inventory"]["json_files"].append(entry)
            elif path.suffix == '.pkl':
                report["inventory"]["pkl_files"].append(entry)
            elif path.suffix == '.wav':
                report["inventory"]["wav_files"].append(entry)
                
    # Inspect a representative MAT file
    if report["inventory"]["mat_files"]:
        sample_mat = root_path / report["inventory"]["mat_files"][0]["path"]
        try:
            mat_data = sio.loadmat(sample_mat, struct_as_record=False, squeeze_me=True)
            report["eeg_structure"]["mat_keys"] = list(mat_data.keys())
            
            if 'data' in mat_data:
                data_struct = mat_data['data']
                report["eeg_structure"]["fields"] = dir(data_struct)
                
                # Check dimensions of 'eeg'
                if hasattr(data_struct, 'eeg'):
                    if isinstance(data_struct.eeg, np.ndarray):
                        report["eeg_structure"]["eeg_type"] = "ndarray"
                        report["eeg_structure"]["eeg_shape"] = data_struct.eeg.shape
                        if data_struct.eeg.dtype == object:
                            report["eeg_structure"]["eeg_is_cell"] = True
                            report["eeg_structure"]["num_trials"] = len(data_struct.eeg)
                            if len(data_struct.eeg) > 0:
                                report["eeg_structure"]["trial_0_shape"] = data_struct.eeg[0].shape
                
                # Check fsample
                if hasattr(data_struct, 'fsample'):
                    fsample = data_struct.fsample
                    report["eeg_structure"]["fsample_fields"] = dir(fsample)
                    if hasattr(fsample, 'eeg'):
                        report["eeg_structure"]["fsample_eeg"] = fsample.eeg
                        
                # Check events
                if hasattr(data_struct, 'event'):
                    event = data_struct.event
                    if hasattr(event, 'eeg'):
                        if hasattr(event.eeg, 'value'):
                            val = event.eeg.value
                            if isinstance(val, np.ndarray):
                                report["eeg_structure"]["event_values_sample"] = val[:5].tolist() if len(val) > 5 else val.tolist()
        except Exception as e:
            report["eeg_structure"]["error"] = str(e)

    # Inspect JSON mapping
    for jfile in report["inventory"]["json_files"]:
        if "audio_mapping" in jfile["name"]:
            try:
                with open(root_path / jfile["path"], 'r') as f:
                    mapping = json.load(f)
                    subjects = list(mapping.keys())
                    report["audio_mapping"]["num_subjects_in_mapping"] = len(subjects)
                    if subjects:
                        sample_sub = subjects[0]
                        trials = list(mapping[sample_sub].keys())
                        report["audio_mapping"]["sample_subject"] = sample_sub
                        report["audio_mapping"]["num_trials_in_sample"] = len(trials)
                        if trials:
                            sample_trial = trials[0]
                            report["audio_mapping"]["sample_trial_data"] = mapping[sample_sub][sample_trial]
            except Exception as e:
                report["audio_mapping"]["error"] = str(e)
                
    write_report(report, output_file)

def write_report(report, output_file):
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Inspection report saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect DTU Dataset on Kaggle")
    parser.add_argument("--data-root", type=str, default="/kaggle/input", help="Root directory of the dataset")
    parser.add_argument("--output", type=str, default="reports/DTU_dataset_inspection.json", help="Output JSON report path")
    args = parser.parse_args()
    
    import numpy as np # Import here to avoid global failure if not installed in basic envs
    inspect_dataset(args.data_root, args.output)
