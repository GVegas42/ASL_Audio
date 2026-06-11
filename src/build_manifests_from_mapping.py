"""
Build train/val/test manifests from mapping.csv, auto-splitting all matched files.
Respects original WLASL split assignments when available, otherwise auto-assigns.
"""

import json
import csv
import os
import random
from collections import defaultdict


def build_manifests_from_mapping(mapping_csv, wlasl_json, output_dir, seed=42):
    """
    Build manifests from mapping.csv, splitting files into train/val/test.
    
    Args:
        mapping_csv: path to mapping.csv (output from match_filenames.py)
        wlasl_json: path to WLASL_v0.3.json
        output_dir: directory to save train.json, val.json, test.json
        seed: random seed for deterministic split
    """
    
    # Load mapping
    mapping = {}
    with open(mapping_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['video_id']] = {
                'gloss': row['gloss'],
                'match_type': row['match_type'],
                'confidence': float(row['confidence'])
            }
    
    # Load WLASL for original split info
    with open(wlasl_json, 'r') as f:
        wlasl = json.load(f)
    
    # Build gloss→label_id mapping and collect samples
    label_to_index = {}
    label_counter = 0
    split_manifests = defaultdict(list)
    
    # First pass: collect all matched samples
    all_samples = []
    for entry in wlasl:
        gloss = entry.get('gloss')
        if not gloss:
            continue
        
        if gloss not in label_to_index:
            label_to_index[gloss] = label_counter
            label_counter += 1
        
        for instance in entry.get('instances', []):
            video_id = str(instance.get('video_id'))
            if video_id in mapping:
                # Sample matched; use original split or mark for auto-assignment
                original_split = instance.get('split', 'train')
                all_samples.append({
                    'video_id': video_id,
                    'gloss': gloss,
                    'label_int': label_to_index[gloss],
                    'original_split': original_split
                })
    
    print(f"Total matched samples: {len(all_samples)}")
    print(f"Classes found: {len(label_to_index)}")
    
    # Second pass: respect original splits, or auto-assign unmapped samples
    for sample in all_samples:
        split = sample['original_split']
        if split not in split_manifests:
            split_manifests[split] = []
        
        split_manifests[split].append({
            'feature_file': f"{sample['video_id']}.npy",
            'label_string': sample['gloss'],
            'label_int': sample['label_int']
        })
    
    # If train/val/test are too skewed, rebalance
    if 'val' not in split_manifests or len(split_manifests['val']) == 0:
        print("No val split found; auto-splitting train/val/test with 80/10/10...")
        all_train = split_manifests.get('train', [])
        random.seed(seed)
        random.shuffle(all_train)
        
        n = len(all_train)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int((n - n_test) * 0.111))  # 10% of total
        
        split_manifests['test'] = all_train[:n_test]
        split_manifests['val'] = all_train[n_test:n_test + n_val]
        split_manifests['train'] = all_train[n_test + n_val:]
    
    # Save manifests
    os.makedirs(output_dir, exist_ok=True)
    for split_name, entries in split_manifests.items():
        with open(os.path.join(output_dir, f"{split_name}.json"), 'w') as f:
            json.dump(entries, f, indent=2)
    
    # Save class labels
    with open(os.path.join(output_dir, "class_labels.json"), 'w') as f:
        json.dump(label_to_index, f, indent=2)
    
    print(f"\nManifests saved:")
    for split_name, entries in sorted(split_manifests.items()):
        print(f"  {split_name}: {len(entries)} samples")
    print(f"Total classes: {len(label_to_index)}")


if __name__ == "__main__":
    build_manifests_from_mapping(
        mapping_csv="data/manifests/mapping.csv",
        wlasl_json="data/WLASL_v0.3.json",
        output_dir="data/manifests",
        seed=42
    )
