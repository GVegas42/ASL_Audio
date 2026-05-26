import json
import os
import shutil

def organize_dataset(json_path, raw_videos_dir, target_words=None):
    """
    Parses WLASL JSON metadata and prints maps for training/validation splits.
    Optionally filters for a specific list of target words.
    """
    with open(json_path, 'r') as f:
        dataset_meta = json.load(f)
        
    split_manifests = {
        "train": [],
        "val": [],
        "test": []
    }
    
    label_to_index = {}
    class_counter = 0
    
    for entry in dataset_meta:
        gloss = entry['gloss']
        
        if target_words and gloss not in target_words:
            continue
            
        # Map word string to a numerical ID for Cross-Entropy Loss later
        if gloss not in label_to_index:
            label_to_index[gloss] = class_counter
            class_counter += 1
            
        for instance in entry['instances']:
            video_id = instance['video_id']
            split = instance['split']
            
            video_filename = f"{video_id}.mp4"
            feature_filename = f"{video_id}.npy"
            video_file_path = os.path.join(raw_videos_dir, video_filename)
            
            if os.path.exists(video_file_path):
                split_manifests[split].append({
                    "feature_file": feature_filename,
                    "label_string": gloss,
                    "label_int": label_to_index[gloss]
                })

    # Save out manifests as clear JSON manifests for later training
    output_dir = "data/manifests"
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, entries in split_manifests.items():
        with open(os.path.join(output_dir, f"{split_name}.json"), 'w') as out_f:
            json.dump(entries, out_f, indent=4)
            
    with open(os.path.join(output_dir, "class_labels.json"), 'w') as out_f:
        json.dump(label_to_index, out_f, indent=4)
        
    print(f"Manifests generated successfully! Found data for {len(label_to_index)} classes.")
    print(f"Train samples: {len(split_manifests['train'])} | Val: {len(split_manifests['val'])} | Test: {len(split_manifests['test'])}")

if __name__ == "__main__":
    vocabulary_subset = ["we","wet"]
    
    organize_dataset(
        json_path="data/WLASL_v0.3.json",
        raw_videos_dir="data/raw_videos",
        target_words=vocabulary_subset                # Pass None to process everything
    )