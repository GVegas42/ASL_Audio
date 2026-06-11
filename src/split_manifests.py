import json
import os
import random
from typing import List


def split_manifest(train_path: str, val_path: str, test_path: str, val_frac=0.2, test_frac=0.1, seed=42):
    with open(train_path, 'r') as f:
        samples = json.load(f)

    if os.path.exists(val_path):
        with open(val_path, 'r') as f:
            val_samples = json.load(f)
    else:
        val_samples = []

    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            test_samples = json.load(f)
    else:
        test_samples = []

    # If val/test already non-empty, do nothing
    if val_samples or test_samples:
        print('Val or test manifests already exist; skipping split.')
        return

    random.seed(seed)
    idxs = list(range(len(samples)))
    random.shuffle(idxs)

    n = len(samples)
    n_test = max(1, int(n * test_frac)) if n >= 3 else 1
    n_val = max(1, int(n * val_frac)) if n - n_test >= 2 else 1

    test_idxs = set(idxs[:n_test])
    val_idxs = set(idxs[n_test:n_test + n_val])

    new_train = [s for i, s in enumerate(samples) if i not in test_idxs and i not in val_idxs]
    new_val = [s for i, s in enumerate(samples) if i in val_idxs]
    new_test = [s for i, s in enumerate(samples) if i in test_idxs]

    with open(train_path, 'w') as f:
        json.dump(new_train, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(new_val, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(new_test, f, indent=2)

    print(f'Split completed. Train: {len(new_train)} | Val: {len(new_val)} | Test: {len(new_test)}')


if __name__ == '__main__':
    manifest_dir = 'data/manifests'
    split_manifest(
        os.path.join(manifest_dir, 'train.json'),
        os.path.join(manifest_dir, 'val.json'),
        os.path.join(manifest_dir, 'test.json'),
        val_frac=0.2,
        test_frac=0.1,
        seed=42,
    )
