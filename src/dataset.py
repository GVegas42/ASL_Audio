import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class IsolatedSignDataset(Dataset):
    def __init__(self, manifest_path: str, features_dir: str, use_velocity: bool = True):
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        self.features_dir = features_dir
        self.use_velocity = use_velocity

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.FloatTensor, int]:
        entry = self.samples[idx]
        feature_file = entry['feature_file']
        label = int(entry['label_int'])

        arr = np.load(os.path.join(self.features_dir, feature_file))
        arr = arr.astype(np.float32)  # (T, 21, 3)

        if self.use_velocity:
            # Frame-to-frame deltas; first frame gets a zero delta
            vel = np.diff(arr, axis=0)
            vel = np.concatenate([np.zeros((1, arr.shape[1], arr.shape[2]), dtype=np.float32), vel], axis=0)
            arr = np.concatenate([arr, vel], axis=2)  # (T, 21, 6)

        # Flatten landmark dims: (T, 21, 3) -> (T, 63)  or  (T, 21, 6) -> (T, 126)
        arr = arr.reshape(arr.shape[0], -1)

        return torch.from_numpy(arr), label


def collate_pad(batch: List[Tuple[torch.FloatTensor, int]], max_len: int = 64):
    sequences, labels = zip(*batch)
    batch_size = len(sequences)
    dim = sequences[0].shape[1]

    padded = torch.zeros((batch_size, max_len, dim), dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, seq in enumerate(sequences):
        L = min(seq.shape[0], max_len)
        padded[i, :L] = seq[:L]
        lengths[i] = L

    return padded, torch.tensor(labels, dtype=torch.long), lengths
