"""
Analyze training results and compute per-class accuracy metrics.
"""

import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import IsolatedSignDataset, collate_pad
from src.transformer_model import TransformerEncoderClassifier


def analyze_predictions(model_path, manifest_path, class_labels_path, feature_dir, device='cuda'):
    """
    Analyze model predictions on test set and compute per-class metrics.
    """
    # Load model checkpoint
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'classifier.1.weight' in state:
        num_classes = state['classifier.1.weight'].shape[0]
        d_model = state['input_proj.weight'].shape[0]
        input_dim = state['input_proj.weight'].shape[1]
        layer_keys = [k for k in state.keys() if k.startswith('transformer.layers.')]
        num_layers = max(int(k.split('.')[2]) for k in layer_keys) + 1 if layer_keys else 1
        if d_model % 8 == 0:
            nhead = 8
        elif d_model % 4 == 0:
            nhead = 4
        elif d_model % 2 == 0:
            nhead = 2
        else:
            nhead = 1
    else:
        raise ValueError('Unsupported checkpoint format or missing classifier weights')

    use_velocity = (input_dim == 126)
    model = TransformerEncoderClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.0
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Load data
    dataset = IsolatedSignDataset(manifest_path, feature_dir, use_velocity=use_velocity)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_pad)
    
    with open(class_labels_path, 'r') as f:
        label_to_gloss = {v: k for k, v in json.load(f).items()}
    
    # Collect predictions
    per_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    per_class_top5 = defaultdict(lambda: {'correct': 0, 'total': 0})
    all_correct = 0
    all_total = 0
    
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            
            logits = model(X, lengths)
            preds = logits.argmax(dim=1)
            top5_preds = logits.topk(5, dim=1)[1]
            
            # Top-1 accuracy
            correct_mask = (preds == y)
            all_correct += correct_mask.sum().item()
            all_total += y.size(0)
            
            # Per-class Top-1
            for i in range(y.size(0)):
                label = y[i].item()
                gloss = label_to_gloss.get(label, f"class_{label}")
                per_class_stats[gloss]['total'] += 1
                if correct_mask[i]:
                    per_class_stats[gloss]['correct'] += 1
                
                # Top-5
                if label in top5_preds[i]:
                    per_class_top5[gloss]['correct'] += 1
                per_class_top5[gloss]['total'] += 1
    
    # Compute metrics
    overall_top1 = all_correct / all_total if all_total > 0 else 0.0
    
    # Per-class accuracy
    per_class_acc = {}
    for gloss, stats in per_class_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        per_class_acc[gloss] = {
            'top1': acc,
            'top5': per_class_top5[gloss]['correct'] / per_class_top5[gloss]['total'],
            'samples': stats['total']
        }
    
    # Sort by accuracy
    sorted_acc = sorted(per_class_acc.items(), key=lambda x: x[1]['top1'], reverse=True)
    
    print(f"\n=== Test Set Evaluation ===")
    print(f"Overall Top-1: {overall_top1:.4f}")
    print(f"Overall Top-5: {sum(s['correct'] for s in per_class_top5.values()) / all_total:.4f}")
    print(f"\nPer-Class Accuracy (Top-10):")
    for gloss, metrics in sorted_acc[:10]:
        print(f"  {gloss:20s} | Top-1: {metrics['top1']:.4f} | Samples: {metrics['samples']}")
    
    print(f"\nPer-Class Accuracy (Bottom-10):")
    for gloss, metrics in sorted_acc[-10:]:
        print(f"  {gloss:20s} | Top-1: {metrics['top1']:.4f} | Samples: {metrics['samples']}")
    
    # Save results
    results = {
        'overall_top1': float(overall_top1),
        'per_class': per_class_acc,
        'num_classes': len(per_class_acc),
        'total_samples': all_total
    }
    
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    analyze_predictions(
        model_path='best_transformer.pt',
        manifest_path='data/manifests/test.json',
        class_labels_path='data/manifests/class_labels.json',
        feature_dir='data/processed_features',
        device=device
    )
