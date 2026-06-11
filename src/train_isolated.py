import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IsolatedSignDataset, collate_pad
from transformer_model import TransformerEncoderClassifier


def load_class_labels(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    correct = 0
    total = 0
    sample_info = None
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            logits = model(X, lengths)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            # Keep the last batch for debugging inspection
            sample_info = (logits.cpu().numpy(), y.cpu().numpy(), preds.cpu().numpy())
    acc = correct / total if total > 0 else 0.0
    return acc, total, sample_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', default='data/manifests')
    parser.add_argument('--features', default='data/processed_features')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use-class-weight', action='store_true', help='Use class weights in CrossEntropyLoss')
    parser.add_argument('--oversample', action='store_true', help='Use oversampling to balance classes during training')
    parser.add_argument('--no-velocity', action='store_true', help='Disable velocity (frame-delta) features; use raw landmarks only')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    class_labels = load_class_labels(os.path.join(args.manifests, 'class_labels.json'))
    num_classes = len(class_labels)
    use_velocity = not args.no_velocity
    input_dim = 126 if use_velocity else 63

    train_ds = IsolatedSignDataset(os.path.join(args.manifests, 'train.json'), args.features, use_velocity=use_velocity)
    val_ds = IsolatedSignDataset(os.path.join(args.manifests, 'val.json'), args.features, use_velocity=use_velocity)

    collate = lambda b: collate_pad(b, max_len=args.max_len)

    # Optionally create an oversampling sampler to balance classes
    if args.oversample:
        labels_list = [int(s['label_int']) for s in train_ds.samples]
        counts = np.bincount(labels_list, minlength=num_classes)
        # weight per class = 1/count
        class_weights = 1.0 / (counts + 1e-8)
        samples_weights = np.array([class_weights[l] for l in labels_list], dtype=np.float32)
        sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Num classes: {num_classes}")

    print(f"Velocity features: {'enabled' if use_velocity else 'disabled'} | Input dim: {input_dim}")
    model = TransformerEncoderClassifier(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model.to(device)

    # Optionally set class-weighted loss
    device = torch.device(args.device)
    if args.use_class_weight:
        labels_list = [int(s['label_int']) for s in train_ds.samples]
        counts = np.bincount(labels_list, minlength=num_classes)
        inv_freq = 1.0 / (counts + 1e-8)
        weights = torch.tensor(inv_freq, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    model.to(device)
    best_val = 0.0
    metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for X, y, lengths in pbar:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            logits = model(X, lengths)
            loss = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]
        val_acc, val_count, sample_info = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} — Train loss: {avg_train_loss:.4f} | Val Top-1: {val_acc:.4f} | LR: {current_lr:.2e} ({val_count} samples)")

        # Debug: show last validation batch logits and labels
        if sample_info is not None:
            logits_np, labels_np, preds_np = sample_info
            print("Validation sample logits:", logits_np.tolist())
            print("Validation sample labels:", labels_np.tolist())
            print("Validation sample preds:", preds_np.tolist())

        metrics.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_top1': val_acc,
            'val_samples': val_count,
            'lr': current_lr,
        })

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), 'best_transformer.pt')

        # Append metrics to a JSON log file
        with open('training_log.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"Training complete. Best val Top-1: {best_val:.4f}")


if __name__ == '__main__':
    main()
