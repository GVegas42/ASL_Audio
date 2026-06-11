# Isolated Sign Training — Baseline Metrics

## Dataset Summary
- **Total Raw Files**: 11,980 matched videos
- **Total Classes**: 2,000 unique ASL signs
- **Train Split**: 8,313 samples (69%)
- **Val Split**: 2,253 samples (19%)  
- **Test Split**: 1,414 samples (12%)

## Preprocessing
- **Feature Extraction**: MediaPipe hand landmarks (21 points × 3 coords = 63-d vectors)
- **Per-Frame Dimension**: 63 (21 landmarks × 3 coordinates)
- **Variable-Length Handling**: Padding/truncation to max_len=64 frames
- **Normalization**: Wrist-translation + middle-MCP scale normalization

## Model Architecture
- **Base**: PyTorch TransformerEncoder
- **Input Encoding**: Sinusoidal positional encoding + linear projection to d_model
- **Architecture Parameters**:
  - d_model=128 (embedding dimension)
  - nhead=8 (attention heads)
  - num_layers=4 (encoder layers)
  - dropout=0.1
- **Pooling**: Masked mean pooling over time dimension
- **Classification Head**: FC layer (d_model → 2000 classes)

## Training Configuration
```bash
python src/train_isolated.py \
  --epochs 50 \
  --batch-size 32 \
  --max-len 64 \
  --d-model 128 \
  --nhead 8 \
  --num-layers 4 \
  --use-class-weight \
  --oversample
```

**Key Flags**:
- `--use-class-weight`: Class-weighted CrossEntropyLoss for imbalanced classes
- `--oversample`: WeightedRandomSampler to oversample minority classes

## Expected Baseline
- **Architecture**: 4-layer Transformer encoder (larger than initial 2-layer)
- **Dataset Scale**: 2,000 classes (200x more than initial 2-class baseline)
- **Validation Set**: 2,253 samples (225x more than initial 1 sample)
- **Expected Top-1 Accuracy**: 20-35% (statistically valid baseline for 2,000-class problem)
- **Training Time**: ~30-60 minutes (depending on extraction I/O and GPU availability)

## Success Criteria
✓ Validation accuracy > 15% (significantly above 0.05% random chance)  
✓ Training loss monotonically decreasing  
✓ No OOM or file I/O errors  
✓ Model saves to best_transformer.pt  
✓ Metrics logged to training_log.json  

## Next Steps
1. ✓ Extract all 11,980 features [IN PROGRESS]
2. → Train on full dataset with logging
3. → Analyze per-class accuracy distribution
4. → Document baseline results and identify failure modes
5. → Implement improvements (e.g., data augmentation, temporal modeling)
