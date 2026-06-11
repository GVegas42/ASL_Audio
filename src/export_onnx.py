"""
Export the trained transformer checkpoint to ONNX for runtime inference.

Usage:
    python src/export_onnx.py
    python src/export_onnx.py --model best_transformer.pt --output best_transformer.onnx --max-len 64
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from transformer_model import TransformerEncoderClassifier


def load_model(model_path: str):
    state = torch.load(model_path, map_location='cpu')
    input_dim = state['input_proj.weight'].shape[1]
    d_model = state['input_proj.weight'].shape[0]
    num_classes = state['classifier.1.weight'].shape[0]
    layer_keys = [k for k in state if k.startswith('transformer.layers.')]
    num_layers = max(int(k.split('.')[2]) for k in layer_keys) + 1
    nhead = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)

    model = TransformerEncoderClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.0,
    )
    model.load_state_dict(state)
    model.eval()
    return model, input_dim


def export(model_path: str, output_path: str, max_len: int):
    model, input_dim = load_model(model_path)

    # Continuous inference always feeds a full max_len window with no padding,
    # so the exported graph takes raw frames only (lengths=None branch).
    dummy = torch.randn(1, max_len, input_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=['frames'],
        output_names=['logits'],
        dynamic_axes={'frames': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {output_path}")
    print(f"  input_dim={input_dim}  max_len={max_len}  velocity={'on' if input_dim == 126 else 'off'}")

    verify(model, output_path, dummy)


def verify(torch_model, onnx_path: str, dummy: torch.Tensor):
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = torch_model(dummy).numpy()

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {'frames': dummy.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max abs diff (torch vs onnx): {max_diff:.2e}")
    print("Verification PASSED" if max_diff < 1e-4 else "Verification WARNING: outputs differ more than expected")


def main():
    parser = argparse.ArgumentParser(description='Export the trained transformer checkpoint to ONNX.')
    parser.add_argument('--model', default='best_transformer.pt')
    parser.add_argument('--output', default='best_transformer.onnx')
    parser.add_argument('--max-len', type=int, default=64, help='Sliding window size (must match continuous inference)')
    args = parser.parse_args()

    export(args.model, args.output, args.max_len)


if __name__ == '__main__':
    main()
