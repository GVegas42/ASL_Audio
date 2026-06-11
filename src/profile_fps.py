"""
Profile inference latency/FPS for the PyTorch and ONNX Runtime backends to
verify the model meets real-time constraints for continuous inference.

Usage:
    python src/profile_fps.py
    python src/profile_fps.py --model best_transformer.pt --onnx best_transformer.onnx
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from export_onnx import load_model


def profile_torch(model, dummy: torch.Tensor, iters: int, warmup: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        start = time.perf_counter()
        for _ in range(iters):
            model(dummy)
        return time.perf_counter() - start


def profile_onnx(onnx_path: str, dummy_np: np.ndarray, iters: int, warmup: int) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {input_name: dummy_np})
    start = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {input_name: dummy_np})
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description='Profile inference FPS for torch vs ONNX backends.')
    parser.add_argument('--model', default='best_transformer.pt')
    parser.add_argument('--onnx', default='best_transformer.onnx')
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--stride', type=int, default=8, help='Frames between classifications in continuous inference')
    parser.add_argument('--camera-fps', type=float, default=30.0)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()

    model, input_dim = load_model(args.model)
    dummy = torch.randn(1, args.max_len, input_dim, dtype=torch.float32)
    dummy_np = dummy.numpy()

    torch_elapsed = profile_torch(model, dummy, args.iters, args.warmup)
    torch_ms = torch_elapsed * 1000 / args.iters
    print(f"PyTorch  : {torch_ms:.2f} ms/inference -> {1000 / torch_ms:.1f} inferences/sec")

    if os.path.exists(args.onnx):
        onnx_elapsed = profile_onnx(args.onnx, dummy_np, args.iters, args.warmup)
        onnx_ms = onnx_elapsed * 1000 / args.iters
        print(f"ONNX RT  : {onnx_ms:.2f} ms/inference -> {1000 / onnx_ms:.1f} inferences/sec")
        print(f"Speedup  : {torch_elapsed / onnx_elapsed:.2f}x")
    else:
        onnx_ms = None
        print(f"ONNX model not found at {args.onnx} -- run src/export_onnx.py first")

    # In continuous inference, classification only runs once every `stride`
    # frames, so the real-time budget is the time between strided windows.
    budget_ms = args.stride / args.camera_fps * 1000
    print(f"\nReal-time budget at {args.camera_fps:.0f} FPS camera, stride={args.stride}: {budget_ms:.2f} ms")
    for name, ms in (('PyTorch', torch_ms), ('ONNX RT', onnx_ms)):
        if ms is None:
            continue
        verdict = 'OK' if ms < budget_ms else 'TOO SLOW'
        print(f"  {name:8s}: {ms:.2f} ms  [{verdict}]")


if __name__ == '__main__':
    main()
