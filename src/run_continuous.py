"""
Real-time continuous ASL inference from a webcam or video file.

Usage:
    # Live webcam
    python src/run_continuous.py

    # Video file
    python src/run_continuous.py --source path/to/video.mp4

    # Lower confidence threshold to see more predictions
    python src/run_continuous.py --confidence 0.4

Controls (when display window is open):
    q  — quit
    r  — reset the sentence buffer
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from feature_extraction import create_hands_detector, normalize_landmarks
from continuous_inference import ContinuousInference, ONNXContinuousInference
from tts_output import TTSEngine
from mediapipe.tasks.python.vision.core import image as mp_image_mod


def run(
    source: str,
    backend: str,
    model_path: str,
    onnx_path: str,
    class_labels_path: str,
    max_len: int,
    stride: int,
    confidence: float,
    cooldown: int,
    device: str,
    no_display: bool,
    tts: bool,
):
    if backend == 'onnx':
        engine = ONNXContinuousInference(
            onnx_path=onnx_path,
            class_labels_path=class_labels_path,
            stride=stride,
            confidence_threshold=confidence,
            sign_cooldown=cooldown,
        )
        max_len = engine.max_len
    else:
        engine = ContinuousInference(
            model_path=model_path,
            class_labels_path=class_labels_path,
            max_len=max_len,
            stride=stride,
            confidence_threshold=confidence,
            sign_cooldown=cooldown,
            device=device,
        )
    print(
        f"Backend={backend} | velocity={'on' if engine.use_velocity else 'off'} | "
        f"window={max_len}f stride={stride}f conf={confidence}"
    )

    speaker = TTSEngine() if tts else None

    video_source = 0 if source == 'webcam' else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'")
        sys.exit(1)

    if not no_display:
        print("Press 'q' to quit, 'r' to reset sentence.")

    frame_idx = 0
    with create_hands_detector() as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp_image_mod.Image(mp_image_mod.ImageFormat.SRGB, rgb)
            results = detector.detect(mp_img)

            if results and results.hand_landmarks:
                coords = [[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0]]
                landmarks = normalize_landmarks(coords)
            else:
                landmarks = np.zeros((21, 3), dtype=np.float32)

            new_sign = engine.push_frame(landmarks)
            if new_sign:
                print(f"[frame {frame_idx:05d}] >> {new_sign}  |  sentence: {engine.sentence}")
                if speaker:
                    speaker.say(new_sign)

            if not no_display:
                # Sentence overlaid at the top; newest sign briefly highlighted below
                display_sentence = engine.sentence[-80:] if engine.sentence else '(waiting for signs...)'
                cv2.putText(
                    frame, display_sentence,
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
                )
                if new_sign:
                    cv2.putText(
                        frame, f'>> {new_sign}',
                        (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2,
                    )
                cv2.imshow('ASL Continuous Inference', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    engine.reset()
                    print("Sentence reset.")

            frame_idx += 1

    cap.release()
    if not no_display:
        cv2.destroyAllWindows()

    print(f"\nFinal sentence: {engine.sentence or '(none)'}")

    if speaker:
        if engine.sentence:
            speaker.say(engine.sentence)
        speaker.stop()


def main():
    parser = argparse.ArgumentParser(description='Continuous ASL inference from webcam or video.')
    parser.add_argument('--source', default='webcam', help="'webcam' or path to a video file")
    parser.add_argument('--backend', choices=['torch', 'onnx'], default='torch', help='Inference backend')
    parser.add_argument('--model', default='best_transformer.pt', help='PyTorch checkpoint (backend=torch)')
    parser.add_argument('--onnx', default='best_transformer.onnx', help='ONNX model path (backend=onnx)')
    parser.add_argument('--labels', default='data/manifests/class_labels.json')
    parser.add_argument('--max-len', type=int, default=64, help='Sliding window size in frames (backend=torch)')
    parser.add_argument('--stride', type=int, default=8, help='Frames between classifications')
    parser.add_argument('--confidence', type=float, default=0.6, help='Min softmax confidence to emit a sign')
    parser.add_argument('--cooldown', type=int, default=16, help='Min frames before same sign emits again')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-display', action='store_true', help='Run headless (no OpenCV window)')
    parser.add_argument('--tts', action='store_true', help='Speak recognized signs aloud')
    args = parser.parse_args()

    run(
        source=args.source,
        backend=args.backend,
        model_path=args.model,
        onnx_path=args.onnx,
        class_labels_path=args.labels,
        max_len=args.max_len,
        stride=args.stride,
        confidence=args.confidence,
        cooldown=args.cooldown,
        device=args.device,
        no_display=args.no_display,
        tts=args.tts,
    )


if __name__ == '__main__':
    main()
