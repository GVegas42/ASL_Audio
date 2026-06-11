import collections
import json
from typing import Optional

import numpy as np
import torch

from transformer_model import TransformerEncoderClassifier


class _BaseContinuousInference:
    """
    Shared sliding-window bookkeeping for continuous ASL recognition.

    Accepts a stream of per-frame hand landmarks, classifies over a rolling
    window, debounces repeated predictions, and accumulates recognized signs
    into a sentence buffer. Subclasses provide `_predict()`.
    """

    def __init__(
        self,
        class_labels_path: str,
        max_len: int,
        stride: int,
        confidence_threshold: float,
        sign_cooldown: int,
        use_velocity: bool,
    ):
        self.max_len = max_len
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.sign_cooldown = sign_cooldown
        self.use_velocity = use_velocity

        with open(class_labels_path, 'r') as f:
            label_map = json.load(f)
        self.idx_to_gloss: dict[int, str] = {v: k for k, v in label_map.items()}

        self._frame_buffer: collections.deque = collections.deque(maxlen=max_len)
        self._frames_since_classify: int = 0
        self._frames_since_emit: int = 0
        self._last_emitted: Optional[str] = None
        self._sentence: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(self, landmarks: np.ndarray) -> Optional[str]:
        """
        Ingest one frame and return a newly emitted sign gloss if one was
        recognized this frame, otherwise None.

        Args:
            landmarks: (21, 3) float32 array of normalized hand landmarks.
                       Pass np.zeros((21, 3)) for frames with no hand detected.
        """
        self._frame_buffer.append(landmarks.astype(np.float32))
        self._frames_since_classify += 1
        self._frames_since_emit += 1

        if len(self._frame_buffer) < self.max_len:
            return None

        if self._frames_since_classify < self.stride:
            return None

        self._frames_since_classify = 0
        return self._classify_and_emit()

    @property
    def sentence(self) -> str:
        """Accumulated recognized signs as a space-joined string."""
        return ' '.join(self._sentence)

    def reset(self):
        """Clear the frame buffer and sentence buffer between utterances."""
        self._frame_buffer.clear()
        self._sentence.clear()
        self._last_emitted = None
        self._frames_since_classify = 0
        self._frames_since_emit = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_input(self) -> np.ndarray:
        frames = np.array(self._frame_buffer, dtype=np.float32)  # (max_len, 21, 3)

        if self.use_velocity:
            vel = np.diff(frames, axis=0)
            vel = np.concatenate(
                [np.zeros((1, 21, 3), dtype=np.float32), vel], axis=0
            )
            frames = np.concatenate([frames, vel], axis=2)  # (max_len, 21, 6)

        frames = frames.reshape(self.max_len, -1)
        return frames[np.newaxis, ...]  # (1, max_len, D)

    def _predict(self, x: np.ndarray) -> tuple[float, int]:
        """Return (confidence, predicted_class_idx) for input (1, max_len, D)."""
        raise NotImplementedError

    def _classify_and_emit(self) -> Optional[str]:
        x = self._prepare_input()
        confidence, pred_idx = self._predict(x)

        if confidence < self.confidence_threshold:
            return None

        gloss = self.idx_to_gloss.get(pred_idx, f'[{pred_idx}]')

        # Suppress re-emission of the same sign before the cooldown elapses
        if gloss == self._last_emitted and self._frames_since_emit < self.sign_cooldown:
            return None

        self._last_emitted = gloss
        self._frames_since_emit = 0
        self._sentence.append(gloss)
        return gloss


class ContinuousInference(_BaseContinuousInference):
    """PyTorch-backed continuous inference engine."""

    def __init__(
        self,
        model_path: str,
        class_labels_path: str,
        max_len: int = 64,
        stride: int = 8,
        confidence_threshold: float = 0.6,
        sign_cooldown: int = 16,
        device: str = 'cpu',
    ):
        """
        Args:
            model_path: Path to best_transformer.pt checkpoint.
            class_labels_path: Path to class_labels.json.
            max_len: Sliding window size in frames (must match training max_len).
            stride: Frames between successive classification attempts.
            confidence_threshold: Minimum softmax confidence to accept a prediction.
            sign_cooldown: Minimum frames before the same sign can be emitted again,
                           preventing rapid stutter on held poses.
            device: 'cuda' or 'cpu'.
        """
        self.device = torch.device(device)

        # Infer all model hyperparameters from the checkpoint weights
        state = torch.load(model_path, map_location=self.device)
        input_dim = state['input_proj.weight'].shape[1]
        d_model = state['input_proj.weight'].shape[0]
        num_classes = state['classifier.1.weight'].shape[0]
        layer_keys = [k for k in state if k.startswith('transformer.layers.')]
        num_layers = max(int(k.split('.')[2]) for k in layer_keys) + 1
        nhead = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)

        # Velocity mode is signalled by input_dim=126 (21 landmarks x 6 = pos+vel)
        use_velocity = (input_dim == 126)

        super().__init__(
            class_labels_path=class_labels_path,
            max_len=max_len,
            stride=stride,
            confidence_threshold=confidence_threshold,
            sign_cooldown=sign_cooldown,
            use_velocity=use_velocity,
        )

        self.model = TransformerEncoderClassifier(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=0.0,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _predict(self, x: np.ndarray) -> tuple[float, int]:
        x_t = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)
        return confidence.item(), pred_idx.item()


class ONNXContinuousInference(_BaseContinuousInference):
    """ONNX Runtime-backed continuous inference engine."""

    def __init__(
        self,
        onnx_path: str,
        class_labels_path: str,
        stride: int = 8,
        confidence_threshold: float = 0.6,
        sign_cooldown: int = 16,
        providers: Optional[list[str]] = None,
    ):
        """
        Args:
            onnx_path: Path to the exported ONNX model (see src/export_onnx.py).
            class_labels_path: Path to class_labels.json.
            stride: Frames between successive classification attempts.
            confidence_threshold: Minimum softmax confidence to accept a prediction.
            sign_cooldown: Minimum frames before the same sign can be emitted again.
            providers: ONNX Runtime execution providers (defaults to CPU).
        """
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ['CPUExecutionProvider']
        )
        self._input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape  # [batch, max_len, input_dim]
        max_len = int(input_shape[1])
        input_dim = int(input_shape[2])

        super().__init__(
            class_labels_path=class_labels_path,
            max_len=max_len,
            stride=stride,
            confidence_threshold=confidence_threshold,
            sign_cooldown=sign_cooldown,
            use_velocity=(input_dim == 126),
        )

    def _predict(self, x: np.ndarray) -> tuple[float, int]:
        logits = self.session.run(None, {self._input_name: x})[0]
        logits = logits[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        pred_idx = int(np.argmax(probs))
        return float(probs[pred_idx]), pred_idx
