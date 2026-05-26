import os
import random
import cv2
import numpy as np

from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision import hand_landmarker

MP_HAND_LANDMARKER_MODEL_PATH = os.getenv(
    "MP_HAND_LANDMARKER_MODEL_PATH",
    "data/models/hand_landmarker.task"
)


def create_hands_detector(
    model_path: str | None = None,
    num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    model_path = model_path or MP_HAND_LANDMARKER_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Hand landmarker model not found at '{model_path}'. "
            "Please download a compatible MediaPipe hand_landmarker task bundle "
            "and set MP_HAND_LANDMARKER_MODEL_PATH to its path."
        )

    base_options = mp_base_options.BaseOptions(model_asset_path=model_path)
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return hand_landmarker.HandLandmarker.create_from_options(options)


def augment_frame(frame, flip=False, brightness_factor=1.0, angle=0):
    """
    Applies data augmentations using pure OpenCV operations.
    """
    augmented = frame.copy()
    
    # 1. Handle left vs right-handedness via horizontal flipping
    if flip:
        augmented = cv2.flip(augmented, 1)
        
    # 2. Adjust Brightness
    if brightness_factor != 1.0:
        # Convert to float to avoid uint8 overflow issues
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
    # 3. Handle subtle spatial rotations
    if angle != 0:
        h, w = augmented.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h))
        
    return augmented

def normalize_landmarks(landmarks_list):
    """
    Normalizes a (21, 3) matrix of coordinates.
    1. Translation Invariance: Subtracts wrist coordinates from all other points.
    2. Scale Invariance: Scales points relative to wrist-to-middle-finger-base distance.
    """
    coords = np.array(landmarks_list) # Shape: (21, 3)
    
    # WRIST is index 0 in MediaPipe Hand Landmark Topology
    wrist = coords[0]
    
    # Step 1: Translation Invariance (Shift origin to wrist)
    translated_coords = coords - wrist
    
    # MIDDLE_FINGER_MCP (Base of middle finger) is index 9
    mcp_base = translated_coords[9]
    
    # Step 2: Compute Euclidean scale metric from wrist (now 0,0,0) to MCP Base
    scale_factor = np.linalg.norm(mcp_base)
    
    # Avoid zero-division if hand landmarks collapse or fail to track cleanly
    if scale_factor == 0:
        return translated_coords
        
    # Step 3: Scale Invariance
    normalized_coords = translated_coords / scale_factor
    return normalized_coords

def process_video(video_path, output_npy_path, augment=False, model_path=None):
    """
    Reads video frames, runs MediaPipe, extracts, normalizes, and saves landmarks.
    """
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    # Establish random augmentation bounds for the sequence if enabled
    flip = random.choice([True, False]) if augment else False
    brightness = random.uniform(0.8, 1.2) if augment else 1.0
    rotation_angle = random.randint(-10, 10) if augment else 0

    with create_hands_detector(model_path=model_path) as hands_detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run OpenCV augmentations
            if augment:
                frame = augment_frame(frame, flip=flip, brightness_factor=brightness, angle=rotation_angle)

            # MediaPipe Tasks expects RGB pixel data.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_frame)
            results = hands_detector.detect(mp_image_obj)

            if results and results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                frame_coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                normalized_frame_coords = normalize_landmarks(frame_coords)
                sequence_data.append(normalized_frame_coords)
            else:
                sequence_data.append(np.zeros((21, 3)))
            
    cap.release()
    
    # Shape: (Total Frames, 21 Landmarks, 3 Axis Dimensions)
    sequence_array = np.array(sequence_data)
    
    # Save processed tensor as a NumPy array binary file
    np.save(output_npy_path, sequence_array)
    print(f"Successfully saved features: {output_npy_path} | Shape: {sequence_array.shape}")

if __name__ == "__main__":
    import json

    manifest_dir = "data/manifests"
    raw_video_dir = "data/raw_videos"
    output_feature_dir = "data/processed_features"
    os.makedirs(output_feature_dir, exist_ok=True)

    # We want to process all splits
    splits = ["train", "val", "test"]

    for split in splits:
        manifest_path = os.path.join(manifest_dir, f"{split}.json")
        
        if not os.path.exists(manifest_path):
            print(f"Manifest not found for split: {split}. Skipping.")
            continue

        with open(manifest_path, 'r') as f:
            samples = json.load(f)

        print(f"\n--- Processing {split} split ({len(samples)} samples) ---")
        
        for sample in samples:
            # Reconstruct paths based on the manifest entries
            # e.g., feature_file might be "06849.npy", video is "06849.mp4"
            feature_filename = sample["feature_file"]
            video_filename = os.path.splitext(feature_filename)[0] + ".mp4"
            
            full_video_path = os.path.join(raw_video_dir, video_filename)
            full_output_path = os.path.join(output_feature_dir, feature_filename)

            # CRITICAL CV PRINCIPLE: Only augment the training data!
            # Augmenting validation or test data skews evaluation metrics.
            should_augment = True if split == "train" else False

            if os.path.exists(full_video_path):
                process_video(full_video_path, full_output_path, augment=should_augment)
            else:
                print(f"Warning: Expected video missing at {full_video_path}")

    print("\nWeek 1 Feature Extraction Complete! Data is ready for Week 2 modeling.")