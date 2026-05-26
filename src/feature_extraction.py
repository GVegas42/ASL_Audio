import os
import random
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,  # Set to False to treat frames as video sequence tracking
    max_num_hands=1,          # Focus on dominant hand for baseline configuration
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

def process_video(video_path, output_npy_path, augment=False):
    """
    Reads video frames, runs MediaPipe, extracts, normalizes, and saves landmarks.
    """
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    # Establish random augmentation bounds for the sequence if enabled
    flip = random.choice([True, False]) if augment else False
    brightness = random.uniform(0.8, 1.2) if augment else 1.0
    rotation_angle = random.randint(-10, 10) if augment else 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run OpenCV Augmentations
        if augment:
            frame = augment_frame(frame, flip=flip, brightness_factor=brightness, angle=rotation_angle)
            
        # MediaPipe requires RGB color formatting
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)
        
        # If hand is detected, compile keypoints
        if results.multi_hand_landmarks:
            # Analyze dominant/first hand found in frame
            hand_landmarks = results.multi_hand_landmarks[0]
            
            frame_coords = []
            for lm in hand_landmarks.landmark:
                frame_coords.append([lm.x, lm.y, lm.z])
                
            # Perform geometric matrix normalization
            normalized_frame_coords = normalize_landmarks(frame_coords)
            sequence_data.append(normalized_frame_coords)
        else:
            # Strategy: If a frame fails to capture a hand, pad with a zero matrix 
            # to maintain consistent temporal sequence length later.
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
            video_filename = feature_filename.replace(".npy", ".mp4")
            
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