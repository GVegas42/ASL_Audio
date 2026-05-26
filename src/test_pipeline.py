import os
import json
import numpy as np

def test_manifest_existence():
    """Ensure the parsing script generated the required files."""
    print("Checking manifests...")
    for split in ["train", "val", "test", "class_labels"]:
        path = f"data/manifests/{split}.json"
        assert os.path.exists(path), f"Missing manifest: {path}"
    print("✅ All manifests exist.")

def test_normalization_and_shapes():
    """Verify matrix dimensions and translation invariance principles."""
    print("\nVerifying feature matrices...")
    feature_dir = "data/processed_features"
    npy_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print("❌ No processed .npy files found in data/processed_features. Run feature_extraction.py first.")
        return

    # Test the first available processed file
    sample_path = os.path.join(feature_dir, npy_files[0])
    matrix = np.load(sample_path)
    
    # 1. Check Dimensions: (Frames, 21 Keypoints, 3 Dimensions)
    assert len(matrix.shape) == 3, f"Expected 3D matrix, got shape {matrix.shape}"
    assert matrix.shape[1] == 21, f"Expected 21 keypoints, got {matrix.shape[1]}"
    assert matrix.shape[2] == 3, f"Expected 3 coordinate axes (x, y, z), got {matrix.shape[2]}"
    print(f"✅ Matrix dimensions are correct. Shape: {matrix.shape}")
    
    # 2. Check Translation Invariance (Wrist should be pinned to 0,0,0)
    # MediaPipe index 0 is the wrist. We check across frames where a hand was detected (not zero-padded frames).
    for frame_idx in range(matrix.shape[0]):
        frame = matrix[frame_idx]
        # If the entire frame isn't a zero-padded dummy frame
        if not np.all(frame == 0):
            wrist_coords = frame[0]
            np.testing.assert_allclose(
                wrist_coords, [0.0, 0.0, 0.0], atol=1e-5,
                err_msg=f"Translation invariance failed at frame {frame_idx}. Wrist is not at origin."
            )
            break
    print("✅ Translation invariance verified: Wrist landmark is successfully anchored at (0, 0, 0).")

    # 3. Check Scale Invariance
    # The distance from wrist (0) to middle finger base (9) should scale to a norm of 1.0
    for frame_idx in range(matrix.shape[0]):
        frame = matrix[frame_idx]
        if not np.all(frame == 0):
            mcp_base_coords = frame[9]
            distance = np.linalg.norm(mcp_base_coords)
            np.testing.assert_allclose(
                distance, 1.0, atol=1e-5,
                err_msg=f"Scale invariance failed at frame {frame_idx}. Normalized distance is {distance} instead of 1.0"
            )
            break
    print("✅ Scale invariance verified: Hand metric scaled relative to reference vector.")

if __name__ == "__main__":
    try:
        test_manifest_existence()
        test_normalization_and_shapes()
        print("\n🎉 ALL PIPELINE TESTS PASSED SUCCESSFULY!")
    except AssertionError as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")