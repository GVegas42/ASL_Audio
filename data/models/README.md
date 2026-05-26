# MediaPipe Hand Landmarker Model

This directory is intended to store the MediaPipe Tasks hand landmarker task bundle used by `src/feature_extraction.py`.

## Required file

- `hand_landmarker.task`

This file is not included in the repository. You must download it separately from the official MediaPipe model resources.

## How to use

1. Download the MediaPipe hand landmarker task bundle.
   - The file should be named `hand_landmarker.task`.
   - Place it in `data/models/`.

2. Verify the file exists:

```powershell
Get-ChildItem data\models\hand_landmarker.task
```

3. Run the feature extraction script:

```powershell
& "c:/Users/Gustavo Vegas/Documents/GitHub/ASL_Audio/.venv/Scripts/python.exe" "c:/Users/Gustavo Vegas/Documents/GitHub/ASL_Audio/src/feature_extraction.py"
```

If you place the model somewhere else, set the environment variable first:

```powershell
$env:MP_HAND_LANDMARKER_MODEL_PATH = "C:\path\to\hand_landmarker.task"
& "c:/Users/Gustavo Vegas/Documents/GitHub/ASL_Audio/.venv/Scripts/python.exe" "c:/Users/Gustavo Vegas/Documents/GitHub/ASL_Audio/src/feature_extraction.py"
```

## Notes

- `src/feature_extraction.py` reads the model path from the `MP_HAND_LANDMARKER_MODEL_PATH` environment variable.
- If the file is missing, the script will raise a clear `FileNotFoundError`.
- The model bundle is required for hand landmark detection and feature extraction.
