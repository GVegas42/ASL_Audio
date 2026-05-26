# ASL Audio Pipeline

This repository contains an audio-focused feature extraction pipeline for sign language videos.

## Project structure

- `requirements.txt` - Python dependencies.
- `src/parse_dataset.py` - Create JSON manifests from the WLASL metadata.
- `src/feature_extraction.py` - Extract hand landmarks from videos and save them as NumPy `.npy` files.
- `src/test_pipeline.py` - Sanity checks for manifest existence and feature normalization.
- `data/` - Dataset and processed output.
- `data/models/` - MediaPipe task model asset directory.

## Setup

1. Activate the project venv (or select the workspace `.venv` in VS Code).

```powershell
& "c:/Users/Gustavo Vegas/Documents/GitHub/ASL_Audio/.venv/Scripts/Activate.ps1"
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Place the MediaPipe hand landmarker model bundle at:

```text
data/models/hand_landmarker.task
```

See `data/models/README.md` for details.

## Running the pipeline

1. Generate manifests from WLASL metadata:

```powershell
python src/parse_dataset.py
```

2. Extract hand landmarks into `data/processed_features`:

```powershell
python src/feature_extraction.py
```

3. Run the pipeline verification script:

```powershell
python src/test_pipeline.py
```

## Custom model location

If you store the MediaPipe model bundle in a different path, set the environment variable before running extraction:

```powershell
$env:MP_HAND_LANDMARKER_MODEL_PATH = "C:\path\to\hand_landmarker.task"
python src/feature_extraction.py
```

## Quick Start

1. Open PowerShell and activate the repository venv:

```powershell
& "c:/Users/Gustavo Vegas/Documents/GitHub/ASL_Audio/.venv/Scripts/Activate.ps1"
```

2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Download the MediaPipe hand landmarker bundle and place it in `data/models/` as `hand_landmarker.task`.

4. Generate manifests from the WLASL metadata:

```powershell
python src/parse_dataset.py
```

5. Extract landmarks from your videos:

```powershell
python src/feature_extraction.py
```

6. Run the pipeline verification script:

```powershell
python src/test_pipeline.py
```

## Notes

- `src/feature_extraction.py` now uses the MediaPipe Tasks `HandLandmarker` API.
- The current repo does not include a trained sign-to-text or text-to-speech model.
- The pipeline is currently focused on dataset parsing and landmark feature extraction.
