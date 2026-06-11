@echo off
REM Full training pipeline: verify extraction → split manifests → train
REM This script waits for extraction to complete, then trains the model

setlocal enabledelayedexpansion

set VENV_PYTHON=c:\Users\Gustavo Vegas\Documents\GitHub\ASL_Audio\.venv\Scripts\python.exe
set DATASET_DIR=c:\Users\Gustavo Vegas\Documents\GitHub\ASL_Audio\data\processed_features

echo [%date% %time%] Starting pipeline monitor...

:CHECK_EXTRACTION
echo [%date% %time%] Checking extraction status...
for /f %%A in ('"%VENV_PYTHON%" -c "import os; print(len([f for f in os.listdir(r'%DATASET_DIR%') if f.endswith('.npy')]))"') do (
    set FILES_COUNT=%%A
)
echo [%date% %time%] Feature files extracted: !FILES_COUNT!/11980

if !FILES_COUNT! LSS 11980 (
    echo [%date% %time%] Extraction not yet complete. Waiting 60 seconds...
    timeout /t 60 /nobreak
    goto CHECK_EXTRACTION
)

echo [%date% %time%] Extraction complete! Starting training...
"%VENV_PYTHON%" src\train_isolated.py --epochs 50 --batch-size 32 --max-len 64 --d-model 128 --nhead 8 --num-layers 4 --use-class-weight --oversample

echo [%date% %time%] Training completed!
