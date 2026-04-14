@echo off
setlocal

rem Train IL — SO-100 pick-and-place with ACT policy on CUDA.
rem Any extra arguments are forwarded to the training script.
rem
rem Usage:
rem   train_il.bat
rem   train_il.bat --device cpu
rem   train_il.bat --il.dataset-repo-id my_org/my_dataset

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

echo Starting IL training (SO-100 pick-and-place — ACT / cuda) ...
uv run python -m luckylab.scripts.train so100_pickandplace --il.policy act --il.dataset-repo-id luckyrobots/so100_pickandplace_sim --device cuda %*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed. Make sure you have installed with: uv sync --group il
    pause
    exit /b 1
)
