@echo off
setlocal

rem Visualize a LeRobot dataset in a web-based Rerun viewer.
rem Defaults to the SO-100 pick-and-place dataset, episode 0.
rem Any extra arguments are forwarded to the visualization script.
rem
rem Usage:
rem   visualize_dataset.bat
rem   visualize_dataset.bat --episode-index 3
rem   visualize_dataset.bat --repo-id my_org/my_dataset --episode-index 0

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

echo Starting dataset visualization (web viewer) ...
uv run python -m luckylab.scripts.visualize_dataset --repo-id luckyrobots/so100_pickandplace_sim --episode-index 0 --web %*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Visualization failed. Make sure you have installed with: uv sync --group il
    pause
    exit /b 1
)
