@echo off
setlocal

rem Resolve the directory this script lives in (the luckylab root)
set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

echo Starting gRPC debug viewer (wiggle mode) ...
uv run --no-sync --group il python grpc_debug_viewer.py --cameras Camera --width 256 --height 256 --wiggle

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Debug viewer failed to run. Make sure you have installed luckylab with:
    echo   uv sync --group il
    pause
    exit /b 1
)
