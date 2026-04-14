@echo off
setlocal

rem Multi-camera stress test for gRPC camera streaming.
rem Auto-discovers all cameras in the scene and streams them simultaneously.
rem Usage: run_multicam_test.bat [host]

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

set HOST=%1
if "%HOST%"=="" set HOST=127.0.0.1

echo Starting multi-camera stress test (host=%HOST%) ...
uv run --no-sync --group il python grpc_multicam_test.py --width 256 --height 256 --wiggle --host %HOST%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Test failed to run. Make sure you have installed luckylab with:
    echo   uv sync --group il
    pause
    exit /b 1
)
