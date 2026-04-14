@echo off
setlocal

rem Validate a task contract against engine capabilities.
rem
rem Usage:
rem   validate_task.bat go2_velocity_flat
rem   validate_task.bat go2_velocity_flat --host localhost --port 50051

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

uv run python -m luckylab.scripts.validate_task %*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Validation failed. Make sure LuckyEngine is running.
    pause
    exit /b 1
)
