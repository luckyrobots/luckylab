@echo off
setlocal

rem Compare capability manifests between engine versions.
rem
rem Usage:
rem   diff_capabilities.bat --save manifest-v1.json
rem   diff_capabilities.bat --old manifest-v1.json --new manifest-v1.1.json
rem   diff_capabilities.bat --old manifest-v1.json --live

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

uv run python -m luckylab.scripts.diff_capabilities %*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Diff failed.
    pause
    exit /b 1
)
