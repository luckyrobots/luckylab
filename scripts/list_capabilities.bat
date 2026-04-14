@echo off
setlocal

rem List available MDP capabilities from a running LuckyEngine instance.
rem
rem Usage:
rem   list_capabilities.bat
rem   list_capabilities.bat --robot unitreego2

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

uv run python -m luckylab.scripts.list_capabilities %*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to list capabilities. Make sure LuckyEngine is running.
    pause
    exit /b 1
)
