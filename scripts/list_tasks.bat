@echo off
setlocal

rem List all registered LuckyLab tasks (RL and IL).
rem
rem Usage:
rem   list_tasks.bat

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

uv run python -m luckylab.scripts.list_envs

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to list tasks. Make sure you have run setup first.
    pause
    exit /b 1
)
