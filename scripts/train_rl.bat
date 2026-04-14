@echo off
setlocal

rem Train RL — Go2 velocity tracking with SAC/skrl on CUDA.
rem Any extra arguments are forwarded to the training script.
rem
rem Usage:
rem   train_rl.bat
rem   train_rl.bat --device cpu
rem   train_rl.bat --agent.algorithm ppo

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

echo Starting RL training (Go2 velocity — SAC / skrl / cuda) ...
uv run python -m luckylab.scripts.train go2_velocity_flat --agent.algorithm sac --agent.backend skrl --device cuda %*

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed. Make sure LuckyEngine is running with gRPC enabled.
    pause
    exit /b 1
)
