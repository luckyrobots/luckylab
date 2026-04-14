@echo off
setlocal

rem Run a trained RL policy (Go2 velocity — SAC / skrl).
rem First argument is the checkpoint path. Extra arguments are forwarded.
rem
rem Usage:
rem   play_rl.bat runs\go2_velocity_sac\checkpoints\agent_25000.pt
rem   play_rl.bat runs\go2_velocity_sac\checkpoints\agent_25000.pt --keyboard

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

if "%~1"=="" (
    echo Usage: play_rl.bat ^<checkpoint^> [extra args...]
    echo.
    echo Example:
    echo   play_rl.bat runs\go2_velocity_sac\checkpoints\agent_25000.pt
    echo   play_rl.bat runs\go2_velocity_sac\checkpoints\agent_25000.pt --keyboard
    pause
    exit /b 1
)

set CHECKPOINT=%~1
shift

echo Running RL inference (Go2 velocity — SAC / skrl) ...
echo   Checkpoint: %CHECKPOINT%
uv run python -m luckylab.scripts.play go2_velocity_flat --algorithm sac --backend skrl --checkpoint %CHECKPOINT% %1 %2 %3 %4 %5 %6 %7 %8 %9

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Inference failed. Make sure LuckyEngine is running with gRPC enabled.
    pause
    exit /b 1
)
