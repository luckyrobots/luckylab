@echo off
setlocal

rem Run a trained IL policy (SO-100 pick-and-place — ACT).
rem First argument is the checkpoint path. Extra arguments are forwarded.
rem
rem Usage:
rem   play_il.bat runs\so100_pickandplace_act\final
rem   play_il.bat runs\so100_pickandplace_act\final --episodes 20

set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

cd /d "%ROOT_DIR%"

if "%~1"=="" (
    echo Usage: play_il.bat ^<checkpoint^> [extra args...]
    echo.
    echo Example:
    echo   play_il.bat runs\so100_pickandplace_act\final
    echo   play_il.bat runs\so100_pickandplace_act\final --episodes 20
    pause
    exit /b 1
)

set CHECKPOINT=%~1
shift

echo Running IL inference (SO-100 pick-and-place — ACT) ...
echo   Checkpoint: %CHECKPOINT%
uv run python -m luckylab.scripts.play so100_pickandplace --policy act --checkpoint %CHECKPOINT% %1 %2 %3 %4 %5 %6 %7 %8 %9

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Inference failed. Make sure LuckyEngine is running with gRPC enabled.
    pause
    exit /b 1
)
