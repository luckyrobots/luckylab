@echo off
setlocal

set REPO=luckyrobots/luckylab
set TAG=demo-v0.1.0
set DEMO_NAME=piper_blockstacking_act
set ZIP_NAME=%DEMO_NAME%.zip
set DOWNLOAD_URL=https://github.com/%REPO%/releases/download/%TAG%/%ZIP_NAME%

rem Resolve the directory this script lives in (the luckylab root)
rem %~dp0 has a trailing backslash — remove it so quoted paths don't break
set ROOT_DIR=%~dp0..
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

echo Downloading demo from %DOWNLOAD_URL% ...
curl -L "%DOWNLOAD_URL%" -o "%ROOT_DIR%\%ZIP_NAME%"
if %errorlevel% neq 0 (
    echo ERROR: Download failed. Make sure curl is available and the URL is correct.
    pause
    exit /b 1
)

echo Extracting demo ...
powershell -Command "Expand-Archive -Path '%ROOT_DIR%\%ZIP_NAME%' -DestinationPath '%ROOT_DIR%' -Force"
if %errorlevel% neq 0 (
    echo ERROR: Extraction failed.
    pause
    exit /b 1
)

del "%ROOT_DIR%\%ZIP_NAME%"

echo.
echo Demo installed successfully.
echo   Model:  runs\%DEMO_NAME%\final\
echo   Script: run_demo.bat
echo.
echo Run 'run_demo.bat' to start the demo.
pause
