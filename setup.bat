@echo off
setlocal

echo Checking for uv...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo Failed to install uv.
        exit /b 1
    )
    echo uv installed successfully.
) else (
    echo uv is already installed.
)

echo.
echo Running uv sync --all-groups...
uv sync --all-groups
if %errorlevel% neq 0 (
    echo uv sync failed.
    exit /b 1
)

echo.
echo Installing pre-commit hooks...
uv run pre-commit install
if %errorlevel% neq 0 (
    echo pre-commit install failed.
    exit /b 1
)

echo.
echo Setup complete!
endlocal
