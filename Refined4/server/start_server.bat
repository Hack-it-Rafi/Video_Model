@echo off
echo Starting Video Classification Server...
echo.
echo Checking dependencies...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if ffmpeg is available
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo Warning: ffmpeg is not installed or not in PATH
    echo Please install ffmpeg to enable video conversion and chunking
    echo.
)

REM Check if model exists
if not exist "..\model.pth" (
    echo Warning: model.pth not found in parent directory
    echo Please train the model first: python main.py train
    echo.
)

echo.
echo Starting Flask server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

cd %~dp0
python app.py

pause
