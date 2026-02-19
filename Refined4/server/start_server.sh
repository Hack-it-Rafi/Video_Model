#!/bin/bash

echo "Starting Video Classification Server..."
echo ""
echo "Checking dependencies..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: ffmpeg is not installed or not in PATH"
    echo "Please install ffmpeg to enable video conversion and chunking"
    echo ""
fi

# Check if model exists
if [ ! -f "../model.pth" ]; then
    echo "Warning: model.pth not found in parent directory"
    echo "Please train the model first: python main.py train"
    echo ""
fi

echo ""
echo "Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 app.py
