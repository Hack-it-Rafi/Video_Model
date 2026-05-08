# Video Action & App Classification System

A deep learning system for classifying user actions (click, scroll, type, etc.) and target applications (Chrome, VSCode, Excel, etc.) from screen recording videos using a Multiscale Vision Transformer (MViT) model with temporal smoothing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Running Inference](#running-inference)
- [Server Deployment](#server-deployment)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## Overview

This project implements a video classification system that:

1. Processes screen recording videos split into 3-second chunks
2. Extracts video features using a pretrained MViT (Multiscale Vision Transformer) model
3. Predicts both the user action and target application for each video chunk
4. Applies temporal smoothing across adjacent chunks for stable predictions
5. Provides a REST API server for real-time video classification

## Features

### Model Features

- **Multi-task Learning**: Simultaneously predicts actions and applications
- **Temporal Modeling**: MViT-based architecture captures temporal dynamics
- **Class Imbalance Handling**: Weighted loss functions for imbalanced datasets
- **Data Augmentation**: Temporal and spatial augmentations for better generalization
- **Temporal Smoothing**: Smooths predictions across adjacent chunks (±1 chunk window)
- **Mixed Precision Training**: Faster training with reduced memory usage

### Server Features

- **REST API**: Simple HTTP endpoints for video upload and prediction
- **Multi-format Support**: Accepts mp4, avi, mov, mkv, flv, wmv, webm, mpeg formats
- **Automatic Preprocessing**: Converts videos and splits into 3-second chunks
- **Batch Processing**: Processes all chunks with temporal smoothing
- **Detailed Results**: Returns per-chunk predictions with confidence scores
- **Summary Statistics**: Provides aggregated statistics and distributions

## Architecture

```
Input Video (3 seconds, ~30 fps)
    ↓
Temporal Sampling (16 frames)
    ↓
Spatial Resizing (224×224)
    ↓
MViT Backbone (Pretrained on Kinetics-400)
    ↓
Feature Extraction (768-dim)
    ↓
┌─────────────────┬─────────────────┐
│   Action Head   │   App Head      │
│   (512→N)       │   (256→M)       │
└─────────────────┴─────────────────┘
    ↓                   ↓
Action Prediction   App Prediction
```

## Prerequisites

### System Requirements

- **Python**: 3.8+ (tested with 3.13)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum 8GB VRAM for batch_size=2
  - CPU training is possible but slower
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for model cache and datasets

### Software Dependencies

- **ffmpeg** (required for video processing)
- **CUDA** (optional, for GPU acceleration)

### Installing ffmpeg

**Windows:**

```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

**Linux:**

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

Verify installation:

```bash
ffmpeg -version
ffprobe -version
```

## Installation

### 1. Clone or Navigate to the Project

```bash
cd f:\Attck\Label\POC\Refined4
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- `torch>=2.0.0` (with CUDA support for GPU)
- `torchvision>=0.15.0`
- `pandas`
- `scikit-learn`
- `flask` (for server)
- `opencv-python`

### 4. Verify CUDA (if using GPU)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Setup

### Dataset Structure

Your data should be organized as follows:

```
Refined4/
  data/
    videos_001/
      annotations.csv
      chunks/
        chunk_000.mp4
        chunk_001.mp4
        ...
    videos_002/
      annotations.csv
      chunks/
        chunk_000.mp4
        chunk_001.mp4
        ...
    ...
```

### Annotations Format

Each `annotations.csv` should have the following columns:

- `filename`: Name of the video file (e.g., `chunk_000.mp4`)
- `action`: Action label (e.g., `click`, `scroll`, `type`)
- `target_app`: Application label (e.g., `chrome`, `vscode`, `excel`)

Example:

```csv
filename,action,target_app
chunk_000.mp4,click,chrome
chunk_001.mp4,scroll,chrome
chunk_002.mp4,type,vscode
```

### Preparing Your Data

If you have full videos instead of chunks:

1. Use the chunking script:

```bash
python chunk_videos.py --input your_video.mp4 --output data/videos_001/chunks
```

2. Create annotations manually or use labeling tools

## Model Training

### Quick Start

```bash
python main.py train
```

This will:

1. Load and preprocess the dataset
2. Split data into train/test sets (80/20 by video ID)
3. Compute class weights for imbalanced data
4. Train the MViT model for 20 epochs
5. Save the best model to `model.pth`
6. Save label encoders to `encoders.pkl`

### Training Configuration

Edit `config.py` to customize training:

```python
BATCH_SIZE = 2           # Reduce if running out of memory
EPOCHS = 20              # Number of training epochs
LEARNING_RATE = 5e-5     # Learning rate
WEIGHT_DECAY = 1e-3      # L2 regularization
PATIENCE = 5             # Early stopping patience
```

### Training Output

During training, you'll see:

- Epoch progress with loss values
- Validation metrics (accuracy, precision, recall, F1)
- Best model checkpoints
- Training time per epoch

Example output:

```
Epoch 1/20: 100%|████████| 50/50 [02:15<00:00]
Train Loss: 2.456 | Val Loss: 2.123
Val Action Acc: 0.456 | Val App Acc: 0.678
Best model saved!
```

### Monitoring Training

The trainer automatically:

- Saves the best model based on combined accuracy
- Applies early stopping if no improvement for 5 epochs
- Uses cosine learning rate scheduling with warmup
- Clips gradients to prevent exploding gradients

## Model Evaluation

### Run Evaluation

```bash
python main.py evaluate
```

This will:

1. Load the trained model from `model.pth`
2. Run inference on the test set
3. Apply temporal smoothing
4. Print detailed metrics

### Evaluation Metrics

The evaluation provides:

- **Overall Accuracy**: Percentage of correct predictions
- **Per-class Metrics**: Precision, Recall, F1 for each action/app
- **Confusion Matrix**: Visual representation of predictions
- **Top-K Accuracy**: Top-3 and Top-5 accuracy
- **Confidence Statistics**: Average confidence scores

## Running Inference

### Inference on Test Set

```bash
python main.py infer
```

This generates `predictions.json` with predictions for all test videos:

```json
[
  {
    "video_id": "videos_001",
    "chunk_id": "chunk_000",
    "action": "click",
    "action_conf": 0.89,
    "app": "chrome",
    "app_conf": 0.95
  },
  ...
]
```

### Inference Summary

The script also prints a summary:

```
=== Predictions Summary by Video ===

videos_001:
  chunk_000: click (0.89) | chrome (0.95)
  chunk_001: scroll (0.76) | chrome (0.92)
  chunk_002: type (0.83) | vscode (0.88)
...
```

## Server Deployment

### Starting the Server

```bash
cd server
python app.py
```

Or from the Refined4 directory:

```bash
python server/app.py
```

The server will start on `http://0.0.0.0:5000`

### Server Startup Output

```
Starting server on http://0.0.0.0:5000
Using device: cuda
Model loaded: True
Loading model and encoders...
Model initialized with 50 action classes and 10 app classes
 * Running on http://0.0.0.0:5000
```

### Windows Batch Script

For convenience on Windows:

```bash
cd server
start_server.bat
```

### Linux/macOS Shell Script

```bash
cd server
chmod +x start_server.sh
./start_server.sh
```

## API Usage

### Health Check

Check if the server is running:

```bash
curl http://localhost:5000/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Model Information

Get information about loaded model:

```bash
curl http://localhost:5000/info
```

Response:

```json
{
  "model_info": {
    "num_action_classes": 50,
    "num_app_classes": 10,
    "action_classes": ["click", "scroll", "type", ...],
    "app_classes": ["chrome", "vscode", "excel", ...],
    "device": "cuda"
  }
}
```

### Video Prediction

Upload a video for prediction:

```bash
curl -X POST -F "video=@my_video.mp4" http://localhost:5000/predict
```

Or use the test client:

```bash
cd server
python test_client.py path/to/video.mp4
```

### Response Format

```json
{
  "request_id": "uuid-here",
  "video_info": {
    "original_filename": "my_video.mp4",
    "duration_seconds": 15.5,
    "num_chunks": 6,
    "chunk_duration": 3
  },
  "predictions": [
    {
      "chunk_id": "chunk_000",
      "chunk_number": 0,
      "time_range": "0s-3s",
      "action": "click",
      "action_confidence": 0.892,
      "app": "chrome",
      "app_confidence": 0.945
    },
    ...
  ],
  "summary": {
    "total_chunks": 6,
    "most_common_action": "click",
    "most_common_app": "chrome",
    "action_distribution": {"click": 3, "scroll": 2, "type": 1},
    "app_distribution": {"chrome": 5, "vscode": 1},
    "average_action_confidence": 0.856,
    "average_app_confidence": 0.923
  }
}
```

### Python Client Example

```python
import requests

url = "http://localhost:5000/predict"
files = {'video': open('my_video.mp4', 'rb')}
response = requests.post(url, files=files)
data = response.json()

# Print summary
print(f"Video: {data['video_info']['original_filename']}")
print(f"Duration: {data['video_info']['duration_seconds']}s")
print(f"Most common action: {data['summary']['most_common_action']}")
print(f"Most common app: {data['summary']['most_common_app']}")

# Print per-chunk predictions
for pred in data['predictions']:
    print(f"{pred['time_range']}: {pred['action']} ({pred['action_confidence']:.2f}) | {pred['app']} ({pred['app_confidence']:.2f})")
```

## Configuration

### Model Configuration (`config.py`)

```python
# Data paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

# Video config
NUM_FRAMES = 16  # Number of frames to sample from each 3-second video

# Model config
MODEL_NAME = 'mvit_v1_b'  # MViT base model from torchvision
PRETRAINED = True  # Use pretrained weights from Kinetics-400

# Training config
BATCH_SIZE = 2  # Adjust based on GPU memory
EPOCHS = 20
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss balancing
ACTION_LOSS_WEIGHT = 2.0  # Action is harder, give it more weight
APP_LOSS_WEIGHT = 1.0

# Temporal smoothing
WINDOW_SIZE = 3  # ±1 clip, total 3 chunks
```

### Server Configuration (`server/app.py`)

```python
# File upload limits
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Allowed video formats
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', 'mpg'}

# Chunk duration
chunk_duration = 3  # seconds
```

## Troubleshooting

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. Reduce batch size in `config.py`:
   ```python
   BATCH_SIZE = 1
   ```
2. Use CPU instead:
   ```python
   DEVICE = 'cpu'
   ```
3. Close other GPU-using applications

### ffmpeg Not Found

**Problem**: `FileNotFoundError: ffmpeg not found`

**Solution**: Install ffmpeg and add to PATH (see [Installing ffmpeg](#installing-ffmpeg))

### Model Not Found

**Problem**: `FileNotFoundError: model.pth not found`

**Solution**: Train the model first:

```bash
python main.py train
```

### Low Accuracy

**Problem**: Model accuracy is too low

**Solutions**:

1. Check dataset quality and annotations
2. Increase training epochs:
   ```python
   EPOCHS = 30
   ```
3. Adjust learning rate:
   ```python
   LEARNING_RATE = 1e-4
   ```
4. Add more training data
5. Check class balance and adjust weights

### Slow Inference

**Problem**: Inference is too slow

**Solutions**:

1. Use GPU instead of CPU
2. Reduce video resolution (modify transforms in `dataset.py`)
3. Use smaller model (if available)
4. Batch multiple videos together

### Server Port Already in Use

**Problem**: `Address already in use`

**Solution**: Change port in `server/app.py`:

```python
app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
```

## Project Structure

```
Refined4/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── main.py                      # Main entry point (train/eval/infer)
├── dataset.py                   # Dataset and data loading
├── model.py                     # MViT-based classifier
├── trainer.py                   # Training loop
├── evaluator.py                 # Evaluation metrics
├── smoother.py                  # Temporal smoothing
├── utils.py                     # Utility functions
├── encoders.pkl                 # Saved label encoders
├── model.pth                    # Trained model weights
├── predictions.json             # Inference output
│
├── data/                        # Training data
│   ├── videos_001/
│   │   ├── annotations.csv
│   │   └── chunks/
│   ├── videos_002/
│   └── ...
│
├── model_cache/                 # Downloaded pretrained models
│   └── hub/
│
└── server/                      # REST API server
    ├── README.md               # Server-specific documentation
    ├── app.py                  # Flask application
    ├── predictor.py            # Prediction logic
    ├── test_client.py          # Test client script
    ├── start_server.bat        # Windows startup script
    ├── start_server.sh         # Linux/macOS startup script
    ├── requirements.txt        # Server dependencies
    ├── uploads/                # Temporary upload directory
    └── chunks/                 # Temporary chunks directory
```

## Additional Resources

### Model Architecture

- **MViT**: [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227)
- **Torchvision**: [Video Models Documentation](https://pytorch.org/vision/stable/models.html#video-classification)

### Training Tips

1. **Data Quality**: Ensure accurate annotations
2. **Data Augmentation**: Helps with generalization
3. **Class Balance**: Use weighted loss for imbalanced data
4. **Learning Rate**: Start with 5e-5, adjust based on loss curves
5. **Early Stopping**: Prevents overfitting
6. **Temporal Smoothing**: Improves prediction stability

### Performance Benchmarks

**Training (on RTX 3090):**

- Batch size: 2
- Time per epoch: ~3 minutes (1000 samples)
- Total training time: ~1 hour (20 epochs)

**Inference:**

- Single video (15s, 5 chunks): ~2-3 seconds (GPU)
- Throughput: ~150-200 chunks/minute (GPU)

### Citation

If you use this project, please cite:

```bibtex
@software{video_action_classifier,
  title={Video Action and App Classification System},
  author={Your Team},
  year={2026},
  description={MViT-based system for screen recording classification}
}
```

## License

[Add your license information here]

## Contact

For questions or issues, please contact [your contact information]

---

**Last Updated**: February 25, 2026
