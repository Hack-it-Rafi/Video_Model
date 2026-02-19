# Video Action & App Classification Server

A Flask-based REST API server that performs video classification to predict actions and target applications using a trained MViT model with temporal smoothing.

## Features

- **Video Upload Support**: Accepts multiple video formats (mp4, avi, mov, mkv, flv, wmv, webm, mpeg, mpg)
- **Automatic Format Conversion**: Converts any video format to mp4 using ffmpeg
- **Video Chunking**: Automatically splits videos into 3-second chunks
- **AI-Powered Predictions**: Uses a trained MViT (Multiscale Vision Transformer) model
- **Temporal Smoothing**: Applies temporal smoothing across chunks for more stable predictions
- **Detailed Results**: Returns predictions with confidence scores for each chunk
- **Summary Statistics**: Provides overall statistics and distribution of actions/apps

## Prerequisites

### System Requirements

- Python 3.8+
- ffmpeg and ffprobe (must be installed and available in PATH)
- CUDA-capable GPU (optional, but recommended for faster inference)

### Installing ffmpeg

**Windows:**

```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**Linux:**

```bash
sudo apt-get install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

## Installation

1. **Install Python dependencies:**

```bash
cd server
pip install -r requirements.txt
```

2. **Ensure model is trained:**
   Make sure `model.pth` exists in the `Refined4` directory. If not, train the model first:

```bash
cd ..
python main.py train
```

## Usage

### Starting the Server

```bash
# From the server directory
python app.py

# Or from the Refined4 directory
python server/app.py
```

The server will start on `http://0.0.0.0:5000`

### API Endpoints

#### 1. Health Check

```bash
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### 2. Model Information

```bash
GET /info
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

#### 3. Video Prediction

```bash
POST /predict
Content-Type: multipart/form-data
Body: video file (key: "video")
```

Example using curl:

```bash
curl -X POST -F "video=@my_video.mp4" http://localhost:5000/predict
```

Example using Python:

```python
import requests

with open('my_video.mp4', 'rb') as f:
    files = {'video': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    data = response.json()
    print(data)
```

Response:

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

### Using the Test Client

A test client script is provided for easy testing:

```bash
# From the server directory
python test_client.py path/to/your/video.mp4

# Example
python test_client.py ../data/videos_001/chunks/chunk_001.mp4
```

The test client will:

1. Check server health
2. Display model information
3. Upload and predict the video
4. Display formatted results
5. Save full results to a JSON file

## How It Works

1. **Video Upload**: Client uploads a video file to the `/predict` endpoint
2. **Format Conversion**: If not mp4, the video is converted using ffmpeg
3. **Chunking**: Video is split into 3-second chunks using ffmpeg segment
4. **Preprocessing**: Each chunk is:
   - Loaded and decoded
   - Sampled to 16 frames
   - Resized to 224x224
   - Normalized
5. **Inference**: The MViT model predicts action and app for each chunk
6. **Temporal Smoothing**: Predictions are smoothed across a 3-chunk window
7. **Response**: Results are formatted and returned as JSON
8. **Cleanup**: Temporary files are automatically cleaned up

## Configuration

You can modify server settings in `app.py`:

- `MAX_CONTENT_LENGTH`: Maximum upload file size (default: 500MB)
- `UPLOAD_FOLDER`: Directory for uploaded videos
- `CHUNKS_FOLDER`: Directory for video chunks

Model configuration is in `../config.py`:

- `NUM_FRAMES`: Number of frames to sample (default: 16)
- `DEVICE`: 'cuda' or 'cpu'
- `WINDOW_SIZE`: Temporal smoothing window size (default: 3)

## Troubleshooting

### ffmpeg not found

Make sure ffmpeg and ffprobe are installed and available in your PATH:

```bash
ffmpeg -version
ffprobe -version
```

### Model not found

Ensure `model.pth` exists in the parent directory (Refined4):

```bash
ls ../model.pth
```

### CUDA out of memory

If you encounter GPU memory issues, the server will automatically fall back to CPU. You can also manually set `DEVICE='cpu'` in `config.py`.

### Port already in use

Change the port in `app.py`:

```python
app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
```

## Performance Tips

1. **Use GPU**: Inference is much faster on GPU (CUDA)
2. **Batch Processing**: The server processes chunks sequentially; for multiple videos, consider parallel requests
3. **Video Size**: Larger videos take longer to chunk and process
4. **Network**: For remote deployment, consider upload bandwidth limitations

## API Integration Examples

### Python

```python
import requests

url = "http://localhost:5000/predict"
files = {'video': open('video.mp4', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### JavaScript (Node.js)

```javascript
const FormData = require("form-data");
const fs = require("fs");
const axios = require("axios");

const form = new FormData();
form.append("video", fs.createReadStream("video.mp4"));

axios
  .post("http://localhost:5000/predict", form, {
    headers: form.getHeaders(),
  })
  .then((response) => {
    console.log(response.data);
  });
```

### cURL

```bash
curl -X POST \
  -F "video=@video.mp4" \
  http://localhost:5000/predict \
  -o results.json
```

## License

This server is part of the Refined4 video classification project.
