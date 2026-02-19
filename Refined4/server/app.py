import os
import sys
import uuid
import shutil
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE
from model import VideoClassifier
from utils import load_checkpoint
from server.predictor import VideoPredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['CHUNKS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'chunks')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHUNKS_FOLDER'], exist_ok=True)

# Initialize predictor (loads model and encoders)
predictor = VideoPredictor()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'mpeg', 'mpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_mp4(input_path, output_path):
    """Convert any video format to mp4 using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',  # Overwrite output file
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e.stderr}")
        return False

def split_video_into_chunks(video_path, output_dir, chunk_duration=3):
    """Split video into 3-second chunks using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # First, get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        
        # Split video into chunks
        cmd = [
            'ffmpeg', '-i', video_path,
            '-c', 'copy',
            '-map', '0',
            '-segment_time', str(chunk_duration),
            '-f', 'segment',
            '-reset_timestamps', '1',
            os.path.join(output_dir, 'chunk_%03d.mp4')
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count generated chunks
        chunks = sorted([f for f in os.listdir(output_dir) if f.startswith('chunk_') and f.endswith('.mp4')])
        return chunks, duration
    except subprocess.CalledProcessError as e:
        print(f"Error splitting video: {e.stderr}")
        return [], 0
    except Exception as e:
        print(f"Error: {e}")
        return [], 0

def cleanup_temp_files(upload_path, chunks_dir):
    """Clean up temporary files"""
    try:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)
    except Exception as e:
        print(f"Error cleaning up files: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'device': str(DEVICE)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts a video file and returns action/app predictions for each 3-second chunk
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{request_id}_{filename}")
    file.save(upload_path)
    
    # Create mp4 version if needed
    file_ext = filename.rsplit('.', 1)[1].lower()
    if file_ext != 'mp4':
        mp4_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{request_id}.mp4")
        print(f"Converting {file_ext} to mp4...")
        if not convert_to_mp4(upload_path, mp4_path):
            cleanup_temp_files(upload_path, None)
            return jsonify({'error': 'Failed to convert video to mp4'}), 500
        os.remove(upload_path)  # Remove original
        upload_path = mp4_path
    
    # Create directory for chunks
    chunks_dir = os.path.join(app.config['CHUNKS_FOLDER'], request_id)
    
    try:
        # Split video into 3-second chunks
        print(f"Splitting video into 3-second chunks...")
        chunks, duration = split_video_into_chunks(upload_path, chunks_dir)
        
        if not chunks:
            return jsonify({'error': 'Failed to split video into chunks'}), 500
        
        print(f"Generated {len(chunks)} chunks from {duration:.2f}s video")
        
        # Run predictions on all chunks with temporal smoothing
        print(f"Running predictions with model and temporal smoothing...")
        predictions = predictor.predict_video(chunks_dir, request_id)
        
        # Prepare response
        response = {
            'request_id': request_id,
            'video_info': {
                'original_filename': filename,
                'duration_seconds': round(duration, 2),
                'num_chunks': len(chunks),
                'chunk_duration': 3
            },
            'predictions': predictions,
            'summary': generate_summary(predictions)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    finally:
        # Clean up temporary files
        cleanup_temp_files(upload_path, chunks_dir)

def generate_summary(predictions):
    """Generate a summary of predictions"""
    if not predictions:
        return {}
    
    # Count action occurrences
    action_counts = {}
    app_counts = {}
    
    for pred in predictions:
        action = pred['action']
        app = pred['app']
        
        action_counts[action] = action_counts.get(action, 0) + 1
        app_counts[app] = app_counts.get(app, 0) + 1
    
    # Sort by frequency
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_apps = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'total_chunks': len(predictions),
        'most_common_action': sorted_actions[0][0] if sorted_actions else None,
        'most_common_app': sorted_apps[0][0] if sorted_apps else None,
        'action_distribution': dict(sorted_actions),
        'app_distribution': dict(sorted_apps),
        'average_action_confidence': round(sum(p['action_confidence'] for p in predictions) / len(predictions), 3),
        'average_app_confidence': round(sum(p['app_confidence'] for p in predictions) / len(predictions), 3)
    }

@app.route('/info', methods=['GET'])
def info():
    """Return model information"""
    return jsonify({
        'model_info': {
            'num_action_classes': predictor.num_action_classes,
            'num_app_classes': predictor.num_app_classes,
            'action_classes': predictor.action_classes,
            'app_classes': predictor.app_classes,
            'device': str(DEVICE)
        }
    })

if __name__ == '__main__':
    print(f"Starting server on http://0.0.0.0:5000")
    print(f"Using device: {DEVICE}")
    print(f"Model loaded: {predictor.model is not None}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
