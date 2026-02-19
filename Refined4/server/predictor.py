import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, NUM_FRAMES
from model import VideoClassifier
from utils import load_encoders
from dataset import get_transforms
import torchvision.io as io

class VideoPredictor:
    def __init__(self, model_path='model.pth', encoders_path='encoders.pkl'):
        """Initialize the predictor with trained model and encoders"""
        self.device = DEVICE
        self.num_frames = NUM_FRAMES
        self.transform = get_transforms(train=False)
        self.window_size = 3  # For temporal smoothing
        
        # Get full paths
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_full_path = os.path.join(base_dir, model_path)
        encoders_full_path = os.path.join(base_dir, encoders_path)
        
        # Check if files exist
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model not found: {model_full_path}. Please train the model first: python main.py train")
        
        if not os.path.exists(encoders_full_path):
            raise FileNotFoundError(f"Encoders not found: {encoders_full_path}. Please train the model first: python main.py train")
        
        # Load checkpoint first to get the actual model architecture
        print(f"Loading model checkpoint from {model_full_path}...")
        checkpoint = torch.load(model_full_path, map_location='cpu')
        
        # Infer number of classes from checkpoint weights
        # The app_head.3.weight has shape [num_app_classes, 256]
        # The action_head.3.weight has shape [num_action_classes, 512]
        model_num_action_classes = checkpoint['action_head.3.weight'].shape[0]
        model_num_app_classes = checkpoint['app_head.3.weight'].shape[0]
        
        print(f"Model architecture: {model_num_action_classes} action classes, {model_num_app_classes} app classes")
        
        # Load encoders (these were saved from training data)
        print(f"Loading encoders from {encoders_full_path}...")
        action_encoder, app_encoder = load_encoders(encoders_full_path)
        
        encoder_num_action_classes = len(action_encoder.classes_)
        encoder_num_app_classes = len(app_encoder.classes_)
        
        print(f"Encoders have: {encoder_num_action_classes} action classes, {encoder_num_app_classes} app classes")
        
        # Adjust encoders to match model if needed
        if encoder_num_action_classes != model_num_action_classes:
            print(f"Adjusting action encoder from {encoder_num_action_classes} to {model_num_action_classes} classes...")
            action_encoder = self._adjust_encoder(action_encoder, model_num_action_classes, 'action')
        
        if encoder_num_app_classes != model_num_app_classes:
            print(f"Adjusting app encoder from {encoder_num_app_classes} to {model_num_app_classes} classes...")
            app_encoder = self._adjust_encoder(app_encoder, model_num_app_classes, 'app')
        
        self.action_encoder = action_encoder
        self.app_encoder = app_encoder
        self.num_action_classes = model_num_action_classes
        self.num_app_classes = model_num_app_classes
        self.action_classes = self.action_encoder.classes_.tolist()
        self.app_classes = self.app_encoder.classes_.tolist()
        
        print(f"Final: {self.num_action_classes} action classes and {self.num_app_classes} app classes")
        
        # Initialize and load model with correct architecture
        self.model = VideoClassifier(self.num_action_classes, self.num_app_classes).to(self.device)
        self.model.load_state_dict(checkpoint)
        print(f"✓ Model loaded successfully")
        
        self.model.eval()
    
    def _adjust_encoder(self, encoder, target_num_classes, label_type):
        """Adjust encoder to match the model's expected number of classes"""
        current_num_classes = len(encoder.classes_)
        
        if target_num_classes > current_num_classes:
            # Model expects more classes, add dummy classes
            num_missing = target_num_classes - current_num_classes
            dummy_classes = [f"unknown_{label_type}_{i}" for i in range(num_missing)]
            encoder.classes_ = np.concatenate([encoder.classes_, dummy_classes])
            print(f"  Added {num_missing} dummy classes: {dummy_classes}")
        elif target_num_classes < current_num_classes:
            # Model expects fewer classes, truncate
            encoder.classes_ = encoder.classes_[:target_num_classes]
            print(f"  Truncated to first {target_num_classes} classes")
        
        return encoder
    
    def load_video_chunk(self, video_path):
        """Load and preprocess a single video chunk"""
        try:
            video, _, _ = io.read_video(video_path, pts_unit='sec')
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None
        
        # Sample frames
        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            video = video[indices]
        else:
            repeat_factor = (self.num_frames + total_frames - 1) // total_frames
            video = video.repeat(repeat_factor, 1, 1, 1)[:self.num_frames]
        
        # Normalize
        video = video.float() / 255.0
        
        # Apply transforms
        frames = []
        for i in range(video.shape[0]):
            frame = video[i].permute(2, 0, 1)  # (C, H, W)
            frame = self.transform(frame)
            frames.append(frame)
        
        video = torch.stack(frames)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return video.unsqueeze(0)  # Add batch dimension
    
    def temporal_smoothing(self, action_probs_list, app_probs_list):
        """Apply temporal smoothing to predictions"""
        if len(action_probs_list) == 1:
            return action_probs_list, app_probs_list
        
        pad = (self.window_size - 1) // 2
        
        # Stack tensors
        action_probs_stacked = torch.stack(action_probs_list)
        app_probs_stacked = torch.stack(app_probs_list)
        
        # Pad temporal dimension
        action_probs_padded = F.pad(action_probs_stacked.unsqueeze(0), (0, 0, pad, pad), mode='replicate').squeeze(0)
        app_probs_padded = F.pad(app_probs_stacked.unsqueeze(0), (0, 0, pad, pad), mode='replicate').squeeze(0)
        
        smoothed_action_probs = []
        smoothed_app_probs = []
        
        for i in range(len(action_probs_list)):
            window_action = action_probs_padded[i:i+self.window_size].mean(dim=0)
            window_app = app_probs_padded[i:i+self.window_size].mean(dim=0)
            smoothed_action_probs.append(window_action)
            smoothed_app_probs.append(window_app)
        
        return smoothed_action_probs, smoothed_app_probs
    
    def predict_video(self, chunks_dir, video_id):
        """
        Predict action and app labels for all chunks in a video with temporal smoothing
        
        Args:
            chunks_dir: Directory containing video chunks
            video_id: Identifier for the video
            
        Returns:
            List of predictions with action, app, and confidence scores
        """
        # Get all chunk files
        chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith('.mp4')])
        
        if not chunk_files:
            return []
        
        # Store predictions
        action_probs_list = []
        app_probs_list = []
        chunk_names = []
        
        # Process each chunk
        with torch.no_grad():
            for chunk_file in chunk_files:
                chunk_path = os.path.join(chunks_dir, chunk_file)
                video_tensor = self.load_video_chunk(chunk_path)
                
                if video_tensor is None:
                    continue
                
                video_tensor = video_tensor.to(self.device)
                
                # Get predictions
                action_logits, app_logits = self.model(video_tensor)
                action_probs = F.softmax(action_logits, dim=1).cpu().squeeze(0)
                app_probs = F.softmax(app_logits, dim=1).cpu().squeeze(0)
                
                action_probs_list.append(action_probs)
                app_probs_list.append(app_probs)
                chunk_names.append(chunk_file)
        
        # Apply temporal smoothing
        if len(action_probs_list) > 1:
            smoothed_action_probs, smoothed_app_probs = self.temporal_smoothing(action_probs_list, app_probs_list)
        else:
            smoothed_action_probs = action_probs_list
            smoothed_app_probs = app_probs_list
        
        # Format predictions
        predictions = []
        for i, chunk_name in enumerate(chunk_names):
            action_idx = torch.argmax(smoothed_action_probs[i]).item()
            app_idx = torch.argmax(smoothed_app_probs[i]).item()
            action_conf = torch.max(smoothed_action_probs[i]).item()
            app_conf = torch.max(smoothed_app_probs[i]).item()
            
            # Calculate time range for this chunk
            chunk_num = int(chunk_name.split('_')[1].split('.')[0])
            start_time = chunk_num * 3
            end_time = start_time + 3
            
            prediction = {
                'chunk_id': chunk_name.replace('.mp4', ''),
                'chunk_number': chunk_num,
                'time_range': f"{start_time}s-{end_time}s",
                'action': self.action_encoder.inverse_transform([action_idx])[0],
                'action_confidence': round(action_conf, 3),
                'app': self.app_encoder.inverse_transform([app_idx])[0],
                'app_confidence': round(app_conf, 3)
            }
            predictions.append(prediction)
        
        return predictions
