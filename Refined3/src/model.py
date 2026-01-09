# model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Set local cache directory for model weights
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_cache')
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = LOCAL_CACHE_DIR

class VideoClassifier(nn.Module):
    def __init__(self, num_actions, num_apps):
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT
        self.backbone = r2plus1d_18(weights=weights)
        
        # R2Plus1D has an 'fc' layer as the classifier
        if hasattr(self.backbone, 'fc') and isinstance(self.backbone.fc, nn.Linear):
            hidden_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # Fallback
            hidden_dim = 512
            if hasattr(self.backbone, 'fc'):
                self.backbone.fc = nn.Identity()
        
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.app_head = nn.Linear(hidden_dim, num_apps)

    def forward(self, x):
        # x: B, C, T, H, W (T=50)
        # R2Plus1D can handle variable number of frames efficiently
        features = self.backbone(x)
        
        # Handle case where backbone might return extra dimensions
        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))
        
        action_logits = self.action_head(features)
        app_logits = self.app_head(features)
        return action_logits, app_logits

# Get preprocessing parameters from weights
def get_preprocess_params():
    weights = R2Plus1D_18_Weights.DEFAULT
    transform = weights.transforms()
    # VideoClassification transform has mean and std as direct attributes
    if hasattr(transform, 'mean') and hasattr(transform, 'std'):
        mean = transform.mean
        std = transform.std
    else:
        # Default normalization values for video models trained on Kinetics-400
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    return mean, std