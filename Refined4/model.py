# project/model.py

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from config import MODEL_VARIANT, PRETRAINED
import os

# Set local cache directory for models
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class VideoClassifier(nn.Module):
    def __init__(self, num_action_classes, num_app_classes):
        super(VideoClassifier, self).__init__()
        
        # Load VideoMAE model with pretrained weights
        if PRETRAINED:
            self.backbone = VideoMAEForVideoClassification.from_pretrained(
                MODEL_VARIANT, 
                cache_dir=MODEL_CACHE_DIR,
                ignore_mismatched_sizes=True,
                num_labels=512  # Use as feature extractor
            )
        else:
            from transformers import VideoMAEConfig
            config = VideoMAEConfig.from_pretrained(MODEL_VARIANT, cache_dir=MODEL_CACHE_DIR)
            self.backbone = VideoMAEForVideoClassification(config)
        
        # Get hidden size from model config
        hidden_size = self.backbone.config.hidden_size
        
        # Remove the original classification head
        self.backbone.classifier = nn.Identity()
        
        # Add dropout and more sophisticated heads
        self.dropout = nn.Dropout(0.3)
        
        # Action head with intermediate layer
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_action_classes)
        )
        
        # App head with intermediate layer
        self.app_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_app_classes)
        )

    def forward(self, x):
        # VideoMAE expects input of shape (batch, num_frames, num_channels, height, width)
        # x comes in as (batch, T, C, H, W) which is correct for VideoMAE
        outputs = self.backbone(x)
        
        # Extract features (pooled output)
        features = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # If the output is still class logits, we need to get hidden states
        if features.dim() == 2 and features.size(1) != self.backbone.config.hidden_size:
            # Get the last hidden state and pool it
            hidden_states = self.backbone.videomae(x).last_hidden_state
            features = hidden_states.mean(dim=1)  # Global average pooling
        
        features = self.dropout(features)
        action_logits = self.action_head(features)
        app_logits = self.app_head(features)
        return action_logits, app_logits