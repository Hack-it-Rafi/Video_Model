# project/model.py

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from config import MODEL_NAME, PRETRAINED
import os

# Set local cache directory for models
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
torch.hub.set_dir(MODEL_CACHE_DIR)

class VideoClassifier(nn.Module):
    def __init__(self, num_action_classes, num_app_classes):
        super(VideoClassifier, self).__init__()
        self.backbone = getattr(video_models, MODEL_NAME)(pretrained=PRETRAINED)
        in_features = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()  # Remove original head
        
        self.action_head = nn.Linear(in_features, num_action_classes)
        self.app_head = nn.Linear(in_features, num_app_classes)

    def forward(self, x):
        features = self.backbone(x)
        action_logits = self.action_head(features)
        app_logits = self.app_head(features)
        return action_logits, app_logits