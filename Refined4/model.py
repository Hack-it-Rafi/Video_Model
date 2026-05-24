# project/model.py

# pyrefly: ignore [missing-import]
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

        # Auto-detect head structure: MViT-v2 uses head.proj, MViT-v1 uses head[1]
        if hasattr(self.backbone.head, 'proj'):
            # MViT-v2 (mvit_v2_s / mvit_v2_b): head is a MultiscaleClassificationHead with a .proj Linear
            in_features = self.backbone.head.proj.in_features
        elif isinstance(self.backbone.head, nn.Sequential):
            # MViT-v1 (mvit_v1_b): head is Sequential([AdaptiveAvgPool3d, Linear])
            in_features = self.backbone.head[1].in_features
        else:
            raise ValueError(f"Unknown head type for model '{MODEL_NAME}': {type(self.backbone.head)}")

        self.backbone.head = nn.Identity()  # Remove original head

        # Stronger regularization to match the deeper MViT-v2 backbone
        self.dropout = nn.Dropout(0.4)

        # Action head with intermediate layer
        self.action_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_action_classes)
        )

        # App head with intermediate layer
        self.app_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_app_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        action_logits = self.action_head(features)
        app_logits = self.app_head(features)
        return action_logits, app_logits