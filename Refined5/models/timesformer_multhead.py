import torch, torch.nn as nn
from timesformer_pytorch import TimeSformer

class TimeSformerMultiHead(nn.Module):
    def __init__(self, num_actions, num_apps):
        super().__init__()
        self.backbone = TimeSformer(img_size=112, frames=50, num_classes=0)
        dim = self.backbone.dim
        self.action_head = nn.Linear(dim, num_actions)
        self.app_head = nn.Linear(dim, num_apps)

    def forward(self, x):
        feat = self.backbone(x)
        return self.action_head(feat), self.app_head(feat)