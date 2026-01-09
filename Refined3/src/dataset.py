# dataset.py
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.io as io
import torch.nn.functional as F
from typing import Dict

class VideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir: str, action_map: Dict[str, int], app_map: Dict[str, int], mean, std, max_samples: int = None):
        self.df = df
        # Limit the dataframe size if max_samples is specified
        if max_samples is not None and max_samples > 0:
            self.df = df.iloc[:max_samples].copy()
        self.video_dir = video_dir
        self.action_map = action_map
        self.app_map = app_map
        # Shape mean and std as (C, 1, 1, 1) for broadcasting over (C, T, H, W)
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        video_path = os.path.join(self.video_dir, filename)
        frames, _, _ = io.read_video(video_path, pts_unit='sec')
        # frames: T H W C, uint8
        frames = frames.float() / 255.0  # Normalize to [0, 1], T H W C
        frames = frames.permute(3, 0, 1, 2)  # C T H W
        
        # X3D can handle 50 frames - no need to downsample!
        # Keep all 50 frames for better temporal understanding
        
        # Resize spatial dimensions to 224x224
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize with ImageNet stats
        frames = (frames - self.mean) / self.std
        # frames is now C T H W (C, 50, 224, 224), which will be batched to B C T H W by DataLoader
        action = row['action']
        # Handle NaN values in target_app by converting to 'none'
        app = row.get('target_app', 'none')
        if pd.isna(app):
            app = 'none'
        label_action = self.action_map[action]
        label_app = self.app_map[app]
        chunk_id = row['chunk_id']
        return frames, label_action, label_app, chunk_id, filename