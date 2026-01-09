# project/dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io')
import torchvision.io as io
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATA_ROOT

class VideoDataset(Dataset):
    def __init__(self, data_df, root_dir, transform=None, num_frames=16):
        self.data_df = data_df
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        video_path = os.path.join(self.root_dir, row['video_dir'], 'chunks', row['filename'])
        video, _, _ = io.read_video(video_path, pts_unit='sec')  # video: (T, H, W, C)
        
        # Sample exactly num_frames frames uniformly
        total_frames = video.shape[0]
        if total_frames >= self.num_frames:
            # Sample uniformly
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            video = video[indices]
        else:
            # Repeat frames if video is too short
            repeat_factor = (self.num_frames + total_frames - 1) // total_frames
            video = video.repeat(repeat_factor, 1, 1, 1)[:self.num_frames]
        
        # Convert to float and normalize to [0, 1]
        video = video.float() / 255.0
        
        # Apply transforms frame by frame
        if self.transform:
            frames = []
            for i in range(video.shape[0]):
                frame = video[i]  # (H, W, C)
                frame = frame.permute(2, 0, 1)  # (C, H, W)
                frame = self.transform(frame)
                frames.append(frame)
            video = torch.stack(frames)  # (T, C, H, W)
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        else:
            video = video.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        action_label = row['action_label']
        app_label = row['app_label']
        
        return video, action_label, app_label, row['chunk_id'], row['video_id']

def get_transforms(train=True):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

def load_dataset():
    all_data = []
    video_ids = []
    for video_dir in os.listdir(DATA_ROOT):
        if video_dir.startswith('videos_'):
            annot_path = os.path.join(DATA_ROOT, video_dir, 'annotations.csv')
            if os.path.exists(annot_path):
                df = pd.read_csv(annot_path)
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()
                df['target_app'] = df['target_app'].fillna('none')
                df['video_dir'] = video_dir
                df['video_id'] = video_dir
                # Add chunk_id from filename
                df['chunk_id'] = df['filename'].str.replace('.mp4', '')
                all_data.append(df)
                video_ids.append(video_dir)
    
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Encode labels
    action_encoder = LabelEncoder()
    app_encoder = LabelEncoder()
    
    full_df['action_label'] = action_encoder.fit_transform(full_df['action'])
    full_df['app_label'] = app_encoder.fit_transform(full_df['target_app'])
    
    num_action_classes = len(action_encoder.classes_)
    num_app_classes = len(app_encoder.classes_)
    
    # Split by video_id to avoid leakage
    train_vids, test_vids = train_test_split(video_ids, test_size=0.2, random_state=42)
    
    train_df = full_df[full_df['video_id'].isin(train_vids)]
    test_df = full_df[full_df['video_id'].isin(test_vids)]
    
    train_dataset = VideoDataset(train_df, DATA_ROOT, transform=get_transforms(train=True))
    test_dataset = VideoDataset(test_df, DATA_ROOT, transform=get_transforms(train=False))
    
    return train_dataset, test_dataset, action_encoder, app_encoder, train_df, test_df, num_action_classes, num_app_classes

def get_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def compute_class_weights(df, num_classes, label_col):
    # Initialize weights for all classes
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    
    # Get class counts for classes that appear in the data
    class_counts = df[label_col].value_counts()
    
    # Compute weights only for classes that appear in the data
    for class_id, count in class_counts.items():
        class_weights[class_id] = len(df) / (num_classes * count)
    
    return class_weights