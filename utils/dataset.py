import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from decord import VideoReader, cpu
import numpy as np

class ScreenActionDataset(Dataset):
    def __init__(self, csv_file, chunks_dir, label2id, num_frames=16):
        self.df = pd.read_csv(csv_file)
        self.chunks_dir = chunks_dir
        self.label2id = label2id
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.chunks_dir, row["video_file"])
        
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).numpy()  # (T, H, W, C)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

        label = self.label2id[row["full_label"]]
        return {"pixel_values": frames, "labels": torch.tensor(label)}