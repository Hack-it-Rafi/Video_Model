import os

os.environ['TORCH_HOME'] = 'F:/Attck/Label/POC/.cache/torch'
os.environ['HF_HOME'] = 'F:/Attck/Label/POC/.cache/huggingface'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import pandas as pd
import cv2


VIDEO_DIR = "./46"  
CSV_FILE = "annotations.csv"
BATCH_SIZE = 8
EPOCHS = 10
NUM_CLASSES = 379

class ScreenRecordingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.labels = self.annotations['full_label'].unique().tolist()
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        
        video_name = self.annotations.iloc[idx, 0] 
        video_path = os.path.join(self.root_dir, video_name)
        
        frames = self._load_video(video_path) 
        
        label_str = self.annotations.iloc[idx, 3] 
        label_id = self.label_to_id[label_str]
        
        return frames, label_id

    def _load_video(self, path):
        # Simplified video loader (resizes to 112x112, takes 16 frames)
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while len(frames) < 16:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to 0-1
                frames.append(torch.tensor(frame).permute(2, 0, 1) / 255.0)
        finally:
            cap.release()
        
        while len(frames) < 16:
            frames.append(frames[-1] if frames else torch.zeros(3, 112, 112))
            
        return torch.stack(frames).permute(1, 0, 2, 3).float()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ScreenRecordingDataset(CSV_FILE, VIDEO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    weights = R2Plus1D_18_Weights.DEFAULT
    model = r2plus1d_18(weights=weights)
    
    model.fc = nn.Linear(model.fc.in_features, len(dataset.labels))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Starting training on {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    torch.save(model.state_dict(), "screen_action_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train() 
    pass