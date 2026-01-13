import os, csv, torch, cv2
from torch.utils.data import Dataset
import numpy as np

class ScreenVideoDataset(Dataset):
    def __init__(self, root, split_files, action_map, app_map):
        self.samples = []
        self.action_map = action_map
        self.app_map = app_map
        for folder in split_files:
            ann = os.path.join(root, folder, "annotations.csv")
            chunk_dir = os.path.join(root, folder, "chunks")
            with open(ann) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    fname = r["filename"]
                    action = r["action"]
                    app = r["target_app"] or "none"
                    self.samples.append((os.path.join(chunk_dir, fname), action, app))

    def __len__(self): return len(self.samples)

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (112,112))
            frames.append(frame[:,:,::-1])
        cap.release()
        arr = np.stack(frames).astype("float32")/255.0
        return torch.from_numpy(arr).permute(0,3,1,2)  # T,C,H,W

    def __getitem__(self, idx):
        path, act, app = self.samples[idx]
        video = self.read_video(path)
        return video, self.action_map[act], self.app_map[app]