# UI Screen-Recording Video Understanding Pipeline
# Production-ready codebase
# ============================================

ui_video_pipeline/
├── configs/
│   └── config.yaml
├── data/
│   └── dataset.py
├── models/
│   └── timesformer_multhead.py
├── training/
│   ├── losses.py
│   ├── trainer.py
│   └── evaluate.py
├── inference/
│   ├── smoothing.py
│   └── predict.py
├── utils/
│   ├── metrics.py
│   ├── video.py
│   └── split.py
└── train.py


========================
configs/config.yaml
========================
project:
  num_frames: 50
  img_size: 112
  batch_size: 4
  epochs: 20
  lr: 3e-4
  num_workers: 4
  mixed_precision: true

labels:
  actions: [user:typing, scrolling, app:switch, idle, reading]
  apps: [vscode, browser, terminal, none]

smoothing:
  window: 2

========================
data/dataset.py
========================
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

========================
models/timesformer_multhead.py
========================
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

========================
training/losses.py
========================
import torch.nn as nn

class MultiHeadLoss(nn.Module):
    def __init__(self, w_action, w_app):
        super().__init__()
        self.la = nn.CrossEntropyLoss(weight=w_action)
        self.lp = nn.CrossEntropyLoss(weight=w_app)
    def forward(self, pa, pp, ta, tp):
        return self.la(pa, ta) + self.lp(pp, tp)

========================
training/trainer.py
========================
import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, opt, loss_fn, device, mp):
        self.model, self.opt, self.loss_fn = model, opt, loss_fn
        self.device, self.mp = device, mp
        self.scaler = GradScaler() if mp else None

    def train_epoch(self, loader):
        self.model.train()
        total=0
        for v,a,p in loader:
            v,a,p = v.to(self.device), a.to(self.device), p.to(self.device)
            self.opt.zero_grad()
            if self.mp:
                with autocast(): pa,pp = self.model(v)
                loss = self.loss_fn(pa,pp,a,p)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                pa,pp = self.model(v); loss = self.loss_fn(pa,pp,a,p)
                loss.backward(); self.opt.step()
            total += loss.item()
        return total/len(loader)

========================
training/evaluate.py
========================
import torch
from utils.metrics import compute_metrics

def evaluate(model, loader, device):
    model.eval(); A,P,TA,TP=[],[],[],[]
    with torch.no_grad():
        for v,a,p in loader:
            v=v.to(device); pa,pp = model(v)
            A.append(pa.cpu()); P.append(pp.cpu())
            TA.append(a); TP.append(p)
    return compute_metrics(torch.cat(A),torch.cat(P),torch.cat(TA),torch.cat(TP))

========================
utils/metrics.py
========================
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def head_metrics(pred, true):
    p = pred.argmax(1)
    acc = accuracy_score(true, p)
    pr,rc,f,_ = precision_recall_fscore_support(true,p,average='macro')
    return acc,pr,rc,f

def compute_metrics(pa,pp,ta,tp):
    return {
        "action": head_metrics(pa,ta),
        "app": head_metrics(pp,tp)
    }

========================
utils/video.py
========================
# shared video utilities (resize, normalize if needed)

========================
utils/split.py
========================
import random

def split_by_recording(folders, ratio=0.8):
    random.shuffle(folders)
    k=int(len(folders)*ratio)
    return folders[:k], folders[k:]

========================
inference/smoothing.py
========================
import numpy as np

def smooth_probs(probs, window):
    out=[]
    for i in range(len(probs)):
        s=max(0,i-window); e=min(len(probs),i+window+1)
        out.append(np.mean(probs[s:e],0))
    return out

========================
inference/predict.py
========================
import torch, json
from smoothing import smooth_probs

def predict(model, clips, maps, window):
    model.eval(); pa,pp=[],[]
    with torch.no_grad():
        for v in clips:
            a,p = model(v.unsqueeze(0))
            pa.append(torch.softmax(a,1).cpu().numpy()[0])
            pp.append(torch.softmax(p,1).cpu().numpy()[0])
    pa_s = smooth_probs(pa,window); pp_s = smooth_probs(pp,window)
    results=[]
    for i,(a,p) in enumerate(zip(pa_s,pp_s)):
        ai=a.argmax(); pi=p.argmax()
        results.append({
            "chunk_id": f"chunk_{i:03d}",
            "action": maps['action_inv'][ai],
            "action_conf": float(a[ai]),
            "app": maps['app_inv'][pi],
            "app_conf": float(p[pi])
        })
    return results

========================
train.py
========================
import torch, yaml, os
from data.dataset import ScreenVideoDataset
from models.timesformer_multhead import TimeSformerMultiHead
from training.trainer import Trainer
from training.losses import MultiHeadLoss
from training.evaluate import evaluate
from utils.split import split_by_recording
from torch.utils.data import DataLoader

cfg=yaml.safe_load(open("configs/config.yaml"))
root="data"
folders=os.listdir(root)
train_f,val_f = split_by_recording(folders)

act_map={n:i for i,n in enumerate(cfg['labels']['actions'])}
app_map={n:i for i,n in enumerate(cfg['labels']['apps'])}

train_ds=ScreenVideoDataset(root,train_f,act_map,app_map)
val_ds=ScreenVideoDataset(root,val_f,act_map,app_map)

train_dl=DataLoader(train_ds,batch_size=cfg['project']['batch_size'],shuffle=True)
val_dl=DataLoader(val_ds,batch_size=cfg['project']['batch_size'])

model=TimeSformerMultiHead(len(act_map),len(app_map)).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=cfg['project']['lr'])
w_action=torch.ones(len(act_map)).cuda()
w_app=torch.ones(len(app_map)).cuda()
loss_fn=MultiHeadLoss(w_action,w_app)
trainer=Trainer(model,opt,loss_fn,'cuda',cfg['project']['mixed_precision'])

for e in range(cfg['project']['epochs']):
    l=trainer.train_epoch(train_dl)
    m=evaluate(model,val_dl,'cuda')
    print(e,l,m)
