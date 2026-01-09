# Project: Screen-recording workflow POC — multi-file implementation (updated)

This canvas now contains a multi-file POC for training a sequence model that uses **whole chunk videos** (≈50 frames per chunk) as input, stores per-chunk features, trains a transformer-based sequence classifier, and performs inference to produce a human-readable workflow.

Changes since the previous version you asked for:
- **Do not extract single frames** — each chunk video is processed as a short clip (T ≈ 50 frames).
- **Separated into multiple Python files** for clarity and modularity.
- **Ollama** integration instead of llama-cpp (`ollama` local REST API). The Ollama client wrapper tries `/api/generate` and falls back to `/v1/chat/completions` if available.
- Annotation format supported (example you gave):
  `chunk_000.mp4 | app:switch | notepad`
  `chunk_020.mp4 | user:typing`
  `chunk_126.mp4 | browser:switch_tab | video_streaming`

---

## Files included below (copy each into its own .py file):

1) `feature_extractor.py`  — loads a short video chunk, resamples/pads to target frame count, runs a 3D CNN (R3D) to get a single feature vector per chunk.

2) `preprocess.py` — walks the dataset folders, reads per-recording `annotations.csv`, extracts features for each chunk video (using the whole chunk), and saves `features.npy` and `labels.npy` per recording.

3) `dataset.py` — PyTorch Dataset class that loads per-recording features/labels and returns padded/truncated sequences for training.

4) `train.py` — training loop for the Transformer sequence model; saves `model_final.pth` into processed folder.

5) `infer.py` — load features for a recording (or extract them if missing), run inference, collapse consecutive labels into steps and print workflow.

6) `ollama_client.py` — small wrapper around Ollama local API (`http://localhost:11434`) to request a textual summary of the workflow.

---

### 1) feature_extractor.py
```python
# feature_extractor.py
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
from pathlib import Path
import numpy as np
from typing import Optional

# Use torchvision video model r3d_18 (3D ResNet-18) — available in torchvision
# Make sure torchvision version includes video models (>=0.8+ usually)

class ChunkFeatureExtractor:
    def __init__(self, device: str = 'cuda', target_frames: int = 50, image_size: int = 112):
        self.device = device
        self.target_frames = target_frames
        self.image_size = image_size
        # Load a pretrained 3D ResNet
        self.model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        # Replace the final fc with identity to get feature vector
        if hasattr(self.model, 'fc'):
            in_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.feature_dim = in_dim
        else:
            # fallback
            self.feature_dim = 512
        self.model = self.model.to(self.device).eval()

        # transforms applied per-frame (we will apply per-frame resize & normalize)
        self.frame_transform = T.Compose([
            T.ConvertImageDtype(torch.float),
            T.Resize((self.image_size, self.image_size)),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

    def _read_video_to_tensor(self, path: Path) -> torch.Tensor:
        # torchvision.io.read_video returns (frames, audio, info), frames as T x H x W x C in uint8
        from torchvision.io import read_video
        video_frames, _, info = read_video(str(path), pts_unit='sec')
        # video_frames: (T, H, W, C) uint8 -> convert to float tensor C x T x H x W
        if video_frames.numel() == 0:
            raise RuntimeError(f"No frames read from {path}")
        # permute -> T x C x H x W
        video_frames = video_frames.permute(0, 3, 1, 2)  # T, C, H, W
        # convert to float in 0..1
        video_frames = video_frames / 255.0
        return video_frames

    def _temporal_resample(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: T x C x H x W
        T_in = frames.shape[0]
        if T_in == self.target_frames:
            return frames
        # sample indices evenly
        if T_in > 1:
            idx = np.linspace(0, T_in - 1, self.target_frames).astype(int)
            return frames[idx]
        else:
            # if only 1 frame, repeat
            return frames.repeat(self.target_frames, 1, 1, 1)

    def extract(self, chunk_path: Path) -> np.ndarray:
        """Read the whole chunk video, resample/pad to target_frames, run through 3D CNN and return 1D feature vector."""
        frames = self._read_video_to_tensor(chunk_path)  # T x C x H x W (float in 0..1)
        frames = self._temporal_resample(frames)
        # apply per-frame resize+normalize using a loop (frames small)
        proc_frames = []
        for f in frames:
            # f: C x H x W float in 0..1; need to make PIL? but transforms expect tensor
            # our transform uses ConvertImageDtype and Resize which accept tensors in newer torchvision
            proc = self.frame_transform(f)
            proc_frames.append(proc)
        # stack -> T x C x H x W, convert to C x T x H x W for video models
        x = torch.stack(proc_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # 1 x C x T x H x W
        with torch.no_grad():
            feat = self.model(x)  # shape 1 x feature_dim (because fc->Identity)
        feat = feat.cpu().numpy().reshape(-1)
        return feat


if __name__ == '__main__':
    # quick test (requires a sample chunk video path)
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('chunk_000.mp4')
    ext = ChunkFeatureExtractor(device='cpu', target_frames=50)
    v = ext.extract(p)
    print('feature dim', v.shape)
```

### 2) preprocess.py
```python
# preprocess.py
from pathlib import Path
import numpy as np
import pandas as pd
from feature_extractor import ChunkFeatureExtractor
import argparse

# expected dataset layout:
# dataset_root/
#   recording_001/
#     chunks/
#       chunk_000.mp4
#       chunk_001.mp4
#     annotations.csv   # lines: chunk_000.mp4 | app:switch | notepad


def parse_annotation_line(line: str):
    # permissive splitter by '|' or ','
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 2:
        parts = [p.strip() for p in line.split(',')]
    # chunk_filename, label, optional extra
    fname = parts[0]
    label = parts[1]
    meta = parts[2] if len(parts) > 2 else ''
    return fname, label, meta


def preprocess_dataset(dataset_root: Path, processed_root: Path, device='cuda'):
    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    extractor = ChunkFeatureExtractor(device=device, target_frames=50, image_size=112)

    for rec in sorted(dataset_root.iterdir()):
        if not rec.is_dir():
            continue
        ann_path = rec / 'annotations.csv'
        if not ann_path.exists():
            print(f"Skip {rec.name}: annotations.csv missing")
            continue
        df_lines = [l for l in open(ann_path, 'r', encoding='utf-8').read().strip().splitlines() if l.strip()]
        chunk_files = []
        labels = []
        for ln in df_lines:
            fname, lbl, meta = parse_annotation_line(ln)
            # try to locate file under rec/ or rec/chunks/
            fpath = rec / fname
            if not fpath.exists():
                fpath = rec / 'chunks' / fname
                if not fpath.exists():
                    print(f"Warning: chunk file not found for {fname} in {rec}")
                    continue
            chunk_files.append(fpath)
            labels.append(lbl)

        feats = []
        for p in chunk_files:
            try:
                fv = extractor.extract(p)
                feats.append(fv)
            except Exception as e:
                print(f"Failed extract {p}: {e}")
        if len(feats) == 0:
            print(f"No features for {rec.name}, skipping save")
            continue
        feats = np.stack(feats, axis=0)
        labels = np.array(labels, dtype=object)
        out = processed_root / rec.name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'features.npy', feats)
        np.save(out / 'labels.npy', labels)
        print(f"Saved {out} features {feats.shape} labels {labels.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--processed_root', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    preprocess_dataset(Path(args.dataset_root), Path(args.processed_root), device=args.device)
```

### 3) dataset.py
```python
# dataset.py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch

class RecordingSequenceDataset(Dataset):
    def __init__(self, processed_root: Path, max_seq_len: int = 300, label_to_id: dict = None):
        self.processed_root = Path(processed_root)
        self.items = sorted([p for p in self.processed_root.iterdir() if p.is_dir()])
        self.max_seq_len = max_seq_len
        self.label_to_id = label_to_id or self._build_label_map()

    def _build_label_map(self):
        labels = set()
        for p in self.items:
            arr = np.load(p / 'labels.npy', allow_pickle=True)
            for l in arr.tolist():
                labels.add(l)
        labels = sorted(list(labels))
        return {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        feats = np.load(p / 'features.npy')  # (L, D)
        labels_raw = np.load(p / 'labels.npy', allow_pickle=True)  # (L,)
        labels = np.array([self.label_to_id[l] for l in labels_raw], dtype=np.int64)
        L = feats.shape[0]
        if L > self.max_seq_len:
            feats = feats[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.bool_)
        else:
            pad = self.max_seq_len - L
            feats = np.pad(feats, ((0, pad), (0, 0)), mode='constant')
            labels = np.pad(labels, ((0, pad),), mode='constant', constant_values=-100)
            mask = np.concatenate([np.ones(L, dtype=np.bool_), np.zeros(pad, dtype=np.bool_)])
        return torch.from_numpy(feats).float(), torch.from_numpy(labels).long(), torch.from_numpy(mask).bool()
```

### 4) train.py
```python
# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RecordingSequenceDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
from tqdm import tqdm

# Simple transformer sequence classifier (same design as before)
class SequenceModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_labels, max_len=300):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x, mask=None):
        x = self.input_proj(x) + self.pos_emb[:, : x.size(1), :]
        x = x.transpose(0, 1)
        src_key_padding_mask = ~mask if mask is not None else None
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0, 1)
        logits = self.classifier(out)
        return logits


def train(processed_root: Path, epochs=8, batch_size=4, device='cuda'):
    ds = RecordingSequenceDataset(processed_root)
    labels_map = ds.label_to_id
    num_labels = len(labels_map)
    idxs = list(range(len(ds)))
    tr_idx, val_idx = train_test_split(idxs, test_size=0.15, random_state=42)
    tr_ds = torch.utils.data.Subset(ds, tr_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: tuple(zip(*b)))

    sample_feat, _, _ = ds[0]
    input_dim = sample_feat.shape[1]
    model = SequenceModel(input_dim=input_dim, d_model=512, num_heads=8, num_layers=3, num_labels=num_labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, epochs+1):
        model.train()
        tot = 0.0
        for batch in tqdm(tr_loader, desc=f"Train E{epoch}"):
            feats_batch, labels_batch, masks_batch = batch
            feats = torch.stack(feats_batch).to(device)
            labels = torch.stack(labels_batch).to(device)
            masks = torch.stack(masks_batch).to(device)
            logits = model(feats, mask=masks)
            B, L, C = logits.shape
            loss = criterion(logits.view(B*L, C), labels.view(B*L))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {epoch} train loss {tot/len(tr_loader):.4f}")
        # validation
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val E{epoch}"):
                feats_batch, labels_batch, masks_batch = batch
                feats = torch.stack(feats_batch).to(device)
                labels = torch.stack(labels_batch).to(device)
                masks = torch.stack(masks_batch).to(device)
                logits = model(feats, mask=masks)
                B, L, C = logits.shape
                loss = criterion(logits.view(B*L, C), labels.view(B*L))
                vtot += loss.item()
        print(f"Epoch {epoch} val loss {vtot/len(val_loader):.4f}")
        torch.save({'model': model.state_dict(), 'labels_map': labels_map}, processed_root / f'model_epoch_{epoch}.pth')
    torch.save({'model': model.state_dict(), 'labels_map': labels_map}, processed_root / 'model_final.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_root', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train(Path(args.processed_root), epochs=args.epochs, batch_size=args.batch_size, device=args.device)
```

### 5) infer.py
```python
# infer.py
from pathlib import Path
import numpy as np
import torch
from train import SequenceModel
from feature_extractor import ChunkFeatureExtractor
import argparse
import requests
from ollama_client import ollama_summarize


def load_model(processed_root: Path, device='cuda'):
    ck = processed_root / 'model_final.pth'
    if not ck.exists():
        pths = sorted(processed_root.glob('model_epoch_*.pth'))
        if not pths:
            raise FileNotFoundError('no model checkpoint')
        ck = pths[-1]
    ckdata = torch.load(ck, map_location='cpu')
    labels_map = ckdata.get('labels_map')
    id_to_label = {v:k for k,v in labels_map.items()}
    sample_feat = np.load(sorted(list(processed_root.iterdir()))[0] / 'features.npy')
    input_dim = sample_feat.shape[1]
    model = SequenceModel(input_dim=input_dim, d_model=512, num_heads=8, num_layers=3, num_labels=len(labels_map))
    model.load_state_dict(ckdata['model'])
    model = model.eval().to(device)
    return model, id_to_label


def infer_on_recording(recording_dir: Path, processed_root: Path, device='cuda'):
    out_dir = processed_root / recording_dir.name
    if not out_dir.exists() or not (out_dir / 'features.npy').exists():
        # extract features for this recording only
        from preprocess import parse_annotation_line
        extractor = ChunkFeatureExtractor(device=device, target_frames=50, image_size=112)
        ann = [l.strip() for l in open(recording_dir / 'annotations.csv','r',encoding='utf-8').read().splitlines() if l.strip()]
        feats = []
        labels = []
        for ln in ann:
            fname, lbl, meta = parse_annotation_line(ln)
            fpath = recording_dir / fname
            if not fpath.exists():
                fpath = recording_dir / 'chunks' / fname
            if not fpath.exists():
                continue
            v = extractor.extract(fpath)
            feats.append(v)
            labels.append(lbl)
        if len(feats) == 0:
            raise RuntimeError('no features extracted')
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'features.npy', np.stack(feats, axis=0))
        np.save(out_dir / 'labels.npy', np.array(labels, dtype=object))

    model, id_to_label = load_model(processed_root, device=device)
    feats = np.load(out_dir / 'features.npy')
    L = feats.shape[0]
    pad_to = 300
    if L < pad_to:
        pad = pad_to - L
        feats_p = np.pad(feats, ((0,pad),(0,0)), mode='constant')
        mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad, dtype=bool)])
    else:
        feats_p = feats[:pad_to]
        mask = np.ones(pad_to, dtype=bool)
    x = torch.from_numpy(feats_p).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x, mask=mask_t)
        probs = torch.softmax(logits, dim=-1)[0]
        pred = probs.argmax(dim=-1).cpu().numpy()[:L]
        conf = probs.max(dim=-1).values.cpu().numpy()[:L]
    # collapse
    steps = []
    current = pred[0]
    confs = [conf[0]]
    for i in range(1, L):
        if pred[i] == current:
            confs.append(conf[i])
        else:
            steps.append((id_to_label[int(current)], float(np.mean(confs))))
            current = pred[i]
            confs = [conf[i]]
    steps.append((id_to_label[int(current)], float(np.mean(confs))))
    print('
'.join([f"Step {i+1}: {s} (conf {c:.2f})" for i,(s,c) in enumerate(steps)]))

    # optional: ask Ollama to make it pretty
    try:
        summary = ollama_summarize(steps, model_name='llama3')
        print('
--- Ollama summary ---
', summary)
    except Exception as e:
        print('Ollama summarization failed:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', required=True)
    parser.add_argument('--processed_root', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    infer_on_recording(Path(args.recording), Path(args.processed_root), device=args.device)
```

### 6) ollama_client.py
```python
# ollama_client.py
import requests
import json
from typing import List, Tuple

OLLAMA_BASE = 'http://localhost:11434'


def ollama_summarize(steps: List[Tuple[str, float]], model_name: str = 'llama3') -> str:
    """Call local Ollama server to produce a natural-language summary for `steps`.
    Tries /api/generate then /v1/chat/completions as fallback.
    Returns the assistant text.
    """
    system_prompt = "You are an assistant that summarizes UI action sequences into a short natural description and a numbered step list. Be concise."
    user_text = 'Given the following steps (label and confidence), write a 2-4 sentence summary and then a numbered list with each step:

'
    for i, (lbl, conf) in enumerate(steps):
        user_text += f"{i+1}. {lbl} (confidence {conf:.2f})
"

    # Try /api/generate
    payload = {"model": model_name, "prompt": system_prompt + "

User:
" + user_text, "stream": False}
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", data=json.dumps(payload), headers={'Content-Type':'application/json'}, timeout=10)
        if r.status_code == 200:
            j = r.json()
            # ollama generate returns {'response': '...'} or similar
            if 'response' in j:
                return j['response']
            # older/newer versions might return choices
            if 'choices' in j:
                return j['choices'][0].get('message', {}).get('content', '') or str(j)
            return str(j)
    except Exception as e:
        # ignore and try chat endpoint
        pass

    # fallback to /v1/chat/completions (OpenAI-compatible)
    try:
        chat_payload = {"model": model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}], "stream": False}
        r = requests.post(f"{OLLAMA_BASE}/v1/chat/completions", data=json.dumps(chat_payload), headers={'Content-Type':'application/json'}, timeout=10)
        r.raise_for_status()
        j = r.json()
        if 'choices' in j and len(j['choices']) > 0:
            return j['choices'][0]['message']['content']
        return str(j)
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")
```

---

## How to run (summary)
1. `pip install torch torchvision pandas numpy tqdm requests` (pick a torch build with CUDA if you want GPU)
2. Place dataset folders under `dataset_root` and ensure each recording folder has `annotations.csv` with lines formatted like:
   `chunk_000.mp4 | app:switch | notepad`
3. Run preprocessing:
   `python preprocess.py --dataset_root /path/to/dataset_root --processed_root /path/to/processed_root --device cuda`
4. Train:
   `python train.py --processed_root /path/to/processed_root --epochs 8 --batch_size 4 --device cuda`
5. Inference/Summarize:
   `python infer.py --recording /path/to/dataset_root/recording_001 --processed_root /path/to/processed_root --device cuda`

Notes & caveats:
- This POC uses torchvision.video `read_video` which may be slower than ffmpeg but reads entire clip reliably. If you have very long chunks adjust `target_frames`.
- Feature extractor uses `r3d_18` pretrained on Kinetics — it should give robust clip-level embeddings. You can swap for other video models.
- Ollama: make sure `ollama serve` (or the Ollama app) is running locally. The client tries `/api/generate` and then `/v1/chat/completions`.

If you want, I can now:
- split these into actual separate files in the canvas for you (I can create separate code docs one per file),
- adapt the label mapping to preserve your textual labels exactly (currently the dataset builds a label->id map automatically),
- change the feature extractor to use optical flow + RGB fusion,
- or implement a lightweight evaluation metric (per-chunk accuracy / F1) and print it during training.

Tell me which of those you'd like and I'll update the canvas files accordingly.
