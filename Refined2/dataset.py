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
        
        # Load metadata if available
        meta_path = p / 'meta.npy'
        if meta_path.exists():
            metas = np.load(meta_path, allow_pickle=True)  # (L,)
        else:
            metas = np.array([''] * len(labels_raw), dtype=object)
        
        L = feats.shape[0]
        if L > self.max_seq_len:
            feats = feats[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            metas = metas[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.bool_)
        else:
            pad = self.max_seq_len - L
            feats = np.pad(feats, ((0, pad), (0, 0)), mode='constant')
            labels = np.pad(labels, ((0, pad),), mode='constant', constant_values=-100)
            metas = np.pad(metas, ((0, pad),), mode='constant', constant_values='')
            mask = np.concatenate([np.ones(L, dtype=np.bool_), np.zeros(pad, dtype=np.bool_)])
        return torch.from_numpy(feats).float(), torch.from_numpy(labels).long(), torch.from_numpy(mask).bool(), metas
