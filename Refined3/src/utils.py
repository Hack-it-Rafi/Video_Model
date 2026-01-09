# utils.py
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def get_label_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    actions = sorted(df['action'].unique())
    apps = sorted(df['target_app'].fillna('none').unique())
    action_map = {a: i for i, a in enumerate(actions)}
    app_map = {a: i for i, a in enumerate(apps)}
    rev_action_map = {i: a for a, i in action_map.items()}
    rev_app_map = {i: a for a, i in app_map.items()}
    return action_map, app_map, rev_action_map, rev_app_map

def get_class_weights(df: pd.DataFrame, label_map: Dict[str, int], col: str, fillna=None) -> torch.Tensor:
    if fillna:
        counts = df[col].fillna(fillna).value_counts()
    else:
        counts = df[col].value_counts()
    class_counts = np.array([counts.get(k, 0) for k in label_map])
    class_weights = class_counts.max() / (class_counts + 1e-6)  # Inverse frequency, avoid div0
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
    return torch.tensor(class_weights, dtype=torch.float32)

def compute_metrics(labels: np.ndarray, preds: np.ndarray, confs: np.ndarray, num_classes: int) -> Dict[str, float]:
    accuracy = (labels == preds).mean()
    mean_conf = confs.mean()
    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for l, p in zip(labels, preds):
        conf_matrix[l, p] += 1
    # Per-class recall
    per_class_recall = np.array([conf_matrix[i, i] / conf_matrix[i].sum() if conf_matrix[i].sum() > 0 else 0 for i in range(num_classes)])
    # Macro precision, recall, f1
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    return {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'per_class_recall': per_class_recall.tolist(),
        'mean_confidence': mean_conf
    }

def temporal_smoothing(probs: torch.Tensor, preds: torch.Tensor, window: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    # probs: N, num_classes
    # preds: N
    N = probs.size(0)
    smoothed_probs = probs.clone()
    for i in range(N):
        start = max(0, i - window)
        end = min(N, i + window + 1)
        avg_prob = probs[start:end].mean(dim=0)
        smoothed_probs[i] = avg_prob
    new_preds = smoothed_probs.argmax(dim=1)
    # Fallback for ties: majority vote on original preds
    for i in range(N):
        if (smoothed_probs[i] == smoothed_probs[i].max()).sum() > 1:
            start = max(0, i - window)
            end = min(N, i + window + 1)
            window_preds = preds[start:end]
            majority = torch.mode(window_preds)[0]
            new_preds[i] = majority
    return smoothed_probs, new_preds