# evaluate.py
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import VideoDataset
from model import VideoClassifier, get_preprocess_params
from utils import get_label_maps, compute_metrics, temporal_smoothing
from train import main as train_main  # For consistency, but not used here

def main(csv_path: str, video_dir: str, model_path: str, output_json: str, batch_size: int = 8, window: int = 1, max_test_samples: int = None):
    df = pd.read_csv(csv_path)
    df['chunk_id'] = df['filename'].apply(lambda x: int(x.split('_')[1].split('.')[0]))
    df = df.sort_values('chunk_id').reset_index(drop=True)
    total = len(df)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    test_df = df.iloc[val_end:]
    # Maps from full
    action_map, app_map, rev_action_map, rev_app_map = get_label_maps(df)
    num_actions = len(action_map)
    num_apps = len(app_map)
    mean, std = get_preprocess_params()
    test_ds = VideoDataset(test_df, video_dir, action_map, app_map, mean, std, max_samples=max_test_samples)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Evaluating with {len(test_ds)} samples (max_test_samples={max_test_samples})")
    
    # Load model
    model = VideoClassifier(num_actions, num_apps)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    # Collect all
    all_logits_a = []
    all_logits_p = []
    all_labels_a = []
    all_labels_p = []
    all_chunk_ids = []
    all_filenames = []
    with torch.no_grad():
        for inputs, labels_a, labels_p, chunk_ids, filenames in test_loader:
            inputs = inputs.cuda()
            logits_a, logits_p = model(inputs)
            all_logits_a.append(logits_a.cpu())
            all_logits_p.append(logits_p.cpu())
            all_labels_a.append(labels_a)
            all_labels_p.append(labels_p)
            all_chunk_ids.extend(chunk_ids.tolist())
            all_filenames.extend(filenames)
    logits_a = torch.cat(all_logits_a)
    logits_p = torch.cat(all_logits_p)
    labels_a = torch.cat(all_labels_a).numpy()
    labels_p = torch.cat(all_labels_p).numpy()
    # Probs
    probs_a = F.softmax(logits_a, dim=1)
    probs_p = F.softmax(logits_p, dim=1)
    confs_a, preds_a = probs_a.max(dim=1)
    confs_p, preds_p = probs_p.max(dim=1)
    preds_a = preds_a.numpy()
    preds_p = preds_p.numpy()
    confs_a = confs_a.numpy()
    confs_p = confs_p.numpy()
    # Metrics before smoothing
    metrics_a = compute_metrics(labels_a, preds_a, confs_a, num_actions)
    metrics_p = compute_metrics(labels_p, preds_p, confs_p, num_apps)
    print("Action Metrics:", metrics_a)
    print("App Metrics:", metrics_p)
    # Smoothing
    _, smoothed_preds_a = temporal_smoothing(probs_a, torch.tensor(preds_a), window)
    _, smoothed_preds_p = temporal_smoothing(probs_p, torch.tensor(preds_p), window)
    smoothed_preds_a = smoothed_preds_a.numpy()
    smoothed_preds_p = smoothed_preds_p.numpy()
    # Recompute confs for smoothed? Use original confs for simplicity, or recompute max from smoothed probs
    smoothed_probs_a, _ = temporal_smoothing(probs_a, torch.tensor(preds_a), window)
    smoothed_probs_p, _ = temporal_smoothing(probs_p, torch.tensor(preds_p), window)
    smoothed_confs_a = smoothed_probs_a.gather(1, smoothed_preds_a.unsqueeze(1)).squeeze().numpy()
    smoothed_confs_p = smoothed_probs_p.gather(1, smoothed_preds_p.unsqueeze(1)).squeeze().numpy()
    # Metrics after
    smoothed_metrics_a = compute_metrics(labels_a, smoothed_preds_a, smoothed_confs_a, num_actions)
    smoothed_metrics_p = compute_metrics(labels_p, smoothed_preds_p, smoothed_confs_p, num_apps)
    print("Smoothed Action Metrics:", smoothed_metrics_a)
    print("Smoothed App Metrics:", smoothed_metrics_p)
    # Output JSON
    outputs = []
    for i, chunk_id in enumerate(all_chunk_ids):
        action = rev_action_map[smoothed_preds_a[i]]
        app = rev_app_map[smoothed_preds_p[i]]
        outputs.append({
            "chunk_id": all_filenames[i].split('.')[0],  # e.g., chunk_010
            "action": action,
            "action_conf": round(smoothed_confs_a[i], 2),
            "app": app,
            "app_conf": round(smoothed_confs_p[i], 2)
        })
    with open(output_json, 'w') as f:
        json.dump(outputs, f, indent=2)

if __name__ == '__main__':
    # Example usage
    # Use max_test_samples to control the amount of test data
    main('annotations.csv', 'videos/', 'model.pth', 'outputs.json', max_test_samples=50)
    # main('annotations.csv', 'videos/', 'model.pth', 'outputs.json')