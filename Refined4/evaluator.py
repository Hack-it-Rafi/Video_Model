# project/evaluator.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import DEVICE

def evaluate(model, loader, action_encoder, app_encoder, is_test=False):
    model.eval()
    all_action_preds, all_action_labels = [], []
    all_app_preds, all_app_labels = [], []
    all_action_confs, all_app_confs = [], []
    chunk_ids = []
    video_ids = []
    
    with torch.no_grad():
        for videos, action_labels, app_labels, chunks, vids in loader:
            videos = videos.to(DEVICE)
            action_logits, app_logits = model(videos)
            
            action_probs = F.softmax(action_logits, dim=1)
            app_probs = F.softmax(app_logits, dim=1)
            
            action_pred = torch.argmax(action_probs, dim=1).cpu()
            app_pred = torch.argmax(app_probs, dim=1).cpu()
            
            action_conf = torch.max(action_probs, dim=1)[0].cpu()
            app_conf = torch.max(app_probs, dim=1)[0].cpu()
            
            all_action_preds.extend(action_pred.numpy())
            all_action_labels.extend(action_labels.numpy())
            all_app_preds.extend(app_pred.numpy())
            all_app_labels.extend(app_labels.numpy())
            all_action_confs.extend(action_conf.numpy())
            all_app_confs.extend(app_conf.numpy())
            chunk_ids.extend(chunks)
            video_ids.extend(vids)
    
    # Metrics for action
    action_acc = accuracy_score(all_action_labels, all_action_preds)
    action_prec = precision_score(all_action_labels, all_action_preds, average='macro', zero_division=0)
    action_rec = recall_score(all_action_labels, all_action_preds, average='macro', zero_division=0)
    action_f1 = f1_score(all_action_labels, all_action_preds, average='macro', zero_division=0)
    action_per_class_rec = recall_score(all_action_labels, all_action_preds, average=None, zero_division=0)
    action_mean_conf = sum(all_action_confs) / len(all_action_confs)
    
    # Metrics for app
    app_acc = accuracy_score(all_app_labels, all_app_preds)
    app_prec = precision_score(all_app_labels, all_app_preds, average='macro', zero_division=0)
    app_rec = recall_score(all_app_labels, all_app_preds, average='macro', zero_division=0)
    app_f1 = f1_score(all_app_labels, all_app_preds, average='macro', zero_division=0)
    app_per_class_rec = recall_score(all_app_labels, all_app_preds, average=None, zero_division=0)
    app_mean_conf = sum(all_app_confs) / len(all_app_confs)
    
    metrics = {
        'action': {
            'accuracy': action_acc,
            'precision': action_prec,
            'recall': action_rec,
            'f1': action_f1,
            'per_class_recall': {action_encoder.classes_[i]: r for i, r in enumerate(action_per_class_rec)},
            'mean_conf': action_mean_conf
        },
        'app': {
            'accuracy': app_acc,
            'precision': app_prec,
            'recall': app_rec,
            'f1': app_f1,
            'per_class_recall': {app_encoder.classes_[i]: r for i, r in enumerate(app_per_class_rec)},
            'mean_conf': app_mean_conf
        }
    }
    
    if is_test:
        print("Evaluation Metrics:")
        print(metrics)
    
    # For smoothing, return probs if needed, but here we return preds for now
    return metrics, all_action_preds, all_app_preds, all_action_confs, all_app_confs, chunk_ids, video_ids