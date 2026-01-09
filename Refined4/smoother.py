import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import torch

from config import DEVICE

def temporal_smoothing(action_probs_seq, app_probs_seq, window_size=3):
    # Assume probs_seq is list of tensors (num_clips, num_classes)
    # Pad for window
    pad = (window_size - 1) // 2
    action_probs_padded = F.pad(torch.stack(action_probs_seq), (0, 0, pad, pad), mode='replicate')
    app_probs_padded = F.pad(torch.stack(app_probs_seq), (0, 0, pad, pad), mode='replicate')
    
    smoothed_action_probs = []
    smoothed_app_probs = []
    for i in range(len(action_probs_seq)):
        window_action = action_probs_padded[i:i+window_size].mean(dim=0)
        window_app = app_probs_padded[i:i+window_size].mean(dim=0)
        smoothed_action_probs.append(window_action)
        smoothed_app_probs.append(window_app)
    
    # If tie (unlikely after avg), use majority vote
    # But since probs, just argmax
    return smoothed_action_probs, smoothed_app_probs

def apply_smoothing(model, loader):
    model.eval()
    video_chunks = defaultdict(list)
    video_action_probs = defaultdict(list)
    video_app_probs = defaultdict(list)
    
    with torch.no_grad():
        for videos, _, _, chunks, vids in loader:
            videos = videos.to(DEVICE)
            action_logits, app_logits = model(videos)
            action_probs = F.softmax(action_logits, dim=1).cpu()
            app_probs = F.softmax(app_logits, dim=1).cpu()
            
            for vid, chunk, a_prob, ap_prob in zip(vids, chunks, action_probs, app_probs):
                video_chunks[vid].append(chunk)
                video_action_probs[vid].append(a_prob)
                video_app_probs[vid].append(ap_prob)
    
    smoothed_outputs = {}
    for vid in video_chunks:
        # Sort by chunk id assuming chunk_001, etc.
        indices = np.argsort([int(c.split('_')[1].split('.')[0]) for c in video_chunks[vid]])
        sorted_action_probs = [video_action_probs[vid][i] for i in indices]
        sorted_app_probs = [video_app_probs[vid][i] for i in indices]
        sorted_chunks = [video_chunks[vid][i] for i in indices]
        
        smoothed_a_probs, smoothed_ap_probs = temporal_smoothing(sorted_action_probs, sorted_app_probs)
        
        for chunk, a_prob, ap_prob in zip(sorted_chunks, smoothed_a_probs, smoothed_ap_probs):
            action = torch.argmax(a_prob).item()
            app = torch.argmax(ap_prob).item()
            action_conf = torch.max(a_prob).item()
            app_conf = torch.max(ap_prob).item()
            smoothed_outputs[(vid, chunk)] = {
                'action': action,
                'action_conf': action_conf,
                'app': app,
                'app_conf': app_conf
            }
    
    return smoothed_outputs