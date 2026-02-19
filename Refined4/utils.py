# project/utils.py

import torch
import os
import pickle

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return True
    return False

def save_encoders(action_encoder, app_encoder, path='encoders.pkl'):
    """Save label encoders to disk"""
    with open(path, 'wb') as f:
        pickle.dump({'action': action_encoder, 'app': app_encoder}, f)
    print(f"Encoders saved to {path}")

def load_encoders(path='encoders.pkl'):
    """Load label encoders from disk"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            encoders = pickle.load(f)
        return encoders['action'], encoders['app']
    return None, None