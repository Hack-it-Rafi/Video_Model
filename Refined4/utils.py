# project/utils.py

import torch
import os
import pickle

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, strict: bool = False):
    """Load a model checkpoint.

    If the architecture has changed (e.g., different number of classes), this will
    automatically skip parameters whose shapes don't match and load the rest.

    Args:
        model: torch.nn.Module
        path: checkpoint path
        strict: when True, enforce exact key/shape match (will raise on mismatch)

    Returns:
        bool indicating whether a checkpoint file was found and processed.
    """
    if not os.path.exists(path):
        return False

    state = torch.load(path, map_location='cpu')

    if strict:
        model.load_state_dict(state, strict=True)
        return True

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            # key missing in current model or shape mismatch
            cur_shape = tuple(model_state[k].shape) if k in model_state else None
            skipped.append((k, tuple(v.shape), cur_shape))

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    if skipped:
        print("[load_checkpoint] Skipped incompatible parameters:")
        for k, ckpt_shape, cur_shape in skipped:
            print(f"  - {k}: checkpoint={ckpt_shape}, current={cur_shape}")

    if missing:
        print(f"[load_checkpoint] Missing keys not loaded (count={len(missing)}).")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys in checkpoint (count={len(unexpected)}).")

    return True

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