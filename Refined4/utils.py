import torch
import os

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return True
    return False