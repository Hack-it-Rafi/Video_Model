import torch
import os

# Data paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

# Model config
MODEL_NAME = 'mvit_v1_b'  # From torchvision, transformer-based video model
NUM_ACTION_CLASSES = None  # To be set dynamically
NUM_APP_CLASSES = None     # To be set dynamically
PRETRAINED = True

# Training config
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIXED_PRECISION = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Temporal smoothing
WINDOW_SIZE = 3  # ±1 clip, total 3

# Other
SEED = 42