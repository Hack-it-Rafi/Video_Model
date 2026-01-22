import torch
import os

# Data paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

# Video config
NUM_FRAMES = 16  # Sample to 16 frames to match MViT positional encoding (from 3-second ~30 frame videos)

# Model config
MODEL_NAME = 'mvit_v1_b'  # From torchvision, transformer-based video model
NUM_ACTION_CLASSES = None  # To be set dynamically
NUM_APP_CLASSES = None     # To be set dynamically
PRETRAINED = True

# Training config
BATCH_SIZE = 2  # Reduced to 1 to avoid CUDA memory issues with MViT
EPOCHS = 20  # Increased from 10
LEARNING_RATE = 5e-5  # Reduced for more stable training
WEIGHT_DECAY = 1e-3  # Increased regularization
MIXED_PRECISION = False  # Disabled to avoid CUBLAS errors with MViT
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss balancing
ACTION_LOSS_WEIGHT = 2.0  # Action is harder, give it more weight
APP_LOSS_WEIGHT = 1.0

# Label smoothing to prevent overconfidence
LABEL_SMOOTHING = 0.1

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Learning rate schedule
LR_SCHEDULER = 'cosine'  # 'cosine' or 'step'
WARMUP_EPOCHS = 2

# Early stopping
PATIENCE = 5

# Temporal smoothing
WINDOW_SIZE = 3  # ±1 clip, total 3

# Other
SEED = 42