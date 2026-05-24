import torch
import os

# Data paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

# Video config
NUM_FRAMES = 32  # Use all 30 frames, padded to 32 to match MViT-v2 positional encoding

# Model config
MODEL_NAME = 'mvit_v2_s'  # MViT-v2 Small: significantly better than v1, trained on 32-frame clips
NUM_ACTION_CLASSES = None  # To be set dynamically
NUM_APP_CLASSES = None     # To be set dynamically
PRETRAINED = True

# Training config
BATCH_SIZE = 4          # Larger batch size — no GPU limit
EPOCHS = 30             # More epochs for the stronger model to converge
LEARNING_RATE = 3e-5   # Slightly lower LR for fine-tuning large transformer
WEIGHT_DECAY = 1e-3    # Regularization
MIXED_PRECISION = True  # Enabled — AMP is safe with MViT-v2 and saves memory/time
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