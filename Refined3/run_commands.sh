# 1. Make sure you're in the project root directory
cd video-workflow-classifier

# 2. (Recommended) Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# or on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# or directly:
pip install torch torchvision pandas numpy tqdm

# ───────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────

# Basic training (adjust paths as needed)
python src/train.py data/annotations.csv data/videos/ models/model.pth --batch_size 8 --epochs 12 --lr 1e-4

# With more control (recommended)
# python src/train.py data/annotations.csv data/videos/ models/mvit_finetuned_epoch12.pth --batch_size 6 --epochs 20 --lr 3e-5
python src/train.py data/annotations.csv data/videos/ models/x3d_finetuned_epoch12.pth --batch_size 6 --epochs 20 --lr 3e-5
python src/train.py data/annotations.csv data/videos/ models/swin3d_finetuned_epoch12.pth --batch_size 6 --epochs 20 --lr 3e-5
python src/train.py data/annotations.csv data/videos/ models/s3d_finetuned_epoch12.pth --batch_size 4 --epochs 20 --lr 3e-5
python src/train.py data/annotations.csv data/videos/ models/r2plus1d_finetuned_epoch12.pth --batch_size 4 --epochs 20 --lr 3e-5
# ───────────────────────────────────────────────────────────────
# Evaluation & Inference (with temporal smoothing + JSON output)
# ───────────────────────────────────────────────────────────────

# Full evaluation on test split + smoothed predictions
python src/evaluate.py data/annotations.csv data/videos/ models/model.pth outputs/predictions_smoothed.json --batch_size 12 --window 2

# Quick test with smaller window and bigger batch
python src/evaluate.py data/annotations.csv data/videos/ models/model.pth outputs/test.json --batch_size 16 --window 1

# ───────────────────────────────────────────────────────────────
# Convenience one-liner script example (optional)
# ───────────────────────────────────────────────────────────────

# You can create run_commands.sh with content like:

#!/usr/bin/env bash

MODE=$1

if [ "$MODE" = "train" ]; then
    python src/train.py data/annotations.csv data/videos/ models/model.pth \
        --batch_size 8 --epochs 15 --lr 1e-4
elif [ "$MODE" = "eval" ]; then
    python src/evaluate.py data/annotations.csv data/videos/ models/model.pth \
        outputs/predictions.json --batch_size 12 --window 2
else
    echo "Usage: ./run_commands.sh {train|eval}"
fi

# Then make executable and use:
chmod +x run_commands.sh
./run_commands.sh train
./run_commands.sh eval