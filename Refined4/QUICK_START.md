# Quick Start

> Make sure you have Python dependencies and ffmpeg installed. See [README.md](README.md) for setup.

---

## 1. Training

```bash
python main.py train
```

What happens:
- Loads all video sessions from `data/`
- Splits 80/20 by session (no data leakage)
- Trains **MViT-v2 Small** on all 32 frames per clip
- Saves best checkpoint to `model.pth` (tracked by combined action+app F1)
- Saves label encoders to `encoders.pkl`
- Early stops after 5 epochs with no improvement (max 30 epochs)

**Key config knobs** (`config.py`):

| Setting | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | `4` | Reduce if VRAM is tight |
| `EPOCHS` | `30` | Hard cap; early stopping applies |
| `LEARNING_RATE` | `3e-5` | AdamW, cosine decay + 2-epoch warmup |
| `PATIENCE` | `5` | Early stopping |
| `MIXED_PRECISION` | `True` | Disable if you hit AMP errors |
| `NUM_FRAMES` | `32` | Full 3-sec clip coverage |

**Expected output:**
```
Epoch 1/30 (LR: 0.000015)
  Batch [20/50] - Loss: 2.4561, Action: 1.8234, App: 0.6327
Train Loss: 2.3821 (Action: 1.7902, App: 0.5919)
Validation - Action: Acc=0.4560, F1=0.4231 | App: Acc=0.6780, F1=0.6512
✓ Saved best model (Combined F1: 0.5372)
```

---

## 2. Evaluation

```bash
python main.py evaluate
```

Loads `model.pth` and runs inference on the held-out test split. Prints:
- Accuracy, Precision, Recall, F1 (macro) for actions and apps
- Per-class recall breakdown
- Mean confidence scores

> No temporal smoothing is applied here — raw per-clip predictions only.

---

## 3. Inference

```bash
python main.py infer
```

Runs on the test split with **temporal smoothing** applied across consecutive clips.  
Writes results to `predictions.json`:

```json
[
  {
    "video_id": "videos_013",
    "chunk_id": "chunk_000",
    "action": "click",
    "action_conf": 0.89,
    "app": "chrome",
    "app_conf": 0.95
  }
]
```

Also prints a per-video summary to stdout.

---

