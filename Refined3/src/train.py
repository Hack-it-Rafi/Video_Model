# train.py
import os
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
from dataset import VideoDataset
from model import VideoClassifier, get_preprocess_params
from utils import get_label_maps, get_class_weights
from typing import Dict

def main(csv_path: str, video_dir: str, model_path: str, batch_size: int = 8, epochs: int = 10, lr: float = 1e-4, max_train_samples: int = None, max_val_samples: int = None):
    df = pd.read_csv(csv_path)
    df['chunk_id'] = df['filename'].apply(lambda x: int(x.split('_')[1].split('.')[0]))
    df = df.sort_values('chunk_id').reset_index(drop=True)
    total = len(df)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    # Get maps from full df
    action_map, app_map, _, _ = get_label_maps(df)
    num_actions = len(action_map)
    num_apps = len(app_map)
    # Weights
    weights_action = get_class_weights(train_df, action_map, 'action')
    weights_app = get_class_weights(train_df, app_map, 'target_app', 'none')
    # Preprocess
    mean, std = get_preprocess_params()
    # Datasets with max_samples control
    train_ds = VideoDataset(train_df, video_dir, action_map, app_map, mean, std, max_samples=max_train_samples)
    val_ds = VideoDataset(val_df, video_dir, action_map, app_map, mean, std, max_samples=max_val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training with {len(train_ds)} samples (max_train_samples={max_train_samples})")
    print(f"Validating with {len(val_ds)} samples (max_val_samples={max_val_samples})")
    
    # Model
    model = VideoClassifier(num_actions, num_apps)
    model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_action = nn.CrossEntropyLoss(weight=weights_action.cuda())
    criterion_app = nn.CrossEntropyLoss(weight=weights_app.cuda())
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels_a, labels_p, _, _ in train_loader:
            inputs, labels_a, labels_p = inputs.cuda(), labels_a.cuda(), labels_p.cuda()
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits_a, logits_p = model(inputs)
                loss_a = criterion_action(logits_a, labels_a)
                loss_p = criterion_app(logits_p, labels_p)
                loss = loss_a + loss_p
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        # Val
        val_metrics = evaluate(model, val_loader, criterion_action, criterion_app)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_metrics['loss']:.4f}")
    torch.save(model.state_dict(), model_path)

def evaluate(model, loader, criterion_a, criterion_p) -> Dict[str, float]:
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inputs, labels_a, labels_p, _, _ in loader:
            inputs, labels_a, labels_p = inputs.cuda(), labels_a.cuda(), labels_p.cuda()
            logits_a, logits_p = model(inputs)
            loss_a = criterion_a(logits_a, labels_a)
            loss_p = criterion_p(logits_p)
            loss += (loss_a + loss_p).item() * inputs.size(0)
    loss /= len(loader.dataset)
    return {'loss': loss}

if __name__ == '__main__':
    # Example usage: adjust paths
    # Use max_train_samples and max_val_samples to control data amount
    # main('annotations.csv', 'videos/', 'model.pth', max_train_samples=100, max_val_samples=20)
    # main('annotations.csv', 'videos/', 'model.pth')
    parser = argparse.ArgumentParser(description='Train video classification model')
    parser.add_argument('csv_path', type=str, help='Path to annotations CSV file')
    parser.add_argument('video_dir', type=str, help='Directory containing video files')
    parser.add_argument('model_path', type=str, help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Maximum number of validation samples')
    
    args = parser.parse_args()
    
    main(args.csv_path, args.video_dir, args.model_path, 
         batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
         max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples)