# project/trainer.py

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from config import DEVICE, MIXED_PRECISION, LEARNING_RATE, WEIGHT_DECAY, EPOCHS
from utils import save_checkpoint

def train(model, train_loader, action_weights, app_weights, optimizer, scheduler=None):
    model.train()
    action_criterion = nn.CrossEntropyLoss(weight=action_weights.to(DEVICE))
    app_criterion = nn.CrossEntropyLoss(weight=app_weights.to(DEVICE))
    
    scaler = GradScaler() if MIXED_PRECISION else None
    
    total_loss = 0
    for videos, action_labels, app_labels, _, _ in train_loader:
        videos = videos.to(DEVICE)
        action_labels = action_labels.to(DEVICE)
        app_labels = app_labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        if MIXED_PRECISION:
            with autocast():
                action_logits, app_logits = model(videos)
                action_loss = action_criterion(action_logits, action_labels)
                app_loss = app_criterion(app_logits, app_labels)
                loss = action_loss + app_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            action_logits, app_logits = model(videos)
            action_loss = action_criterion(action_logits, action_labels)
            app_loss = app_criterion(app_logits, app_labels)
            loss = action_loss + app_loss
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    if scheduler:
        scheduler.step()
    
    return total_loss / len(train_loader)

def train_model(model, train_loader, test_loader, action_weights, app_weights, train_df):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, action_weights, app_weights, optimizer)
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}')
    
    save_checkpoint(model, 'model.pth')