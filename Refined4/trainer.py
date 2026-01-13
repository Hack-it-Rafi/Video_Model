import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from config import (DEVICE, MIXED_PRECISION, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
                    ACTION_LOSS_WEIGHT, APP_LOSS_WEIGHT, LABEL_SMOOTHING, 
                    MAX_GRAD_NORM, LR_SCHEDULER, WARMUP_EPOCHS, PATIENCE)
from utils import save_checkpoint
from evaluator import evaluate
import math

def train_epoch(model, train_loader, action_weights, app_weights, optimizer, scheduler, scaler, epoch):
    model.train()
    action_criterion = nn.CrossEntropyLoss(weight=action_weights.to(DEVICE), label_smoothing=LABEL_SMOOTHING)
    app_criterion = nn.CrossEntropyLoss(weight=app_weights.to(DEVICE), label_smoothing=LABEL_SMOOTHING)
    
    total_loss = 0
    action_loss_sum = 0
    app_loss_sum = 0
    
    for batch_idx, (videos, action_labels, app_labels, _, _) in enumerate(train_loader):
        videos = videos.to(DEVICE)
        action_labels = action_labels.to(DEVICE)
        app_labels = app_labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        if MIXED_PRECISION:
            with autocast(device_type='cuda'):
                action_logits, app_logits = model(videos)
                action_loss = action_criterion(action_logits, action_labels)
                app_loss = app_criterion(app_logits, app_labels)
                loss = ACTION_LOSS_WEIGHT * action_loss + APP_LOSS_WEIGHT * app_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            action_logits, app_logits = model(videos)
            action_loss = action_criterion(action_logits, action_labels)
            app_loss = app_criterion(app_logits, app_labels)
            loss = ACTION_LOSS_WEIGHT * action_loss + APP_LOSS_WEIGHT * app_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
        
        total_loss += loss.item()
        action_loss_sum += action_loss.item()
        app_loss_sum += app_loss.item()
        
        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] - '
                  f'Loss: {loss.item():.4f}, Action: {action_loss.item():.4f}, App: {app_loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    avg_action_loss = action_loss_sum / len(train_loader)
    avg_app_loss = app_loss_sum / len(train_loader)
    
    return avg_loss, avg_action_loss, avg_app_loss

def get_lr_scheduler(optimizer):
    if LR_SCHEDULER == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
                return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif LR_SCHEDULER == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        return None

def train_model(model, train_loader, test_loader, action_weights, app_weights, train_df, action_encoder, app_encoder):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_lr_scheduler(optimizer)
    scaler = GradScaler() if MIXED_PRECISION else None
    
    best_val_f1 = 0
    patience_counter = 0
    
    print(f"\n=== Starting Training ===")
    print(f"Epochs: {EPOCHS}, Batch Size: {train_loader.batch_size}")
    print(f"Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    print(f"Action Loss Weight: {ACTION_LOSS_WEIGHT}, App Loss Weight: {APP_LOSS_WEIGHT}")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch+1}/{EPOCHS} (LR: {current_lr:.6f})')
        
        # Train
        train_loss, action_loss, app_loss = train_epoch(
            model, train_loader, action_weights, app_weights, 
            optimizer, scheduler, scaler, epoch
        )
        print(f'Train Loss: {train_loss:.4f} (Action: {action_loss:.4f}, App: {app_loss:.4f})')
        
        # Validate
        print("Running validation...")
        val_metrics, _, _, _, _, _, _ = evaluate(model, test_loader, action_encoder, app_encoder, is_test=False)
        
        val_action_acc = val_metrics['action']['accuracy']
        val_action_f1 = val_metrics['action']['f1']
        val_app_acc = val_metrics['app']['accuracy']
        val_app_f1 = val_metrics['app']['f1']
        
        # Combined F1 for early stopping
        combined_f1 = (val_action_f1 + val_app_f1) / 2
        
        print(f'Validation - Action: Acc={val_action_acc:.4f}, F1={val_action_f1:.4f} | '
              f'App: Acc={val_app_acc:.4f}, F1={val_app_f1:.4f}')
        
        # Save best model
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            patience_counter = 0
            save_checkpoint(model, 'model.pth')
            print(f'✓ Saved best model (Combined F1: {combined_f1:.4f})')
        else:
            patience_counter += 1
            print(f'No improvement ({patience_counter}/{PATIENCE})')
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
        
        if scheduler:
            scheduler.step()
    
    print(f'\n=== Training Complete ===')
    print(f'Best Combined F1: {best_val_f1:.4f}')