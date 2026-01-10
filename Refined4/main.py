# project/main.py

import torch
from config import DEVICE, SEED, BATCH_SIZE
from dataset import load_dataset, get_dataloaders, compute_class_weights
from model import VideoClassifier
from trainer import train_model
from evaluator import evaluate
from smoother import apply_smoothing
from utils import load_checkpoint
import json
import sys

torch.manual_seed(SEED)

def main(mode='train'):
    train_dataset, test_dataset, action_encoder, app_encoder, train_df, test_df, num_action_classes, num_app_classes = load_dataset()
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, BATCH_SIZE)
    
    action_weights = compute_class_weights(train_df, num_action_classes, 'action_label')
    app_weights = compute_class_weights(train_df, num_app_classes, 'app_label')
    
    print(f"\n=== Class Weights ===")
    print(f"Action weights (top 5): {action_weights.topk(5)}")
    print(f"App weights: {app_weights}")
    
    model = VideoClassifier(num_action_classes, num_app_classes).to(DEVICE)
    
    if mode == 'train':
        train_model(model, train_loader, test_loader, action_weights, app_weights, train_df, action_encoder, app_encoder)
    
    load_checkpoint(model, 'model.pth')  # Load for eval/infer
    
    if mode == 'evaluate':
        print("\n=== Final Evaluation on Test Set ===")
        metrics, _, _, _, _, _, _ = evaluate(model, test_loader, action_encoder, app_encoder, is_test=True)
    
    if mode == 'infer':
        # Assuming infer on test for example
        smoothed = apply_smoothing(model, test_loader)
        
        outputs = []
        for (vid, chunk), pred in smoothed.items():
            output = {
                "chunk_id": chunk.replace('.mp4', ''),
                "action": action_encoder.inverse_transform([pred['action']])[0],
                "action_conf": round(pred['action_conf'], 2),
                "app": app_encoder.inverse_transform([pred['app']])[0],
                "app_conf": round(pred['app_conf'], 2)
            }
            outputs.append(output)
        
        with open('predictions.json', 'w') as f:
            json.dump(outputs, f, indent=2)
        print(f"Saved {len(outputs)} predictions to predictions.json")

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    main(mode)