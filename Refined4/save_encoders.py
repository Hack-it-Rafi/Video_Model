"""
Script to save label encoders from the training data
Run this once if you already have a trained model but missing encoders.pkl
"""

import os
import sys
from dataset import load_dataset
from utils import save_encoders

def main():
    print("Loading dataset to extract encoders...")
    _, _, action_encoder, app_encoder, _, _, _, _ = load_dataset()
    
    print(f"\nAction classes ({len(action_encoder.classes_)}): {action_encoder.classes_[:10]}...")
    print(f"App classes ({len(app_encoder.classes_)}): {app_encoder.classes_}")
    
    # Save encoders
    save_encoders(action_encoder, app_encoder, 'encoders.pkl')
    print("\n✓ Encoders saved successfully!")

if __name__ == '__main__':
    main()
