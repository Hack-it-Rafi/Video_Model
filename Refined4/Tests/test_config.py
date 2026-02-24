import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import (
    DEVICE, SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    NUM_FRAMES, MODEL_NAME, PRETRAINED, WEIGHT_DECAY,
    ACTION_LOSS_WEIGHT, APP_LOSS_WEIGHT, LABEL_SMOOTHING,
    MAX_GRAD_NORM, LR_SCHEDULER, WARMUP_EPOCHS, PATIENCE,
    WINDOW_SIZE, DATA_ROOT
)


class TestConfig(unittest.TestCase):
    """Test configuration parameters and settings"""
    
    def test_device_config(self):
        """Test that device is properly configured"""
        self.assertIn(DEVICE, ['cuda', 'cpu'])
        if torch.cuda.is_available():
            self.assertEqual(DEVICE, 'cuda')
        else:
            self.assertEqual(DEVICE, 'cpu')
    
    def test_hyperparameters_valid(self):
        """Test that hyperparameters are within valid ranges"""
        self.assertGreater(BATCH_SIZE, 0)
        self.assertGreater(EPOCHS, 0)
        self.assertGreater(LEARNING_RATE, 0)
        self.assertLess(LEARNING_RATE, 1.0)
        self.assertGreaterEqual(WEIGHT_DECAY, 0)
        
    def test_loss_weights(self):
        """Test loss weight configuration"""
        self.assertGreater(ACTION_LOSS_WEIGHT, 0)
        self.assertGreater(APP_LOSS_WEIGHT, 0)
        
    def test_label_smoothing(self):
        """Test label smoothing is in valid range"""
        self.assertGreaterEqual(LABEL_SMOOTHING, 0)
        self.assertLess(LABEL_SMOOTHING, 1.0)
    
    def test_model_config(self):
        """Test model configuration"""
        self.assertEqual(MODEL_NAME, 'mvit_v1_b')
        self.assertIsInstance(PRETRAINED, bool)
        self.assertEqual(NUM_FRAMES, 16)
    
    def test_scheduler_config(self):
        """Test learning rate scheduler configuration"""
        self.assertIn(LR_SCHEDULER, ['cosine', 'step'])
        self.assertGreaterEqual(WARMUP_EPOCHS, 0)
        self.assertLess(WARMUP_EPOCHS, EPOCHS)
    
    def test_early_stopping_config(self):
        """Test early stopping patience"""
        self.assertGreater(PATIENCE, 0)
        self.assertLess(PATIENCE, EPOCHS)
    
    def test_gradient_clipping(self):
        """Test gradient clipping configuration"""
        self.assertGreater(MAX_GRAD_NORM, 0)
    
    def test_temporal_smoothing_config(self):
        """Test temporal smoothing window size"""
        self.assertGreater(WINDOW_SIZE, 0)
        self.assertEqual(WINDOW_SIZE % 2, 1)  # Should be odd
    
    def test_data_root_exists(self):
        """Test that data root directory is configured"""
        self.assertIsNotNone(DATA_ROOT)
        self.assertIsInstance(DATA_ROOT, str)


if __name__ == '__main__':
    unittest.main()
