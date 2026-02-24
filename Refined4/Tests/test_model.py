import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import VideoClassifier
from config import NUM_FRAMES, DEVICE


class TestVideoClassifier(unittest.TestCase):
    """Test VideoClassifier model architecture"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.num_action_classes = 10
        cls.num_app_classes = 5
        cls.model = VideoClassifier(cls.num_action_classes, cls.num_app_classes)
        cls.batch_size = 2
        cls.num_frames = NUM_FRAMES
        cls.height = 224
        cls.width = 224
        cls.channels = 3
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.backbone)
        self.assertIsNotNone(self.model.action_head)
        self.assertIsNotNone(self.model.app_head)
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shapes"""
        # Create dummy input: (batch, channels, frames, height, width)
        dummy_input = torch.randn(
            self.batch_size, 
            self.channels, 
            self.num_frames, 
            self.height, 
            self.width
        )
        
        action_logits, app_logits = self.model(dummy_input)
        
        # Check output shapes
        self.assertEqual(action_logits.shape, (self.batch_size, self.num_action_classes))
        self.assertEqual(app_logits.shape, (self.batch_size, self.num_app_classes))
    
    def test_forward_pass_no_nan(self):
        """Test forward pass doesn't produce NaN values"""
        dummy_input = torch.randn(
            self.batch_size, 
            self.channels, 
            self.num_frames, 
            self.height, 
            self.width
        )
        
        action_logits, app_logits = self.model(dummy_input)
        
        self.assertFalse(torch.isnan(action_logits).any())
        self.assertFalse(torch.isnan(app_logits).any())
    
    def test_model_parameters(self):
        """Test model has trainable parameters"""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)
        
        # Check at least some parameters require gradients
        trainable_params = [p for p in params if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)
    
    def test_model_mode_switch(self):
        """Test model can switch between train and eval modes"""
        self.model.train()
        self.assertTrue(self.model.training)
        
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_dropout_behavior(self):
        """Test dropout is active in train mode and inactive in eval mode"""
        dummy_input = torch.randn(1, self.channels, self.num_frames, self.height, self.width)
        
        # Train mode - multiple forward passes should give different results
        self.model.train()
        with torch.no_grad():
            output1 = self.model(dummy_input)
            output2 = self.model(dummy_input)
        
        # Eval mode - should give consistent results
        self.model.eval()
        with torch.no_grad():
            output3 = self.model(dummy_input)
            output4 = self.model(dummy_input)
        
        # Outputs in eval mode should be identical
        self.assertTrue(torch.allclose(output3[0], output4[0]))
        self.assertTrue(torch.allclose(output3[1], output4[1]))
    
    def test_different_batch_sizes(self):
        """Test model handles different batch sizes"""
        for batch_size in [1, 2, 4]:
            dummy_input = torch.randn(
                batch_size, 
                self.channels, 
                self.num_frames, 
                self.height, 
                self.width
            )
            
            action_logits, app_logits = self.model(dummy_input)
            
            self.assertEqual(action_logits.shape[0], batch_size)
            self.assertEqual(app_logits.shape[0], batch_size)


if __name__ == '__main__':
    unittest.main()
