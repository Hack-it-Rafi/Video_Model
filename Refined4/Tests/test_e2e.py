import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
from unittest.mock import Mock, patch, MagicMock
from model import VideoClassifier
from trainer import train_epoch
from evaluator import evaluate
from smoother import apply_smoothing
from utils import save_checkpoint, load_checkpoint
from config import NUM_FRAMES


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflow scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_action_classes = 5
        self.num_app_classes = 3
        self.model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_model_forward_backward_pass(self):
        """Test complete forward and backward pass"""
        # Create dummy input and labels
        videos = torch.randn(2, 3, NUM_FRAMES, 224, 224)
        action_labels = torch.randint(0, self.num_action_classes, (2,))
        app_labels = torch.randint(0, self.num_app_classes, (2,))
        
        # Forward pass
        action_logits, app_logits = self.model(videos)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        action_loss = criterion(action_logits, action_labels)
        app_loss = criterion(app_logits, app_labels)
        total_loss = action_loss + app_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        self.assertTrue(has_gradients)
    
    def test_train_eval_cycle(self):
        """Test switching between train and eval modes"""
        # Start in eval mode
        self.model.eval()
        self.assertFalse(self.model.training)
        
        # Switch to train
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Switch back to eval
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_save_load_inference_cycle(self):
        """Test save model, load it, and perform inference"""
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pth')
        
        # Save model
        save_checkpoint(self.model, checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        load_checkpoint(new_model, checkpoint_path)
        
        # Perform inference
        new_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, NUM_FRAMES, 224, 224)
            action_logits, app_logits = new_model(dummy_input)
        
        self.assertEqual(action_logits.shape, (1, self.num_action_classes))
        self.assertEqual(app_logits.shape, (1, self.num_app_classes))
    
    def test_optimizer_step(self):
        """Test optimizer updates model parameters"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Get initial parameter values
        initial_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # Perform forward and backward pass
        videos = torch.randn(2, 3, NUM_FRAMES, 224, 224)
        action_labels = torch.randint(0, self.num_action_classes, (2,))
        app_labels = torch.randint(0, self.num_app_classes, (2,))
        
        action_logits, app_logits = self.model(videos)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(action_logits, action_labels) + criterion(app_logits, app_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        parameters_changed = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if not torch.allclose(param.data, initial_params[name]):
                    parameters_changed = True
                    break
        
        self.assertTrue(parameters_changed)


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline from loading to prediction"""
    
    def test_batch_processing(self):
        """Test processing a batch of videos"""
        model = VideoClassifier(5, 3)
        model.eval()
        
        # Create batch
        batch_size = 4
        videos = torch.randn(batch_size, 3, NUM_FRAMES, 224, 224)
        
        with torch.no_grad():
            action_logits, app_logits = model(videos)
        
        # Check batch dimension preserved
        self.assertEqual(action_logits.shape[0], batch_size)
        self.assertEqual(app_logits.shape[0], batch_size)
    
    def test_prediction_extraction(self):
        """Test extracting predictions from logits"""
        logits = torch.randn(4, 5)
        
        # Get predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = torch.max(probs, dim=1)[0]
        
        self.assertEqual(preds.shape[0], 4)
        self.assertEqual(confs.shape[0], 4)
        
        # Check predictions are valid class indices
        self.assertTrue(all(0 <= p < 5 for p in preds))
        
        # Check confidences in valid range
        self.assertTrue(all(0 <= c <= 1 for c in confs))


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def test_invalid_input_shape(self):
        """Test model handles invalid input shape gracefully"""
        model = VideoClassifier(5, 3)
        
        # Wrong input shape (missing frame dimension)
        with self.assertRaises((RuntimeError, ValueError)):
            invalid_input = torch.randn(2, 3, 224, 224)  # Missing time dimension
            model(invalid_input)
    
    def test_empty_batch(self):
        """Test handling of empty batch"""
        model = VideoClassifier(5, 3)
        
        # Batch size 0
        empty_batch = torch.randn(0, 3, NUM_FRAMES, 224, 224)
        
        try:
            action_logits, app_logits = model(empty_batch)
            self.assertEqual(action_logits.shape[0], 0)
            self.assertEqual(app_logits.shape[0], 0)
        except RuntimeError:
            # Some models may not support empty batches
            pass
    
    def test_nan_in_input(self):
        """Test detection of NaN in input"""
        videos = torch.randn(2, 3, NUM_FRAMES, 224, 224)
        videos[0, 0, 0, 0, 0] = float('nan')
        
        self.assertTrue(torch.isnan(videos).any())


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency"""
    
    def test_no_grad_inference(self):
        """Test inference without gradient computation"""
        model = VideoClassifier(5, 3)
        model.eval()
        
        videos = torch.randn(2, 3, NUM_FRAMES, 224, 224)
        
        with torch.no_grad():
            action_logits, app_logits = model(videos)
        
        # No gradients should be computed
        self.assertFalse(action_logits.requires_grad)
        self.assertFalse(app_logits.requires_grad)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation for larger effective batch size"""
        model = VideoClassifier(5, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        accumulation_steps = 2
        
        for step in range(accumulation_steps):
            videos = torch.randn(1, 3, NUM_FRAMES, 224, 224)
            action_labels = torch.randint(0, 5, (1,))
            app_labels = torch.randint(0, 3, (1,))
            
            action_logits, app_logits = model(videos)
            loss = criterion(action_logits, action_labels) + criterion(app_logits, app_labels)
            loss = loss / accumulation_steps
            loss.backward()
        
        # Gradients should be accumulated
        has_gradients = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_gradients)
        
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    unittest.main()
