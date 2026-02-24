import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import pickle
from sklearn.preprocessing import LabelEncoder
from utils import save_checkpoint, load_checkpoint, save_encoders, load_encoders
from model import VideoClassifier


class TestCheckpointFunctions(unittest.TestCase):
    """Test checkpoint save/load functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_action_classes = 5
        self.num_app_classes = 3
        self.model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'test_model.pth')
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test saving model checkpoint"""
        save_checkpoint(self.model, self.checkpoint_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(self.checkpoint_path))
        
        # Check file is not empty
        self.assertGreater(os.path.getsize(self.checkpoint_path), 0)
    
    def test_load_checkpoint_existing(self):
        """Test loading existing checkpoint"""
        # Save checkpoint first
        save_checkpoint(self.model, self.checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        result = load_checkpoint(new_model, self.checkpoint_path)
        
        self.assertTrue(result)
    
    def test_load_checkpoint_nonexistent(self):
        """Test loading non-existent checkpoint"""
        new_model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        result = load_checkpoint(new_model, 'nonexistent.pth')
        
        self.assertFalse(result)
    
    def test_save_load_preserves_weights(self):
        """Test that save/load preserves model weights"""
        # Get initial weights
        initial_weights = {}
        for name, param in self.model.named_parameters():
            initial_weights[name] = param.data.clone()
        
        # Save and load
        save_checkpoint(self.model, self.checkpoint_path)
        new_model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        load_checkpoint(new_model, self.checkpoint_path)
        
        # Compare weights
        for name, param in new_model.named_parameters():
            self.assertTrue(torch.allclose(param.data, initial_weights[name]))


class TestEncoderFunctions(unittest.TestCase):
    """Test encoder save/load functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.action_encoder = LabelEncoder()
        self.action_encoder.fit(['click', 'scroll', 'type', 'drag', 'hover'])
        
        self.app_encoder = LabelEncoder()
        self.app_encoder.fit(['chrome', 'firefox', 'safari', 'none'])
        
        self.temp_dir = tempfile.mkdtemp()
        self.encoder_path = os.path.join(self.temp_dir, 'test_encoders.pkl')
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.encoder_path):
            os.remove(self.encoder_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_save_encoders(self):
        """Test saving encoders"""
        save_encoders(self.action_encoder, self.app_encoder, self.encoder_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(self.encoder_path))
        
        # Check file is not empty
        self.assertGreater(os.path.getsize(self.encoder_path), 0)
    
    def test_load_encoders_existing(self):
        """Test loading existing encoders"""
        # Save encoders first
        save_encoders(self.action_encoder, self.app_encoder, self.encoder_path)
        
        # Load encoders
        loaded_action, loaded_app = load_encoders(self.encoder_path)
        
        self.assertIsNotNone(loaded_action)
        self.assertIsNotNone(loaded_app)
    
    def test_load_encoders_nonexistent(self):
        """Test loading non-existent encoders"""
        loaded_action, loaded_app = load_encoders('nonexistent.pkl')
        
        self.assertIsNone(loaded_action)
        self.assertIsNone(loaded_app)
    
    def test_save_load_preserves_classes(self):
        """Test that save/load preserves encoder classes"""
        # Save and load
        save_encoders(self.action_encoder, self.app_encoder, self.encoder_path)
        loaded_action, loaded_app = load_encoders(self.encoder_path)
        
        # Compare classes
        self.assertEqual(
            list(self.action_encoder.classes_),
            list(loaded_action.classes_)
        )
        self.assertEqual(
            list(self.app_encoder.classes_),
            list(loaded_app.classes_)
        )
    
    def test_save_load_preserves_encoding(self):
        """Test that save/load preserves encoding functionality"""
        # Save and load
        save_encoders(self.action_encoder, self.app_encoder, self.encoder_path)
        loaded_action, loaded_app = load_encoders(self.encoder_path)
        
        # Test encoding
        test_actions = ['click', 'scroll', 'type']
        original_encoded = self.action_encoder.transform(test_actions)
        loaded_encoded = loaded_action.transform(test_actions)
        
        self.assertTrue((original_encoded == loaded_encoded).all())
    
    def test_encoder_inverse_transform(self):
        """Test encoder inverse transform"""
        labels = [0, 1, 2]
        decoded = self.action_encoder.inverse_transform(labels)
        
        self.assertEqual(list(decoded), ['click', 'drag', 'hover'])


if __name__ == '__main__':
    unittest.main()
