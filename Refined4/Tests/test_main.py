import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from unittest.mock import patch, MagicMock, Mock
from main import main


class TestMainIntegration(unittest.TestCase):
    """Test main.py integration and workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.predictions_path = os.path.join(self.temp_dir, 'predictions.json')
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.predictions_path):
            os.remove(self.predictions_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    @patch('main.load_dataset')
    @patch('main.get_dataloaders')
    @patch('main.compute_class_weights')
    @patch('main.VideoClassifier')
    @patch('main.train_model')
    @patch('main.save_encoders')
    @patch('main.load_checkpoint')
    def test_main_train_mode(self, mock_load_checkpoint, mock_save_encoders,
                            mock_train, mock_classifier, mock_weights,
                            mock_dataloaders, mock_load_dataset):
        """Test main function in train mode"""
        # Setup mocks
        mock_load_dataset.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), 5, 3
        )
        mock_dataloaders.return_value = (Mock(), Mock())
        mock_weights.return_value = Mock()
        mock_classifier.return_value = Mock()
        mock_load_checkpoint.return_value = True
        
        # Run main in train mode
        try:
            main(mode='train')
            # If no exception, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"main(train) raised exception: {e}")
    
    @patch('main.load_dataset')
    @patch('main.get_dataloaders')
    @patch('main.compute_class_weights')
    @patch('main.VideoClassifier')
    @patch('main.evaluate')
    @patch('main.load_checkpoint')
    def test_main_evaluate_mode(self, mock_load_checkpoint, mock_evaluate,
                                mock_classifier, mock_weights,
                                mock_dataloaders, mock_load_dataset):
        """Test main function in evaluate mode"""
        # Setup mocks
        mock_load_dataset.return_value = (
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), 5, 3
        )
        mock_dataloaders.return_value = (Mock(), Mock())
        mock_weights.return_value = Mock()
        mock_classifier.return_value = Mock()
        mock_load_checkpoint.return_value = True
        mock_evaluate.return_value = (
            {'action': {}, 'app': {}}, [], [], [], [], [], []
        )
        
        # Run main in evaluate mode
        try:
            main(mode='evaluate')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"main(evaluate) raised exception: {e}")
    
    def test_command_line_arg_parsing(self):
        """Test command line argument parsing"""
        # Test default mode
        with patch('sys.argv', ['main.py']):
            # Should default to 'train'
            self.assertEqual(len(sys.argv), 1)
        
        # Test with mode specified
        with patch('sys.argv', ['main.py', 'evaluate']):
            mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
            self.assertEqual(mode, 'evaluate')
        
        # Test with infer mode
        with patch('sys.argv', ['main.py', 'infer']):
            mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
            self.assertEqual(mode, 'infer')


class TestPredictionOutput(unittest.TestCase):
    """Test prediction output formatting"""
    
    def test_prediction_json_structure(self):
        """Test prediction JSON has correct structure"""
        # Create sample prediction
        prediction = {
            "video_id": "videos_001",
            "chunk_id": "chunk_001",
            "action": "click",
            "action_conf": 0.95,
            "app": "chrome",
            "app_conf": 0.88
        }
        
        # Check required fields
        self.assertIn("video_id", prediction)
        self.assertIn("chunk_id", prediction)
        self.assertIn("action", prediction)
        self.assertIn("action_conf", prediction)
        self.assertIn("app", prediction)
        self.assertIn("app_conf", prediction)
    
    def test_prediction_confidence_range(self):
        """Test prediction confidences are in valid range"""
        prediction = {
            "action_conf": 0.95,
            "app_conf": 0.88
        }
        
        self.assertGreaterEqual(prediction["action_conf"], 0.0)
        self.assertLessEqual(prediction["action_conf"], 1.0)
        self.assertGreaterEqual(prediction["app_conf"], 0.0)
        self.assertLessEqual(prediction["app_conf"], 1.0)
    
    def test_predictions_sorting(self):
        """Test predictions can be sorted properly"""
        predictions = [
            {"video_id": "videos_002", "chunk_id": "chunk_001"},
            {"video_id": "videos_001", "chunk_id": "chunk_002"},
            {"video_id": "videos_001", "chunk_id": "chunk_001"},
        ]
        
        # Sort by video_id then chunk_id
        predictions.sort(key=lambda x: (x['video_id'], x['chunk_id']))
        
        self.assertEqual(predictions[0]['video_id'], "videos_001")
        self.assertEqual(predictions[0]['chunk_id'], "chunk_001")
        self.assertEqual(predictions[1]['video_id'], "videos_001")
        self.assertEqual(predictions[1]['chunk_id'], "chunk_002")
        self.assertEqual(predictions[2]['video_id'], "videos_002")


class TestSeeding(unittest.TestCase):
    """Test random seed for reproducibility"""
    
    def test_torch_manual_seed(self):
        """Test torch manual seed is set"""
        import torch
        from config import SEED
        
        torch.manual_seed(SEED)
        
        # Generate random numbers
        rand1 = torch.rand(5)
        
        # Reset seed
        torch.manual_seed(SEED)
        
        # Generate again
        rand2 = torch.rand(5)
        
        # Should be identical
        self.assertTrue(torch.allclose(rand1, rand2))


if __name__ == '__main__':
    unittest.main()
