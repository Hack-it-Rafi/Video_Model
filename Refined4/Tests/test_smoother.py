import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from smoother import temporal_smoothing, apply_smoothing
from model import VideoClassifier
from unittest.mock import Mock, MagicMock


class TestTemporalSmoothing(unittest.TestCase):
    """Test temporal smoothing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 5
        self.window_size = 3
    
    def test_temporal_smoothing_output_length(self):
        """Test smoothing preserves sequence length"""
        seq_length = 10
        action_probs_seq = [torch.randn(self.num_classes).softmax(0) for _ in range(seq_length)]
        app_probs_seq = [torch.randn(self.num_classes).softmax(0) for _ in range(seq_length)]
        
        smoothed_action, smoothed_app = temporal_smoothing(
            action_probs_seq, app_probs_seq, self.window_size
        )
        
        self.assertEqual(len(smoothed_action), seq_length)
        self.assertEqual(len(smoothed_app), seq_length)
    
    def test_temporal_smoothing_probability_valid(self):
        """Test smoothed probabilities are valid"""
        seq_length = 5
        action_probs_seq = [torch.randn(self.num_classes).softmax(0) for _ in range(seq_length)]
        app_probs_seq = [torch.randn(self.num_classes).softmax(0) for _ in range(seq_length)]
        
        smoothed_action, smoothed_app = temporal_smoothing(
            action_probs_seq, app_probs_seq, self.window_size
        )
        
        # Check each smoothed probability is valid
        for prob in smoothed_action:
            self.assertTrue((prob >= 0).all())
            self.assertTrue((prob <= 1).all())
            # Should approximately sum to 1 (within tolerance for averaging)
            self.assertAlmostEqual(prob.sum().item(), 1.0, places=5)
    
    def test_temporal_smoothing_reduces_noise(self):
        """Test smoothing reduces temporal jitter"""
        # Create noisy sequence that alternates between two classes
        seq_length = 10
        action_probs_seq = []
        for i in range(seq_length):
            probs = torch.zeros(self.num_classes)
            # Alternate between class 0 and class 1
            probs[i % 2] = 1.0
            action_probs_seq.append(probs)
        
        app_probs_seq = action_probs_seq.copy()
        
        smoothed_action, _ = temporal_smoothing(
            action_probs_seq, app_probs_seq, self.window_size
        )
        
        # Smoothed probabilities should be more balanced
        for prob in smoothed_action:
            max_prob = prob.max().item()
            # Max prob should be less than 1.0 due to smoothing
            self.assertLess(max_prob, 1.0)
    
    def test_temporal_smoothing_window_size_1(self):
        """Test smoothing with window size 1 (no smoothing)"""
        seq_length = 5
        action_probs_seq = [torch.randn(self.num_classes).softmax(0) for _ in range(seq_length)]
        app_probs_seq = [torch.randn(self.num_classes).softmax(0) for _ in range(seq_length)]
        
        smoothed_action, smoothed_app = temporal_smoothing(
            action_probs_seq, app_probs_seq, window_size=1
        )
        
        # With window size 1, output should be identical to input
        for orig, smooth in zip(action_probs_seq, smoothed_action):
            self.assertTrue(torch.allclose(orig, smooth, atol=1e-6))


class TestApplySmoothing(unittest.TestCase):
    """Test apply_smoothing function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_action_classes = 5
        self.num_app_classes = 3
        self.model = VideoClassifier(self.num_action_classes, self.num_app_classes)
    
    def test_apply_smoothing_mock_loader(self):
        """Test apply_smoothing with mock data loader"""
        # Create mock batch with multiple chunks from same video
        mock_batch = (
            torch.randn(3, 3, 16, 224, 224),  # videos
            torch.tensor([0, 1, 0]),  # action_labels (unused in smoothing)
            torch.tensor([0, 1, 0]),  # app_labels (unused in smoothing)
            ['chunk_001.mp4', 'chunk_002.mp4', 'chunk_003.mp4'],  # chunks
            ['videos_001', 'videos_001', 'videos_001']  # vids (same video)
        )
        
        mock_loader = MagicMock()
        mock_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        smoothed_outputs = apply_smoothing(self.model, mock_loader)
        
        # Check output structure
        self.assertIsInstance(smoothed_outputs, dict)
        self.assertEqual(len(smoothed_outputs), 3)
        
        # Check each output has required fields
        for key, value in smoothed_outputs.items():
            self.assertIn('action', value)
            self.assertIn('action_conf', value)
            self.assertIn('app', value)
            self.assertIn('app_conf', value)
    
    def test_apply_smoothing_confidence_range(self):
        """Test smoothed confidences are in valid range"""
        mock_batch = (
            torch.randn(2, 3, 16, 224, 224),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            ['chunk_001.mp4', 'chunk_002.mp4'],
            ['videos_001', 'videos_001']
        )
        
        mock_loader = MagicMock()
        mock_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        smoothed_outputs = apply_smoothing(self.model, mock_loader)
        
        for value in smoothed_outputs.values():
            self.assertGreaterEqual(value['action_conf'], 0.0)
            self.assertLessEqual(value['action_conf'], 1.0)
            self.assertGreaterEqual(value['app_conf'], 0.0)
            self.assertLessEqual(value['app_conf'], 1.0)


if __name__ == '__main__':
    unittest.main()
