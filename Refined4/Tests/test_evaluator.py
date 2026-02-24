import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock
from sklearn.preprocessing import LabelEncoder
from evaluator import evaluate
from model import VideoClassifier


class TestEvaluator(unittest.TestCase):
    """Test evaluator module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_action_classes = 5
        self.num_app_classes = 3
        self.model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        
        # Create mock encoders
        self.action_encoder = LabelEncoder()
        self.action_encoder.classes_ = ['click', 'scroll', 'type', 'drag', 'hover']
        
        self.app_encoder = LabelEncoder()
        self.app_encoder.classes_ = ['chrome', 'firefox', 'none']
    
    def test_evaluate_mock_loader(self):
        """Test evaluate function with mock data"""
        # Create mock data loader with one batch
        mock_batch = (
            torch.randn(2, 3, 16, 224, 224),  # videos
            torch.tensor([0, 1]),  # action_labels
            torch.tensor([0, 1]),  # app_labels
            ['chunk_001', 'chunk_002'],  # chunks
            ['videos_001', 'videos_001']  # vids
        )
        
        mock_loader = MagicMock()
        mock_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        metrics, action_preds, app_preds, action_confs, app_confs, chunk_ids, video_ids = evaluate(
            self.model, mock_loader, self.action_encoder, self.app_encoder, is_test=False
        )
        
        # Check metrics structure
        self.assertIn('action', metrics)
        self.assertIn('app', metrics)
        self.assertIn('accuracy', metrics['action'])
        self.assertIn('precision', metrics['action'])
        self.assertIn('recall', metrics['action'])
        self.assertIn('f1', metrics['action'])
        
        # Check predictions
        self.assertEqual(len(action_preds), 2)
        self.assertEqual(len(app_preds), 2)
        self.assertEqual(len(action_confs), 2)
        self.assertEqual(len(app_confs), 2)


class TestMetrics(unittest.TestCase):
    """Test metric calculations"""
    
    def test_accuracy_perfect(self):
        """Test accuracy calculation with perfect predictions"""
        from sklearn.metrics import accuracy_score
        
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(acc, 1.0)
    
    def test_accuracy_half(self):
        """Test accuracy calculation with 50% correct"""
        from sklearn.metrics import accuracy_score
        
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 1, 1]
        
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(acc, 0.5)
    
    def test_f1_score(self):
        """Test F1 score calculation"""
        from sklearn.metrics import f1_score
        
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1, 1]
        
        f1 = f1_score(y_true, y_pred, average='macro')
        
        self.assertGreater(f1, 0)
        self.assertLessEqual(f1, 1.0)
    
    def test_precision_recall(self):
        """Test precision and recall calculations"""
        from sklearn.metrics import precision_score, recall_score
        
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1]
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1.0)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1.0)


class TestSoftmax(unittest.TestCase):
    """Test softmax probability computation"""
    
    def test_softmax_sum_to_one(self):
        """Test softmax outputs sum to 1"""
        logits = torch.randn(4, 5)
        probs = F.softmax(logits, dim=1)
        
        # Each row should sum to 1
        row_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(4)))
    
    def test_softmax_positive(self):
        """Test softmax outputs are positive"""
        logits = torch.randn(4, 5)
        probs = F.softmax(logits, dim=1)
        
        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())
    
    def test_argmax_prediction(self):
        """Test argmax gives correct class prediction"""
        logits = torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        self.assertEqual(preds[0].item(), 1)  # Index with highest value
        self.assertEqual(preds[1].item(), 2)


if __name__ == '__main__':
    unittest.main()
