import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from trainer import train_epoch, get_lr_scheduler, train_model
from model import VideoClassifier
from config import EPOCHS, WARMUP_EPOCHS, LR_SCHEDULER, LEARNING_RATE


class TestTrainerFunctions(unittest.TestCase):
    """Test trainer module functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_action_classes = 5
        self.num_app_classes = 3
        self.model = VideoClassifier(self.num_action_classes, self.num_app_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
    
    def test_get_lr_scheduler_cosine(self):
        """Test cosine learning rate scheduler creation"""
        with patch('config.LR_SCHEDULER', 'cosine'):
            scheduler = get_lr_scheduler(self.optimizer)
            self.assertIsNotNone(scheduler)
    
    def test_get_lr_scheduler_step(self):
        """Test step learning rate scheduler creation"""
        with patch('config.LR_SCHEDULER', 'step'):
            scheduler = get_lr_scheduler(self.optimizer)
            self.assertIsNotNone(scheduler)
    
    def test_lr_warmup(self):
        """Test learning rate warmup schedule"""
        scheduler = get_lr_scheduler(self.optimizer)
        
        if LR_SCHEDULER == 'cosine':
            initial_lr = self.optimizer.param_groups[0]['lr']
            
            # During warmup, LR should increase
            for epoch in range(WARMUP_EPOCHS):
                scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                # LR should be positive
                self.assertGreater(current_lr, 0)
    
    def test_train_epoch_mock(self):
        """Test train_epoch function with mock data"""
        # Create mock data loader
        mock_loader = MagicMock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_loader.__len__ = Mock(return_value=1)
        
        action_weights = torch.ones(self.num_action_classes)
        app_weights = torch.ones(self.num_app_classes)
        
        scheduler = get_lr_scheduler(self.optimizer)
        
        # Should handle empty loader without error
        avg_loss, avg_action_loss, avg_app_loss = train_epoch(
            self.model, mock_loader, action_weights, app_weights,
            self.optimizer, scheduler, None, 0
        )
        
        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(avg_action_loss, float)
        self.assertIsInstance(avg_app_loss, float)


class TestLossComputation(unittest.TestCase):
    """Test loss computation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 5
        self.batch_size = 4
    
    def test_cross_entropy_loss(self):
        """Test cross entropy loss computation"""
        criterion = nn.CrossEntropyLoss()
        
        logits = torch.randn(self.batch_size, self.num_classes)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = criterion(logits, labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_weighted_loss(self):
        """Test weighted cross entropy loss"""
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0])
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        logits = torch.randn(self.batch_size, self.num_classes)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss = criterion(logits, labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_label_smoothing(self):
        """Test label smoothing"""
        criterion_no_smooth = nn.CrossEntropyLoss(label_smoothing=0.0)
        criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        logits = torch.randn(self.batch_size, self.num_classes)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
        loss_no_smooth = criterion_no_smooth(logits, labels)
        loss_smooth = criterion_smooth(logits, labels)
        
        # Losses should be different
        self.assertNotEqual(loss_no_smooth.item(), loss_smooth.item())


class TestGradientClipping(unittest.TestCase):
    """Test gradient clipping"""
    
    def test_gradient_clipping(self):
        """Test gradient clipping reduces large gradients"""
        model = nn.Linear(10, 5)
        
        # Create scenario with large gradients
        x = torch.randn(8, 10)
        y = torch.randint(0, 5, (8,))
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(model(x), y)
        loss.backward()
        
        # Get norm before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        # Apply clipping
        max_norm = 1.0
        total_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # If gradients were large, they should be clipped
        if total_norm_before > max_norm:
            self.assertLessEqual(total_norm_after, max_norm + 1e-6)


if __name__ == '__main__':
    unittest.main()
