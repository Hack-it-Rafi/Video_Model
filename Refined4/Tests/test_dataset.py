import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from unittest.mock import Mock, patch
from dataset import (
    VideoDataset, get_transforms, compute_class_weights,
    get_dataloaders
)
from config import NUM_FRAMES, BATCH_SIZE


class TestVideoDataset(unittest.TestCase):
    """Test VideoDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock dataframe
        self.mock_df = pd.DataFrame({
            'filename': ['chunk_001.mp4', 'chunk_002.mp4'],
            'action': ['click', 'scroll'],
            'target_app': ['chrome', 'firefox'],
            'action_label': [0, 1],
            'app_label': [0, 1],
            'video_dir': ['videos_001', 'videos_001'],
            'video_id': ['videos_001', 'videos_001'],
            'chunk_id': ['chunk_001', 'chunk_002']
        })
        self.root_dir = 'mock_data'
    
    def test_dataset_initialization(self):
        """Test dataset initializes with valid data"""
        with patch('os.path.exists', return_value=False):
            dataset = VideoDataset(
                self.mock_df, 
                self.root_dir, 
                transform=None, 
                num_frames=NUM_FRAMES
            )
            # Dataset should filter out non-existent videos
            self.assertGreaterEqual(0, len(dataset))
    
    def test_dataset_length(self):
        """Test dataset length matches valid videos"""
        with patch('os.path.exists', return_value=True):
            dataset = VideoDataset(
                self.mock_df, 
                self.root_dir, 
                transform=None
            )
            self.assertEqual(len(dataset), len(self.mock_df))
    
    def test_transform_application(self):
        """Test transforms are applied correctly"""
        train_transform = get_transforms(train=True)
        test_transform = get_transforms(train=False)
        
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(test_transform)


class TestComputeClassWeights(unittest.TestCase):
    """Test class weight computation"""
    
    def test_balanced_classes(self):
        """Test weights for balanced classes"""
        df = pd.DataFrame({
            'action_label': [0, 1, 2, 0, 1, 2]
        })
        weights = compute_class_weights(df, 3, 'action_label')
        
        self.assertEqual(len(weights), 3)
        # For balanced classes, weights should be close to 1
        self.assertTrue(all(0.5 < w < 1.5 for w in weights))
    
    def test_imbalanced_classes(self):
        """Test weights for imbalanced classes"""
        df = pd.DataFrame({
            'action_label': [0, 0, 0, 0, 1, 2]
        })
        weights = compute_class_weights(df, 3, 'action_label')
        
        self.assertEqual(len(weights), 3)
        # Minority classes should have higher weights
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[2], weights[0])
    
    def test_missing_classes(self):
        """Test weights when some classes are missing from data"""
        df = pd.DataFrame({
            'action_label': [0, 0, 1, 1]
        })
        weights = compute_class_weights(df, 5, 'action_label')
        
        self.assertEqual(len(weights), 5)
        # Missing classes should have weight of 1
        self.assertEqual(weights[2].item(), 1.0)
        self.assertEqual(weights[3].item(), 1.0)
        self.assertEqual(weights[4].item(), 1.0)


class TestDataLoaders(unittest.TestCase):
    """Test data loader creation"""
    
    def setUp(self):
        """Set up mock datasets"""
        self.mock_df = pd.DataFrame({
            'filename': ['chunk_001.mp4'] * 10,
            'action': ['click'] * 10,
            'target_app': ['chrome'] * 10,
            'action_label': [0] * 10,
            'app_label': [0] * 10,
            'video_dir': ['videos_001'] * 10,
            'video_id': ['videos_001'] * 10,
            'chunk_id': [f'chunk_{i:03d}' for i in range(10)]
        })
    
    def test_dataloader_creation(self):
        """Test dataloaders are created properly"""
        with patch('os.path.exists', return_value=True):
            train_dataset = VideoDataset(self.mock_df, 'mock', get_transforms(True))
            test_dataset = VideoDataset(self.mock_df, 'mock', get_transforms(False))
            
            train_loader, test_loader = get_dataloaders(
                train_dataset, test_dataset, BATCH_SIZE
            )
            
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(test_loader)
            self.assertEqual(train_loader.batch_size, BATCH_SIZE)
            self.assertEqual(test_loader.batch_size, BATCH_SIZE)


if __name__ == '__main__':
    unittest.main()
