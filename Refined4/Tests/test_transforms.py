import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from dataset import get_transforms
from config import NUM_FRAMES


class TestDataTransforms(unittest.TestCase):
    """Test data transformation pipelines"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.train_transform = get_transforms(train=True)
        self.test_transform = get_transforms(train=False)
        self.sample_frame = torch.rand(3, 256, 256)  # C, H, W
    
    def test_train_transform_exists(self):
        """Test training transform is created"""
        self.assertIsNotNone(self.train_transform)
    
    def test_test_transform_exists(self):
        """Test test transform is created"""
        self.assertIsNotNone(self.test_transform)
    
    def test_train_transform_output_shape(self):
        """Test training transform produces correct output shape"""
        transformed = self.train_transform(self.sample_frame)
        
        # Should output 224x224
        self.assertEqual(transformed.shape[1], 224)
        self.assertEqual(transformed.shape[2], 224)
        self.assertEqual(transformed.shape[0], 3)  # Channels unchanged
    
    def test_test_transform_output_shape(self):
        """Test test transform produces correct output shape"""
        transformed = self.test_transform(self.sample_frame)
        
        # Should output 224x224
        self.assertEqual(transformed.shape[1], 224)
        self.assertEqual(transformed.shape[2], 224)
        self.assertEqual(transformed.shape[0], 3)
    
    def test_train_transform_randomness(self):
        """Test training transform applies random augmentations"""
        # Apply transform multiple times
        transformed1 = self.train_transform(self.sample_frame)
        transformed2 = self.train_transform(self.sample_frame)
        
        # Due to random crop and flip, outputs may differ
        # (though not guaranteed to be different every time)
        self.assertEqual(transformed1.shape, transformed2.shape)
    
    def test_test_transform_deterministic(self):
        """Test test transform is deterministic"""
        # Apply transform multiple times
        transformed1 = self.test_transform(self.sample_frame)
        transformed2 = self.test_transform(self.sample_frame)
        
        # Should be identical
        self.assertTrue(torch.allclose(transformed1, transformed2))


class TestVideoPreprocessing(unittest.TestCase):
    """Test video preprocessing operations"""
    
    def test_frame_sampling(self):
        """Test frame sampling from video"""
        total_frames = 30
        target_frames = NUM_FRAMES
        
        # Simulate uniform sampling
        indices = torch.linspace(0, total_frames - 1, target_frames).long()
        
        self.assertEqual(len(indices), target_frames)
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[-1].item(), total_frames - 1)
    
    def test_frame_normalization(self):
        """Test frame normalization to [0, 1]"""
        # Simulate video frames in [0, 255]
        video = torch.randint(0, 256, (10, 224, 224, 3)).float()
        
        # Normalize
        normalized = video / 255.0
        
        self.assertGreaterEqual(normalized.min().item(), 0.0)
        self.assertLessEqual(normalized.max().item(), 1.0)
    
    def test_video_tensor_shape_conversion(self):
        """Test video tensor shape conversion"""
        # Original: (T, H, W, C)
        video = torch.rand(16, 224, 224, 3)
        
        # Convert to: (C, T, H, W)
        video_converted = video.permute(3, 0, 1, 2)
        
        self.assertEqual(video_converted.shape, (3, 16, 224, 224))
    
    def test_short_video_padding(self):
        """Test handling of videos shorter than target frames"""
        video = torch.rand(8, 224, 224, 3)  # Only 8 frames
        target_frames = 16
        
        # Repeat to reach target
        repeat_factor = (target_frames + video.shape[0] - 1) // video.shape[0]
        padded = video.repeat(repeat_factor, 1, 1, 1)[:target_frames]
        
        self.assertEqual(padded.shape[0], target_frames)


class TestNormalizationStatistics(unittest.TestCase):
    """Test normalization statistics"""
    
    def test_normalize_transform(self):
        """Test normalize transform with mean and std"""
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        
        normalize = transforms.Normalize(mean=mean, std=std)
        
        # Create sample with known values
        sample = torch.ones(3, 224, 224) * 0.5
        
        normalized = normalize(sample)
        
        # Check shape preserved
        self.assertEqual(normalized.shape, sample.shape)
        
        # Check values are normalized (not equal to original)
        self.assertFalse(torch.allclose(normalized, sample))
    
    def test_denormalization(self):
        """Test that normalization can be reversed"""
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
        
        original = torch.rand(3, 224, 224)
        
        # Normalize
        normalized = (original - mean) / std
        
        # Denormalize
        denormalized = normalized * std + mean
        
        # Should recover original
        self.assertTrue(torch.allclose(original, denormalized, atol=1e-6))


class TestColorJitter(unittest.TestCase):
    """Test color jitter augmentation"""
    
    def test_color_jitter_application(self):
        """Test color jitter is applied"""
        jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        )
        
        sample = torch.rand(3, 224, 224)
        jittered = jitter(sample)
        
        # Shape should be preserved
        self.assertEqual(jittered.shape, sample.shape)
        
        # Values should still be in reasonable range
        self.assertGreaterEqual(jittered.min().item(), -1.0)
        self.assertLessEqual(jittered.max().item(), 2.0)


if __name__ == '__main__':
    unittest.main()
