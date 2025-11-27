"""
Unit tests for Conditional GAN models.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cgan import Generator, Discriminator, initialize_models
from data.mnist import create_one_hot_labels, get_data_info
from utils.training import set_seed, get_device


class TestGenerator:
    """Test cases for Generator model."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = Generator(z_dim=100, label_dim=10, img_size=28)
        
        assert generator.z_dim == 100
        assert generator.label_dim == 10
        assert generator.img_size == 28
        assert generator.img_channels == 1
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = Generator(z_dim=100, label_dim=10, img_size=28)
        
        batch_size = 4
        z = torch.randn(batch_size, 100)
        labels = torch.zeros(batch_size, 10)
        labels[:, 0] = 1  # All samples are class 0
        
        output = generator(z, labels)
        
        assert output.shape == (batch_size, 1, 28, 28)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # Tanh output range
    
    def test_generator_different_classes(self):
        """Test generator with different class labels."""
        generator = Generator(z_dim=100, label_dim=10, img_size=28)
        
        batch_size = 2
        z = torch.randn(batch_size, 100)
        
        # Test different classes
        for class_idx in range(10):
            labels = torch.zeros(batch_size, 10)
            labels[:, class_idx] = 1
            
            output = generator(z, labels)
            assert output.shape == (batch_size, 1, 28, 28)


class TestDiscriminator:
    """Test cases for Discriminator model."""
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        discriminator = Discriminator(label_dim=10, img_size=28)
        
        assert discriminator.label_dim == 10
        assert discriminator.img_size == 28
        assert discriminator.img_channels == 1
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = Discriminator(label_dim=10, img_size=28)
        
        batch_size = 4
        images = torch.randn(batch_size, 1, 28, 28)
        labels = torch.zeros(batch_size, 10)
        labels[:, 0] = 1  # All samples are class 0
        
        output = discriminator(images, labels)
        
        assert output.shape == (batch_size, 1)
        # Output should be raw logits (no activation)
        assert torch.all(torch.isfinite(output))
    
    def test_discriminator_different_classes(self):
        """Test discriminator with different class labels."""
        discriminator = Discriminator(label_dim=10, img_size=28)
        
        batch_size = 2
        images = torch.randn(batch_size, 1, 28, 28)
        
        # Test different classes
        for class_idx in range(10):
            labels = torch.zeros(batch_size, 10)
            labels[:, class_idx] = 1
            
            output = discriminator(images, labels)
            assert output.shape == (batch_size, 1)


class TestDataUtils:
    """Test cases for data utilities."""
    
    def test_create_one_hot_labels(self):
        """Test one-hot label creation."""
        labels = torch.tensor([0, 1, 2, 3])
        num_classes = 10
        
        one_hot = create_one_hot_labels(labels, num_classes)
        
        assert one_hot.shape == (4, 10)
        assert torch.all(one_hot[0, 0] == 1)  # First sample is class 0
        assert torch.all(one_hot[1, 1] == 1)  # Second sample is class 1
        assert torch.all(one_hot[2, 2] == 1)  # Third sample is class 2
        assert torch.all(one_hot[3, 3] == 1)  # Fourth sample is class 3
    
    def test_get_data_info(self):
        """Test data info function."""
        info = get_data_info()
        
        assert info['name'] == 'MNIST'
        assert info['num_classes'] == 10
        assert info['img_size'] == 28
        assert info['img_channels'] == 1
        assert len(info['class_names']) == 10


class TestTrainingUtils:
    """Test cases for training utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed is set
        torch.manual_seed(42)
        rand1 = torch.randn(1)
        
        torch.manual_seed(42)
        rand2 = torch.randn(1)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_initialize_models(self):
        """Test model initialization."""
        generator, discriminator = initialize_models(z_dim=100, label_dim=10, img_size=28)
        
        assert isinstance(generator, Generator)
        assert isinstance(discriminator, Discriminator)
        
        # Test that models can be moved to device
        device = get_device()
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        
        assert next(generator.parameters()).device == device
        assert next(discriminator.parameters()).device == device


class TestIntegration:
    """Integration tests."""
    
    def test_generator_discriminator_integration(self):
        """Test generator and discriminator working together."""
        generator = Generator(z_dim=100, label_dim=10, img_size=28)
        discriminator = Discriminator(label_dim=10, img_size=28)
        
        batch_size = 4
        z = torch.randn(batch_size, 100)
        labels = torch.zeros(batch_size, 10)
        labels[:, 0] = 1  # All samples are class 0
        
        # Generate fake images
        fake_images = generator(z, labels)
        
        # Discriminate fake images
        fake_outputs = discriminator(fake_images, labels)
        
        assert fake_images.shape == (batch_size, 1, 28, 28)
        assert fake_outputs.shape == (batch_size, 1)
    
    def test_training_step_simulation(self):
        """Test a simulated training step."""
        generator = Generator(z_dim=100, label_dim=10, img_size=28)
        discriminator = Discriminator(label_dim=10, img_size=28)
        
        # Loss function
        criterion = torch.nn.BCEWithLogitsLoss()
        
        batch_size = 4
        real_images = torch.randn(batch_size, 1, 28, 28)
        labels = torch.zeros(batch_size, 10)
        labels[:, 0] = 1  # All samples are class 0
        
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Generate fake images
        z = torch.randn(batch_size, 100)
        fake_images = generator(z, labels)
        
        # Discriminator loss
        real_outputs = discriminator(real_images, labels)
        fake_outputs = discriminator(fake_images.detach(), labels)
        
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        # Generator loss
        fake_outputs = discriminator(fake_images, labels)
        g_loss = criterion(fake_outputs, real_labels)
        
        assert torch.isfinite(d_loss)
        assert torch.isfinite(g_loss)
        assert d_loss > 0
        assert g_loss > 0


if __name__ == "__main__":
    pytest.main([__file__])
