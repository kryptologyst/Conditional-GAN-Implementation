"""
Conditional GAN Models

This module contains the Generator and Discriminator models for conditional GANs.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Generator(nn.Module):
    """
    Conditional Generator for cGAN.
    
    Generates images conditioned on class labels using a fully connected architecture.
    
    Args:
        z_dim: Dimension of the noise vector
        label_dim: Number of classes for conditioning
        img_size: Size of generated images (assumes square images)
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, z_dim: int = 100, label_dim: int = 10, img_size: int = 28, hidden_dim: int = 256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim
        self.img_size = img_size
        self.img_channels = 1  # MNIST is grayscale
        
        # Progressive upsampling architecture
        self.fc1 = nn.Linear(z_dim + label_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, img_size * img_size * self.img_channels)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
 
    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Noise tensor of shape (batch_size, z_dim)
            label: One-hot encoded labels of shape (batch_size, label_dim)
            
        Returns:
            Generated images of shape (batch_size, 1, img_size, img_size)
        """
        # Concatenate noise vector and label
        input_tensor = torch.cat((z, label), dim=-1)
        
        # Forward pass with batch normalization and dropout
        x = self.relu(self.bn1(self.fc1(input_tensor)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.tanh(self.fc4(x))
        
        # Reshape to image dimensions
        return x.view(-1, self.img_channels, self.img_size, self.img_size)


class Discriminator(nn.Module):
    """
    Conditional Discriminator for cGAN.
    
    Distinguishes between real and fake images conditioned on class labels.
    
    Args:
        label_dim: Number of classes for conditioning
        img_size: Size of input images (assumes square images)
        hidden_dim: Hidden layer dimension
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, label_dim: int = 10, img_size: int = 28, hidden_dim: int = 256, dropout_rate: float = 0.3):
        super(Discriminator, self).__init__()
        self.label_dim = label_dim
        self.img_size = img_size
        self.img_channels = 1  # MNIST is grayscale
        
        # Progressive downsampling architecture
        self.fc1 = nn.Linear(img_size * img_size * self.img_channels + label_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # Output single value: real or fake
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
 
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input images of shape (batch_size, 1, img_size, img_size)
            label: One-hot encoded labels of shape (batch_size, label_dim)
            
        Returns:
            Discriminator output of shape (batch_size, 1)
        """
        # Flatten the image
        x = x.view(-1, self.img_size * self.img_size * self.img_channels)
        
        # Concatenate image and label
        input_tensor = torch.cat((x, label), dim=-1)
        
        # Forward pass with batch normalization and dropout
        x = self.leaky_relu(self.bn1(self.fc1(input_tensor)))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        # Output layer (no activation for raw logits)
        x = self.fc4(x)
        
        return x


def initialize_models(z_dim: int = 100, label_dim: int = 10, img_size: int = 28, device: torch.device = None) -> Tuple[Generator, Discriminator]:
    """
    Initialize generator and discriminator models.
    
    Args:
        z_dim: Dimension of noise vector
        label_dim: Number of classes
        img_size: Size of images
        device: Device to move models to
        
    Returns:
        Tuple of (generator, discriminator)
    """
    generator = Generator(z_dim=z_dim, label_dim=label_dim, img_size=img_size)
    discriminator = Discriminator(label_dim=label_dim, img_size=img_size)
    
    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
    
    return generator, discriminator
