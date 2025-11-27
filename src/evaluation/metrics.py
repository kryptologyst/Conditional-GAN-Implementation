"""
Evaluation metrics for Conditional GAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from typing import List, Tuple, Optional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class InceptionScore:
    """Calculate Inception Score for generated images."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        # Load pre-trained Inception model
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        self.inception = self.inception.to(self.device)
        
        # Remove the final classification layer
        self.inception.fc = nn.Identity()
    
    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images using Inception model."""
        # Resize images to 299x299 for Inception
        if images.size(-1) != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Normalize to [0, 1] then to ImageNet stats
        images = (images + 1) / 2  # [-1, 1] to [0, 1]
        images = transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225])(images)
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features
    
    def calculate_score(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """
        Calculate Inception Score.
        
        Args:
            images: Generated images tensor
            splits: Number of splits for calculation
            
        Returns:
            Tuple of (mean_score, std_score)
        """
        features = self.get_features(images)
        
        # Calculate softmax probabilities
        probs = F.softmax(features, dim=1)
        
        # Split into batches
        batch_size = len(images) // splits
        scores = []
        
        for i in range(splits):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            if start_idx >= len(images):
                break
                
            batch_probs = probs[start_idx:end_idx]
            
            # Calculate marginal distribution
            marginal = torch.mean(batch_probs, dim=0)
            
            # Calculate KL divergence
            kl_div = batch_probs * (torch.log(batch_probs + 1e-16) - torch.log(marginal + 1e-16))
            kl_div = torch.sum(kl_div, dim=1)
            
            # Calculate IS
            is_score = torch.exp(torch.mean(kl_div))
            scores.append(is_score.item())
        
        return np.mean(scores), np.std(scores)


class FIDScore:
    """Calculate Fréchet Inception Distance (FID) for generated images."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        # Load pre-trained Inception model
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        self.inception = self.inception.to(self.device)
        
        # Remove the final classification layer
        self.inception.fc = nn.Identity()
    
    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images using Inception model."""
        # Resize images to 299x299 for Inception
        if images.size(-1) != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Normalize to [0, 1] then to ImageNet stats
        images = (images + 1) / 2  # [-1, 1] to [0, 1]
        images = transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225])(images)
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features
    
    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Calculate FID between real and fake images.
        
        Args:
            real_images: Real images tensor
            fake_images: Generated images tensor
            
        Returns:
            FID score
        """
        # Extract features
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # Convert to numpy
        real_features = real_features.cpu().numpy()
        fake_features = fake_features.cpu().numpy()
        
        # Calculate mean and covariance
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        
        # Calculate trace of sqrt of product of covariances
        covmean = self._sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
        
        return fid
    
    def _sqrtm(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix square root."""
        try:
            from scipy.linalg import sqrtm
            return sqrtm(matrix)
        except ImportError:
            # Fallback implementation
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            return eigenvecs.dot(np.diag(np.sqrt(eigenvals))).dot(eigenvecs.T)


class PrecisionRecall:
    """Calculate Precision and Recall for generated images."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        # Load pre-trained Inception model
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        self.inception = self.inception.to(self.device)
        
        # Remove the final classification layer
        self.inception.fc = nn.Identity()
    
    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images using Inception model."""
        # Resize images to 299x299 for Inception
        if images.size(-1) != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Normalize to [0, 1] then to ImageNet stats
        images = (images + 1) / 2  # [-1, 1] to [0, 1]
        images = transforms.Normalize(mean=[0.485, 0.486, 0.406], std=[0.229, 0.224, 0.225])(images)
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features
    
    def calculate_precision_recall(self, real_images: torch.Tensor, fake_images: torch.Tensor, 
                                 k: int = 5) -> Tuple[float, float]:
        """
        Calculate Precision and Recall.
        
        Args:
            real_images: Real images tensor
            fake_images: Generated images tensor
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (precision, recall)
        """
        # Extract features
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # Calculate pairwise distances
        real_distances = torch.cdist(real_features, real_features)
        fake_distances = torch.cdist(fake_features, fake_features)
        cross_distances = torch.cdist(fake_features, real_features)
        
        # Calculate Precision
        fake_to_real_distances = torch.min(cross_distances, dim=1)[0]
        fake_to_fake_distances = torch.topk(fake_distances, k=k+1, dim=1, largest=False)[0][:, 1:]
        fake_to_fake_distances = torch.mean(fake_to_fake_distances, dim=1)
        
        precision = torch.mean((fake_to_real_distances < fake_to_fake_distances).float()).item()
        
        # Calculate Recall
        real_to_fake_distances = torch.min(cross_distances, dim=0)[0]
        real_to_real_distances = torch.topk(real_distances, k=k+1, dim=1, largest=False)[0][:, 1:]
        real_to_real_distances = torch.mean(real_to_real_distances, dim=1)
        
        recall = torch.mean((real_to_fake_distances < real_to_real_distances).float()).item()
        
        return precision, recall


def evaluate_model(generator: nn.Module, real_loader: DataLoader, device: torch.device,
                  z_dim: int = 100, label_dim: int = 10, num_samples: int = 1000) -> dict:
    """
    Comprehensive evaluation of the trained generator.
    
    Args:
        generator: Trained generator model
        real_loader: DataLoader for real images
        device: Device to run on
        z_dim: Noise dimension
        label_dim: Number of classes
        num_samples: Number of samples to generate for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    generator.eval()
    
    # Generate fake images
    fake_images = []
    with torch.no_grad():
        for _ in range(num_samples // 32 + 1):
            batch_size = min(32, num_samples - len(fake_images))
            if batch_size <= 0:
                break
                
            z = torch.randn(batch_size, z_dim, device=device)
            labels = torch.randint(0, label_dim, (batch_size,), device=device)
            label_one_hot = torch.zeros(batch_size, label_dim, device=device)
            label_one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            fake_batch = generator(z, label_one_hot)
            fake_images.append(fake_batch)
    
    fake_images = torch.cat(fake_images, dim=0)[:num_samples]
    
    # Get real images
    real_images = []
    for batch_images, _ in real_loader:
        real_images.append(batch_images.to(device))
        if len(torch.cat(real_images, dim=0)) >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Calculate metrics
    results = {}
    
    # Inception Score
    try:
        is_calculator = InceptionScore(device)
        is_mean, is_std = is_calculator.calculate_score(fake_images)
        results['inception_score'] = {'mean': is_mean, 'std': is_std}
    except Exception as e:
        print(f"Inception Score calculation failed: {e}")
        results['inception_score'] = {'mean': 0.0, 'std': 0.0}
    
    # FID Score
    try:
        fid_calculator = FIDScore(device)
        fid_score = fid_calculator.calculate_fid(real_images, fake_images)
        results['fid_score'] = fid_score
    except Exception as e:
        print(f"FID calculation failed: {e}")
        results['fid_score'] = float('inf')
    
    # Precision and Recall
    try:
        pr_calculator = PrecisionRecall(device)
        precision, recall = pr_calculator.calculate_precision_recall(real_images, fake_images)
        results['precision'] = precision
        results['recall'] = recall
    except Exception as e:
        print(f"Precision/Recall calculation failed: {e}")
        results['precision'] = 0.0
        results['recall'] = 0.0
    
    return results
