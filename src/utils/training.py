"""
Utility functions for Conditional GAN training and evaluation.
"""

import torch
import torch.optim as optim
import random
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import torchvision


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def initialize_optimizers(generator: torch.nn.Module, discriminator: torch.nn.Module, 
                         lr: float = 0.0002) -> Tuple[optim.Adam, optim.Adam]:
    """
    Initialize optimizers for generator and discriminator.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        lr: Learning rate
        
    Returns:
        Tuple of (generator_optimizer, discriminator_optimizer)
    """
    beta1 = 0.5
    beta2 = 0.999
    
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    return optimizer_g, optimizer_d


def generate_and_save_samples(generator: torch.nn.Module, z_dim: int, label_dim: int, 
                            epoch: int, device: torch.device, save_path: str = None) -> None:
    """
    Generate and save sample images.
    
    Args:
        generator: Generator model
        z_dim: Dimension of noise vector
        label_dim: Number of classes
        epoch: Current epoch number
        device: Device to run on
        save_path: Path to save images (optional)
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate samples for each class
        samples_per_class = 8
        all_samples = []
        
        for class_idx in range(label_dim):
            # Create one-hot label for this class
            class_label = torch.zeros(samples_per_class, label_dim, device=device)
            class_label[:, class_idx] = 1
            
            # Generate noise
            z = torch.randn(samples_per_class, z_dim, device=device)
            
            # Generate images
            fake_images = generator(z, class_label)
            all_samples.append(fake_images)
        
        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
        
        # Create grid and save
        grid_img = torchvision.utils.make_grid(all_samples, nrow=samples_per_class, normalize=True)
        
        # Convert to numpy for matplotlib
        grid_np = grid_img.cpu().permute(1, 2, 0).numpy()
        grid_np = (grid_np + 1) / 2  # Denormalize to [0, 1]
        
        # Plot
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np, cmap='gray')
        plt.title(f'Generated Samples - Epoch {epoch}')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    generator.train()


def plot_training_curves(g_losses: list, d_losses: list, save_path: str = None) -> None:
    """
    Plot training curves.
    
    Args:
        g_losses: List of generator losses
        d_losses: List of discriminator losses
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/training_curves.png', dpi=150, bbox_inches='tight')
    
    plt.show()


def save_model_checkpoint(generator: torch.nn.Module, discriminator: torch.nn.Module,
                         optimizer_g: optim.Adam, optimizer_d: optim.Adam,
                         epoch: int, loss: float, save_path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)


def load_model_checkpoint(checkpoint_path: str, generator: torch.nn.Module, 
                         discriminator: torch.nn.Module, optimizer_g: optim.Adam, 
                         optimizer_d: optim.Adam) -> int:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    return checkpoint['epoch']
