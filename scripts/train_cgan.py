"""
Training script for Conditional GAN on MNIST dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from typing import Tuple

from src.models.cgan import Generator, Discriminator, initialize_models
from src.data.mnist import load_mnist_data, create_one_hot_labels
from src.utils.training import (
    set_seed, get_device, initialize_optimizers, 
    generate_and_save_samples, plot_training_curves,
    save_model_checkpoint
)


def train_cgan(generator: Generator, discriminator: Discriminator, 
               optimizer_g: torch.optim.Adam, optimizer_d: torch.optim.Adam,
               train_loader: DataLoader, num_epochs: int = 50,
               z_dim: int = 100, label_dim: int = 10, 
               device: torch.device = None, save_dir: str = None) -> Tuple[list, list]:
    """
    Train the conditional GAN.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        train_loader: Training data loader
        num_epochs: Number of training epochs
        z_dim: Dimension of noise vector
        label_dim: Number of classes
        device: Device to run on
        save_dir: Directory to save outputs
        
    Returns:
        Tuple of (generator_losses, discriminator_losses)
    """
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    g_losses = []
    d_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for i, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            
            # Move data to device
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Create labels for real and fake data
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # One-hot encode labels
            label_one_hot = create_one_hot_labels(labels, label_dim).to(device)
            
            # Train the Discriminator
            optimizer_d.zero_grad()
            
            # Train on real images
            real_outputs = discriminator(real_images, label_one_hot)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Train on fake images
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(z, label_one_hot)
            fake_outputs = discriminator(fake_images.detach(), label_one_hot)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # Train the Generator
            optimizer_g.zero_grad()
            
            # Try to fool the discriminator
            fake_outputs = discriminator(fake_images, label_one_hot)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
        
        # Average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}')
        
        # Generate sample images
        if (epoch + 1) % 10 == 0:
            generate_and_save_samples(
                generator, z_dim, label_dim, epoch + 1, device, save_dir
            )
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0 and save_dir:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_model_checkpoint(
                generator, discriminator, optimizer_g, optimizer_d,
                epoch + 1, avg_g_loss, checkpoint_path
            )
    
    print("Training completed!")
    return g_losses, d_losses


def main():
    """Main function to run the conditional GAN training."""
    parser = argparse.ArgumentParser(description='Train Conditional GAN on MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=100, help='Noise dimension')
    parser.add_argument('--label_dim', type=int, default=10, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./assets', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize models
    generator, discriminator = initialize_models(
        z_dim=args.z_dim, 
        label_dim=args.label_dim, 
        img_size=args.img_size, 
        device=device
    )
    
    # Initialize optimizers
    optimizer_g, optimizer_d = initialize_optimizers(
        generator, discriminator, lr=args.lr
    )
    
    # Load data
    train_loader = load_mnist_data(
        batch_size=args.batch_size, 
        data_dir=args.data_dir
    )
    
    print("Conditional GAN Implementation")
    print("=" * 40)
    print(f"Training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    
    # Train the model
    g_losses, d_losses = train_cgan(
        generator, discriminator, optimizer_g, optimizer_d,
        train_loader, num_epochs=args.epochs, 
        z_dim=args.z_dim, label_dim=args.label_dim,
        device=device, save_dir=args.save_dir
    )
    
    # Plot training curves
    plot_training_curves(g_losses, d_losses, args.save_dir)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.save_dir, 'final_model.pth')
    save_model_checkpoint(
        generator, discriminator, optimizer_g, optimizer_d,
        args.epochs, g_losses[-1], final_checkpoint_path
    )
    
    print(f"\nTraining completed! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()