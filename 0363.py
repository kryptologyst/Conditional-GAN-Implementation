"""
Conditional GAN Implementation

A Conditional Generative Adversarial Network (cGAN) is an extension of the GAN architecture 
that generates images based on conditional information, such as labels. For example, a cGAN 
can generate images of handwritten digits (from MNIST) based on a digit label (0-9), or it 
can generate images of clothes conditioned on the type of clothing (e.g., shirts, pants).

In this project, we implement a cGAN where the generator and discriminator take both random 
noise and a label as input, allowing for the generation of specific types of images based 
on the provided label.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Tuple, Optional


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


# Set seed for reproducibility
set_seed(42)
device = get_device()
print(f"Using device: {device}")
 
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
 
def create_one_hot_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create one-hot encoded labels."""
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot


def initialize_models(z_dim: int = 100, label_dim: int = 10, img_size: int = 28) -> Tuple[Generator, Discriminator]:
    """Initialize generator and discriminator models."""
    generator = Generator(z_dim=z_dim, label_dim=label_dim, img_size=img_size)
    discriminator = Discriminator(label_dim=label_dim, img_size=img_size)
    
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    return generator, discriminator


def initialize_optimizers(generator: Generator, discriminator: Discriminator, lr: float = 0.0002) -> Tuple[optim.Adam, optim.Adam]:
    """Initialize optimizers for generator and discriminator."""
    beta1 = 0.5
    beta2 = 0.999
    
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    return optimizer_g, optimizer_d


# Initialize models and optimizers
z_dim = 100  # Latent vector dimension (noise)
label_dim = 10  # Number of classes for MNIST (0-9)
img_size = 28  # MNIST image size

generator, discriminator = initialize_models(z_dim, label_dim, img_size)
optimizer_g, optimizer_d = initialize_optimizers(generator, discriminator)

# Loss function (using BCE with logits for numerical stability)
criterion = nn.BCEWithLogitsLoss()
 
def load_mnist_data(batch_size: int = 64, data_dir: str = './data') -> DataLoader:
    """Load MNIST dataset with proper transforms."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return train_loader


def train_cgan(generator: Generator, discriminator: Discriminator, 
               optimizer_g: optim.Adam, optimizer_d: optim.Adam,
               train_loader: DataLoader, num_epochs: int = 50,
               z_dim: int = 100, label_dim: int = 10) -> None:
    """Train the conditional GAN."""
    
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
            generate_and_save_samples(generator, z_dim, label_dim, epoch + 1)
    
    print("Training completed!")
    return g_losses, d_losses


def generate_and_save_samples(generator: Generator, z_dim: int, label_dim: int, epoch: int) -> None:
    """Generate and save sample images."""
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
        plt.savefig(f'samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    generator.train()


def main():
    """Main function to run the conditional GAN training."""
    print("Conditional GAN Implementation")
    print("=" * 40)
    
    # Load data and start training
    train_loader = load_mnist_data(batch_size=64)
    g_losses, d_losses = train_cgan(
        generator, discriminator, optimizer_g, optimizer_d,
        train_loader, num_epochs=50, z_dim=z_dim, label_dim=label_dim
    )
    
    # Plot training curves
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
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nTraining completed! Check the generated sample images and training curves.")


if __name__ == "__main__":
    main()