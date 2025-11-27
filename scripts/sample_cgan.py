"""
Sampling script for Conditional GAN - generate images with specific conditions.
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import torchvision
from typing import List, Optional

from src.models.cgan import Generator
from src.utils.training import set_seed, get_device


def generate_samples(generator: Generator, z_dim: int, label_dim: int, 
                   device: torch.device, num_samples: int = 64,
                   class_idx: Optional[int] = None, seed: int = 42) -> torch.Tensor:
    """
    Generate samples from the trained generator.
    
    Args:
        generator: Trained generator model
        z_dim: Dimension of noise vector
        label_dim: Number of classes
        device: Device to run on
        num_samples: Number of samples to generate
        class_idx: Specific class to generate (None for random)
        seed: Random seed
        
    Returns:
        Generated images tensor
    """
    generator.eval()
    
    with torch.no_grad():
        # Set seed for reproducible generation
        torch.manual_seed(seed)
        
        if class_idx is not None:
            # Generate samples for specific class
            labels = torch.zeros(num_samples, label_dim, device=device)
            labels[:, class_idx] = 1
        else:
            # Generate random samples
            labels = torch.zeros(num_samples, label_dim, device=device)
            random_classes = torch.randint(0, label_dim, (num_samples,), device=device)
            labels.scatter_(1, random_classes.unsqueeze(1), 1)
        
        # Generate noise
        z = torch.randn(num_samples, z_dim, device=device)
        
        # Generate images
        fake_images = generator(z, labels)
        
    return fake_images


def generate_class_grid(generator: Generator, z_dim: int, label_dim: int,
                       device: torch.device, samples_per_class: int = 8,
                       seed: int = 42) -> torch.Tensor:
    """
    Generate a grid of samples for each class.
    
    Args:
        generator: Trained generator model
        z_dim: Dimension of noise vector
        label_dim: Number of classes
        device: Device to run on
        samples_per_class: Number of samples per class
        seed: Random seed
        
    Returns:
        Grid of generated images
    """
    generator.eval()
    
    with torch.no_grad():
        # Set seed for reproducible generation
        torch.manual_seed(seed)
        
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
        
    return all_samples


def interpolate_between_classes(generator: Generator, z_dim: int, label_dim: int,
                               device: torch.device, class1: int, class2: int,
                               num_steps: int = 10, seed: int = 42) -> torch.Tensor:
    """
    Interpolate between two classes.
    
    Args:
        generator: Trained generator model
        z_dim: Dimension of noise vector
        label_dim: Number of classes
        device: Device to run on
        class1: First class
        class2: Second class
        num_steps: Number of interpolation steps
        seed: Random seed
        
    Returns:
        Interpolated images
    """
    generator.eval()
    
    with torch.no_grad():
        # Set seed for reproducible generation
        torch.manual_seed(seed)
        
        # Generate noise (same for all interpolations)
        z = torch.randn(1, z_dim, device=device)
        
        interpolated_images = []
        
        for step in range(num_steps):
            # Interpolate between class labels
            alpha = step / (num_steps - 1)
            
            label = torch.zeros(1, label_dim, device=device)
            label[0, class1] = 1 - alpha
            label[0, class2] = alpha
            
            # Generate image
            fake_image = generator(z, label)
            interpolated_images.append(fake_image)
        
        # Concatenate all interpolated images
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
    return interpolated_images


def save_samples(images: torch.Tensor, save_path: str, title: str = "Generated Samples") -> None:
    """
    Save generated samples as images.
    
    Args:
        images: Generated images tensor
        save_path: Path to save images
        title: Title for the plot
    """
    # Create grid
    grid_img = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    
    # Convert to numpy for matplotlib
    grid_np = grid_img.cpu().permute(1, 2, 0).numpy()
    grid_np = (grid_np + 1) / 2  # Denormalize to [0, 1]
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function for sampling."""
    parser = argparse.ArgumentParser(description='Generate samples from trained Conditional GAN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--z_dim', type=int, default=100, help='Noise dimension')
    parser.add_argument('--label_dim', type=int, default=10, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--class_idx', type=int, default=None, help='Specific class to generate')
    parser.add_argument('--samples_per_class', type=int, default=8, help='Samples per class for grid')
    parser.add_argument('--save_dir', type=str, default='./assets/samples', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--interpolate', action='store_true', help='Interpolate between classes')
    parser.add_argument('--class1', type=int, default=0, help='First class for interpolation')
    parser.add_argument('--class2', type=int, default=1, help='Second class for interpolation')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load generator from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator = Generator(z_dim=args.z_dim, label_dim=args.label_dim, img_size=args.img_size)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator = generator.to(device)
    
    print("Conditional GAN Sampling")
    print("=" * 30)
    
    if args.interpolate:
        # Interpolate between classes
        print(f"Interpolating between class {args.class1} and class {args.class2}")
        interpolated_images = interpolate_between_classes(
            generator, args.z_dim, args.label_dim, device,
            args.class1, args.class2, seed=args.seed
        )
        
        save_path = os.path.join(args.save_dir, f'interpolation_{args.class1}_to_{args.class2}.png')
        save_samples(interpolated_images, save_path, 
                    f'Interpolation: Class {args.class1} → Class {args.class2}')
    
    elif args.class_idx is not None:
        # Generate samples for specific class
        print(f"Generating {args.num_samples} samples for class {args.class_idx}")
        samples = generate_samples(
            generator, args.z_dim, args.label_dim, device,
            args.num_samples, args.class_idx, args.seed
        )
        
        save_path = os.path.join(args.save_dir, f'class_{args.class_idx}_samples.png')
        save_samples(samples, save_path, f'Generated Samples - Class {args.class_idx}')
    
    else:
        # Generate class grid
        print(f"Generating class grid with {args.samples_per_class} samples per class")
        class_grid = generate_class_grid(
            generator, args.z_dim, args.label_dim, device,
            args.samples_per_class, args.seed
        )
        
        save_path = os.path.join(args.save_dir, 'class_grid.png')
        save_samples(class_grid, save_path, 'Generated Samples - All Classes')
        
        # Also generate random samples
        print(f"Generating {args.num_samples} random samples")
        random_samples = generate_samples(
            generator, args.z_dim, args.label_dim, device,
            args.num_samples, None, args.seed
        )
        
        save_path = os.path.join(args.save_dir, 'random_samples.png')
        save_samples(random_samples, save_path, 'Generated Samples - Random Classes')
    
    print(f"Samples saved to {args.save_dir}")


if __name__ == "__main__":
    main()
