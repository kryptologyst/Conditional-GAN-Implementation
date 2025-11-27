"""
Evaluation script for Conditional GAN.
"""

import torch
import argparse
import os
import json
from torch.utils.data import DataLoader

from src.models.cgan import Generator
from src.data.mnist import load_mnist_data
from src.evaluation.metrics import evaluate_model
from src.utils.training import get_device


def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained Conditional GAN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--z_dim', type=int, default=100, help='Noise dimension')
    parser.add_argument('--label_dim', type=int, default=10, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=28, help='Image size')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./assets/evaluation', help='Save directory')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load generator from checkpoint
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator = Generator(z_dim=args.z_dim, label_dim=args.label_dim, img_size=args.img_size)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator = generator.to(device)
    
    # Load real data
    print("Loading real data...")
    real_loader = load_mnist_data(batch_size=64, data_dir=args.data_dir, train=False)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(
        generator, real_loader, device,
        z_dim=args.z_dim, label_dim=args.label_dim,
        num_samples=args.num_samples
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 40)
    
    if 'inception_score' in results:
        is_score = results['inception_score']
        print(f"Inception Score: {is_score['mean']:.4f} ± {is_score['std']:.4f}")
    
    if 'fid_score' in results:
        print(f"FID Score: {results['fid_score']:.4f}")
    
    if 'precision' in results and 'recall' in results:
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
    
    # Save results
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
