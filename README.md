# Conditional GAN Implementation

A production-ready implementation of Conditional Generative Adversarial Networks (cGAN) for MNIST digit generation.

## Overview

This project implements a Conditional GAN that generates MNIST digits conditioned on class labels. The generator takes both random noise and a class label as input to produce specific digit images, while the discriminator learns to distinguish between real and fake images for each class.

## Features

- **Modern Architecture**: Fully connected networks with batch normalization and dropout
- **Device Support**: Automatic CUDA/MPS/CPU detection and usage
- **Reproducible**: Deterministic seeding for consistent results
- **Modular Design**: Clean separation of models, data, and utilities
- **Interactive Demo**: Streamlit web app for easy image generation
- **Comprehensive Evaluation**: Training curves and sample visualization
- **Production Ready**: Proper configuration management and logging

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Conditional-GAN-Implementation.git
cd Conditional-GAN-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train the Conditional GAN on MNIST:

```bash
python scripts/train_cgan.py --epochs 50 --batch_size 64
```

### Sampling

Generate samples from the trained model:

```bash
# Generate random samples
python scripts/sample_cgan.py --checkpoint ./assets/final_model.pth

# Generate samples for specific class
python scripts/sample_cgan.py --checkpoint ./assets/final_model.pth --class_idx 5

# Generate class grid
python scripts/sample_cgan.py --checkpoint ./assets/final_model.pth --samples_per_class 8
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/
│   ├── models/          # Model definitions
│   ├── data/            # Data loading utilities
│   ├── utils/           # Training utilities
│   └── evaluation/      # Evaluation metrics
├── scripts/
│   ├── train_cgan.py    # Training script
│   └── sample_cgan.py   # Sampling script
├── configs/
│   └── config.yaml      # Configuration file
├── demo/
│   └── app.py           # Streamlit demo
├── tests/               # Unit tests
├── assets/              # Generated outputs
└── requirements.txt     # Dependencies
```

## Model Architecture

### Generator
- Input: Noise vector (100D) + One-hot class label (10D)
- Architecture: 4-layer fully connected network
- Hidden dimensions: 256 → 512 → 1024 → 784
- Activations: ReLU + BatchNorm + Dropout
- Output: Tanh activation for [-1, 1] range

### Discriminator
- Input: Flattened image (784D) + One-hot class label (10D)
- Architecture: 4-layer fully connected network
- Hidden dimensions: 1024 → 512 → 256 → 1
- Activations: LeakyReLU + BatchNorm + Dropout
- Output: Raw logits (no activation)

## Training Details

- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam (β₁=0.5, β₂=0.999)
- **Learning Rate**: 0.0002
- **Batch Size**: 64
- **Epochs**: 50
- **Device**: Auto-detected (CUDA > MPS > CPU)

## Configuration

The training can be customized using the configuration file `configs/config.yaml`:

```yaml
model:
  z_dim: 100
  label_dim: 10
  img_size: 28
  hidden_dim: 256

training:
  epochs: 50
  batch_size: 64
  lr: 0.0002
```

## Evaluation

The model generates high-quality MNIST digits conditioned on class labels. Key metrics:

- **Visual Quality**: Generated digits are recognizable and diverse
- **Class Conditioning**: Generated images match the specified class labels
- **Training Stability**: Smooth loss curves without mode collapse

## Usage Examples

### Generate Specific Digits

```python
from src.models.cgan import Generator
import torch

# Load trained model
generator = Generator()
generator.load_state_dict(torch.load('final_model.pth'))

# Generate digit '5'
z = torch.randn(1, 100)
label = torch.zeros(1, 10)
label[0, 5] = 1  # Class 5

generated_image = generator(z, label)
```

### Batch Generation

```python
# Generate 16 samples of digit '3'
z = torch.randn(16, 100)
labels = torch.zeros(16, 10)
labels[:, 3] = 1  # All samples are class 3

generated_images = generator(z, labels)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Model Card

### Intended Use
This model is designed for educational and research purposes to demonstrate conditional image generation using GANs.

### Training Data
- **Dataset**: MNIST handwritten digits
- **Size**: 60,000 training images
- **Classes**: 10 digit classes (0-9)
- **Resolution**: 28×28 grayscale images
- **License**: Public domain

### Limitations
- Limited to MNIST digit generation
- May not generalize to other datasets
- Generated images may have artifacts
- Training can be unstable without proper hyperparameter tuning

### Bias Considerations
- Model inherits any biases present in MNIST dataset
- Generated digits may not represent all handwriting styles
- No explicit bias mitigation implemented

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original GAN paper: Goodfellow et al. (2014)
- Conditional GAN paper: Mirza & Osindero (2014)
- MNIST dataset: LeCun et al. (1998)
- PyTorch framework and community
# Conditional-GAN-Implementation
