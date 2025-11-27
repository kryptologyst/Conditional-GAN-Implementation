"""
Streamlit demo for Conditional GAN image generation.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import io
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cgan import Generator
from utils.training import set_seed, get_device


@st.cache_resource
def load_model(checkpoint_path: str, z_dim: int = 100, label_dim: int = 10, img_size: int = 28):
    """Load the trained generator model."""
    device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize generator
    generator = Generator(z_dim=z_dim, label_dim=label_dim, img_size=img_size)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator = generator.to(device)
    generator.eval()
    
    return generator, device


def generate_images(generator, device, z_dim, label_dim, num_samples, class_idx=None, seed=42):
    """Generate images using the trained generator."""
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


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:
        # Batch of images
        images = []
        for i in range(tensor.size(0)):
            img_np = tensor[i].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np.squeeze(), mode='L'))
        return images
    else:
        # Single image
        img_np = tensor.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np.squeeze(), mode='L')


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Conditional GAN Demo",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Conditional GAN Image Generation")
    st.markdown("Generate MNIST digits conditioned on class labels using a trained Conditional GAN.")
    
    # Sidebar for controls
    st.sidebar.header("Generation Controls")
    
    # Model selection
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path", 
        value="./assets/final_model.pth",
        help="Path to the trained model checkpoint"
    )
    
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint file not found: {checkpoint_path}")
        st.info("Please train a model first using the training script.")
        return
    
    # Generation parameters
    num_samples = st.sidebar.slider("Number of Samples", 1, 64, 16)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    
    # Class selection
    generation_mode = st.sidebar.selectbox(
        "Generation Mode",
        ["Random Classes", "Specific Class", "All Classes Grid"]
    )
    
    class_idx = None
    if generation_mode == "Specific Class":
        class_idx = st.sidebar.selectbox("Select Class", list(range(10)))
    
    # Load model
    try:
        generator, device = load_model(checkpoint_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Generate button
    if st.sidebar.button("🎲 Generate Images", type="primary"):
        with st.spinner("Generating images..."):
            # Generate images
            if generation_mode == "All Classes Grid":
                # Generate grid for all classes
                samples_per_class = max(1, num_samples // 10)
                all_images = []
                
                for class_idx in range(10):
                    class_images = generate_images(
                        generator, device, 100, 10, samples_per_class, class_idx, seed
                    )
                    all_images.append(class_images)
                
                # Concatenate all images
                all_images = torch.cat(all_images, dim=0)
                
                # Create grid
                grid_img = torchvision.utils.make_grid(all_images, nrow=samples_per_class, normalize=True)
                
                # Convert to PIL
                grid_pil = tensor_to_pil(grid_img)
                
                st.subheader("Generated Images - All Classes")
                st.image(grid_pil, caption="Generated samples for all classes (0-9)", use_column_width=True)
                
            else:
                # Generate regular samples
                images = generate_images(
                    generator, device, 100, 10, num_samples, class_idx, seed
                )
                
                # Convert to PIL images
                pil_images = tensor_to_pil(images)
                
                if generation_mode == "Random Classes":
                    st.subheader("Generated Images - Random Classes")
                    caption = f"Generated {num_samples} random samples"
                else:
                    st.subheader(f"Generated Images - Class {class_idx}")
                    caption = f"Generated {num_samples} samples for class {class_idx}"
                
                # Display images in a grid
                cols = st.columns(4)
                for i, img in enumerate(pil_images):
                    col_idx = i % 4
                    cols[col_idx].image(img, caption=f"Sample {i+1}", use_column_width=True)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info("""
    **Model**: Conditional GAN  
    **Dataset**: MNIST  
    **Classes**: 0-9 (digits)  
    **Image Size**: 28x28 pixels  
    **Architecture**: Fully Connected
    """)
    
    # Instructions
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. **Load Model**: Make sure you have a trained model checkpoint
    2. **Select Parameters**: Choose number of samples and random seed
    3. **Choose Mode**: 
       - **Random Classes**: Generate images with random class labels
       - **Specific Class**: Generate images for a chosen digit (0-9)
       - **All Classes Grid**: Generate a grid showing all classes
    4. **Generate**: Click the generate button to create new images
    """)
    
    # Technical details
    with st.expander("Technical Details"):
        st.markdown("""
        **Architecture**:
        - Generator: 4-layer fully connected network with batch normalization
        - Discriminator: 4-layer fully connected network with batch normalization
        - Loss: Binary Cross-Entropy with Logits
        - Optimizer: Adam (β₁=0.5, β₂=0.999)
        
        **Training**:
        - Dataset: MNIST (60,000 training images)
        - Batch Size: 64
        - Learning Rate: 0.0002
        - Epochs: 50
        - Device: CUDA/MPS/CPU (auto-detected)
        """)


if __name__ == "__main__":
    main()
