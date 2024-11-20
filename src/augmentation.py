from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from PIL import Image

def get_base_transforms():
    """Basic transformation pipeline for MNIST digits"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_augmented_transforms(angle):
    """Transformation pipeline with rotation augmentation"""
    return transforms.Compose([
        transforms.RandomRotation([angle, angle]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def visualize_augmentations():
    """Visualize different augmentations on a sample digit"""
    dataset = MNIST('./data', train=True, download=True, transform=None)
    sample_digit, _ = dataset[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Digit Augmentation Examples', fontsize=16)
    
    # Display original
    axes[0][0].imshow(sample_digit, cmap='gray')
    axes[0][0].set_title('Original')
    axes[0][0].axis('off')
    
    # Display augmented versions
    angles = [15, 30, 45, -30, -15]
    positions = [(0,1), (0,2), (1,0), (1,1), (1,2)]
    
    for angle, (row, col) in zip(angles, positions):
        transform = get_augmented_transforms(angle)
        augmented = transform(sample_digit)
        axes[row][col].imshow(augmented.squeeze(), cmap='gray')
        axes[row][col].set_title(f'{angle}° Rotation')
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_augmentations() 