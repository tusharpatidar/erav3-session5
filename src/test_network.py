import pytest
import torch
import numpy as np
from pathlib import Path
from train import build_model, SAVE_PATH, train
from augmentation import get_base_transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

@pytest.fixture(scope="session")
def device():
    """Determine and provide the available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(scope="session")
def digit_classifier(device):
    """Initialize or load the pre-trained classifier"""
    print(f"\nUsing device: {device}")
    
    if Path(SAVE_PATH).exists():
        print("Loading pre-trained model...")
        model = build_model()
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    else:
        print("Training new model...")
        model = train()
    
    return model.to(device)

def test_model_size(digit_classifier):
    """Test if model meets size constraints"""
    param_count = sum(p.numel() for p in digit_classifier.parameters())
    print(f"\nModel parameters: {param_count:,}")
    
    assert param_count < 25000, (
        f"Model exceeds size limit: {param_count:,} parameters (max: 25,000)"
    )

def test_model_accuracy(digit_classifier, device):
    """Test if model meets accuracy requirements"""
    digit_classifier.eval()
    
    # Prepare test data
    transform = get_base_transforms()
    test_dataset = MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = digit_classifier(images)
            predictions = outputs.argmax(dim=1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nTest accuracy: {accuracy:.2f}%")
    
    assert accuracy >= 95, f"Model accuracy ({accuracy:.2f}%) below 95% threshold"

def test_output_format(digit_classifier, device):
    """Test model output dimensions and values"""
    digit_classifier.eval()
    
    # Test different batch sizes
    test_batches = [(1, "single sample"), 
                    (32, "small batch"), 
                    (64, "medium batch")]
    
    for batch_size, desc in test_batches:
        # Generate random test input
        test_input = torch.randn(batch_size, 1, 28, 28).to(device)
        
        with torch.no_grad():
            output = digit_classifier(test_input)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
        
        # Check output shape
        expected_shape = (batch_size, 10)
        assert output.shape == expected_shape, (
            f"Wrong output shape for {desc}: "
            f"got {output.shape}, expected {expected_shape}"
        )
        
        # Check if outputs are valid probabilities after softmax
        assert torch.allclose(probs.sum(dim=1), 
                            torch.ones(batch_size).to(device), 
                            rtol=1e-3), (
            f"Probabilities for {desc} don't sum to 1"
        )

def test_training_behavior(digit_classifier, device):
    """Test model's training vs evaluation behavior"""
    # Test input
    test_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Test basic training mode
    digit_classifier.train()
    train_output = digit_classifier(test_input)
    assert train_output.requires_grad, "Training mode should enable gradients"
    
    # Test evaluation mode
    digit_classifier.eval()
    with torch.no_grad():
        eval_output1 = digit_classifier(test_input)
        eval_output2 = digit_classifier(test_input)
        
        # Check deterministic behavior
        assert torch.allclose(eval_output1, eval_output2, rtol=1e-5), (
            "Model should be deterministic in eval mode"
        )
        
        # Check no gradients in eval mode
        assert not eval_output1.requires_grad, "Eval mode should disable gradients"

def test_noise_robustness(digit_classifier, device):
    """Test model's robustness to input noise"""
    digit_classifier.eval()
    
    # Get a real MNIST sample
    transform = get_base_transforms()
    test_dataset = MNIST('./data', train=False, download=True, transform=transform)
    sample_image, true_label = test_dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Test with different noise levels
    noise_levels = [0.1, 0.2, 0.3]
    original_prediction = digit_classifier(sample_image).argmax(dim=1).item()
    
    print(f"\nTesting noise robustness:")
    print(f"Original prediction: {original_prediction}")
    
    for noise_level in noise_levels:
        # Add Gaussian noise
        noise = torch.randn_like(sample_image) * noise_level
        noisy_image = sample_image + noise
        
        with torch.no_grad():
            prediction = digit_classifier(noisy_image).argmax(dim=1).item()
        
        print(f"Prediction with {noise_level:.1f} noise: {prediction}")
        
        # Model should be somewhat robust to moderate noise
        if noise_level <= 0.2:  # Only check for lower noise levels
            assert prediction == original_prediction, (
                f"Model prediction changed with noise_level={noise_level}"
            )

def test_gradient_flow(digit_classifier, device):
    """Test if gradients flow properly through the model"""
    digit_classifier.train()
    
    # Create a simple batch
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)
    dummy_labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Forward pass
    outputs = digit_classifier(test_input)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, dummy_labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    gradient_exists = False
    max_gradient = -float('inf')
    min_gradient = float('inf')
    
    for name, param in digit_classifier.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                gradient_exists = True
                max_gradient = max(max_gradient, grad_norm)
                min_gradient = min(min_gradient, grad_norm)
    
    print(f"\nGradient flow check:")
    print(f"Max gradient norm: {max_gradient:.6f}")
    print(f"Min gradient norm: {min_gradient:.6f}")
    
    assert gradient_exists, "No gradients found in model"
    assert max_gradient < 10.0, "Gradient explosion detected"
    assert min_gradient > 1e-6, "Vanishing gradient detected"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 