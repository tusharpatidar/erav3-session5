import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from augmentation import get_base_transforms

SAVE_PATH = 'digit_classifier.pth'

def build_model():
    """Initialize the neural network"""
    from network import DigitClassifier
    return DigitClassifier()

def train(epochs=1, batch_size=64, learning_rate=1e-3):
    """Train the digit classifier"""
    print("\n🚀 Initializing Training Pipeline")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    transform = get_base_transforms()
    
    # Data loading
    train_data = MNIST('./data', train=True, download=True, transform=transform)
    val_data = MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        correct = total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f"\n📊 Validation Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), SAVE_PATH)
            
        if accuracy >= 95:
            print(f"🎯 Target accuracy achieved!")
            break
    
    return model

if __name__ == "__main__":
    train() 