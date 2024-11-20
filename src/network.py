import torch.nn as nn

class DigitClassifier(nn.Module):
    """
    A lightweight CNN architecture for digit classification.
    Designed to achieve >95% accuracy with <25K parameters.
    """
    def __init__(self):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x 