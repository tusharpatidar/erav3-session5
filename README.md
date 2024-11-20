# MNIST Digit Classification

# Lightweight CNN for Digit Recognition

A PyTorch implementation of a memory-efficient convolutional neural network for MNIST digit classification. Achieves >95% accuracy in single epoch with <25K parameters.

## Key Features

- Efficient CNN architecture
- Single epoch training
- Automated testing pipeline
- Data augmentation support
- GPU acceleration support

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tusharpatidar/erav3-session5.git
cd erav3-session5
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── network.py        # Neural network architecture
│   ├── train.py          # Training pipeline
│   ├── augmentation.py   # Data transformation utilities
│   ├── test_network.py   # Model validation and testing
├── data/                 # Dataset directory (auto-created)
├── requirements.txt      # Project dependencies
└── README.md            # Documentation
```

## Usage

### Training

To train the model:
```bash
python src/train.py
```

The model will train until it reaches 95% accuracy or completes one epoch.

### Testing

To run the test suite:
```bash
pytest src/test_network.py -v
```

This will verify:
- Model architecture constraints (<25K parameters)
- Performance requirements (>95% accuracy)
- Input/output behavior
- Training/inference consistency
- Noise robustness
- Gradient flow characteristics

The test suite includes:
1. Architecture Tests
   - Parameter count verification
   - Output shape validation
   - Batch size handling

2. Performance Tests
   - Accuracy measurement
   - Training behavior
   - Inference consistency

3. Robustness Tests
   - Input noise tolerance
   - Gradient stability checks
   - Numerical stability

4. Training Tests
   - Gradient flow analysis
   - Training/eval mode behavior
   - Loss computation verification

## Development

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run the test suite
5. Commit changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request


