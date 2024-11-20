from setuptools import setup, find_packages

setup(
    name="mnist_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch==2.1.0',
        'torchvision==0.16.0',
        'matplotlib==3.7.1',
        'pytest==7.4.0',
        'numpy==1.24.3',
        'pillow>=9.0.0',
    ],
    python_requires='>=3.8',
    description="A lightweight CNN for MNIST digit classification",
    author="Tushar Patidar",
    author_email="tushar.p.patidar@gmail.com",
) 