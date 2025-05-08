
# AI-Powered Image Restoration

A unified pipeline for noise removal, super-resolution, and colorization of degraded images.


## 🔍 Overview

This project combines deep learning techniques to create a comprehensive image restoration pipeline that:
- Removes noise and artifacts from damaged images
- Enhances resolution of low-quality images
- Adds realistic color to black and white photographs

The system integrates three specialized models into a seamless workflow, accessible through an intuitive GUI built with Gradio.

## ✨ Features

- **Denoising**: Convolutional autoencoder that removes noise, scratches, and artifacts
- **Super-Resolution**: Real-ESRGAN model that upscales images by 4x with enhanced details
- **Colorization**: DeOldify model that adds realistic colors to grayscale images
- **User-Friendly Interface**: Interactive web application for easy image processing
- **Modular Architecture**: Each component can be used independently or as part of the pipeline


## 📊 Dataset Structure

The project uses the following dataset structure:

```
data/
├── raw/            # Original unprocessed data
└── processed/      # Preprocessed data
    ├── train/
    │   ├── clean/
    │   │   ├── clean_0.png
    │   │   └── noisy_0.png
    │   └── noisy/
    └── val/
        └── ...
```

Training data should be organized with matching pairs of clean/noisy images for supervised learning of the denoising model.


## 🏗️ Project Structure

```
project/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── data/                  # Dataset directory
├── models/                # Pre-trained model weights
├── example_images/        # Sample test images
├── src/                   # Source code
│   ├── DataPreprocessing.py  # Data preparation utilities
│   ├── Denoising_model.py    # Denoising model implementation
│   ├── Super_Resolution.py   # Super-resolution model wrapper
│   ├── Colorization.py       # Colorization model wrapper
│   └── Pipeline.py           # Unified pipeline with Gradio UI
└── logs/                  # Training logs
```

## 🧠 Model Architecture

### Denoising Autoencoder
- **Architecture**: Convolutional autoencoder
- **Framework**: PyTorch
- **Details**: Features an encoder with two convolutional layers that reduce spatial dimensions while increasing feature channels, and a decoder with transposed convolutions that restore the original resolution.

### Super-Resolution
- **Architecture**: Real-ESRGAN
- **Framework**: PyTorch (via torch.hub)
- **Details**: Pre-trained model that upscales images by 4x while maintaining realistic textures and details.

### Colorization
- **Architecture**: DeOldify (Conditional GAN)
- **Framework**: PyTorch
- **Details**: Artistic colorization model trained on historical photographs to generate realistic colorization.
