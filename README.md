
# AI-Powered Image Restoration

A unified pipeline for noise removal, super-resolution, and colorization of degraded images.

The output is as displayed below: 


<img width="1512" alt="Screenshot 2025-05-11 at 11 28 51 PM" src="https://github.com/user-attachments/assets/b4d17021-4465-445a-ac57-109fcfcbb19a" />
<img width="1512" alt="Screenshot 2025-05-11 at 11 47 59 PM" src="https://github.com/user-attachments/assets/bb822cab-0d37-491d-9441-0387f0f5505c" />

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
    │   │   
    │   └── noisy/
    └── val/
        └── ...
```

Training data should be organized with matching pairs of clean/noisy images for supervised learning of the denoising model.


## 🏗️ Project Structure

```text
AI-Image-Restoration/
├── README.md              # Project documentation
├── DataPreprocessing.py   # Data preparation utilities
├── Denoising_model.py     # Denoising autoencoder and training script
├── Super_Resolution.py    # Super-resolution wrapper
├── Colorization.py        # DeOldify colorization wrapper
├── Pipeline.py            # Image restoration pipeline and Gradio UI
├── dae_model.h5           # Included model artifact
└── output.log             # Included execution log
```

The Python modules are in the repository root. Training datasets and the
PyTorch `.pth` checkpoint referenced by the scripts are external inputs;
they are not included here. A `requirements.txt` file is not currently provided.


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

## 📊 Current Results & Limitations
The pipeline is currently in development and generates the following outputs:
**Denoising Output**

Preliminary noise reduction with some remaining artifacts
Performance varies significantly based on input image quality


**Super-Resolution Output**

Increased image resolution (4x) with some detail enhancement
May introduce artifacts in complex texture regions
Some edge cases remain challenging (e.g., text, fine patterns)

**Colorization Output**

Basic color added to grayscale photographs
Color accuracy needs improvement, especially for unusual scenes
Results are better for common objects and standard scenes

**Pipeline Integration Issues**

The sequential application of models can sometimes compound errors
Performance bottlenecks when processing high-resolution images
Occasional GPU memory issues with larger images

## Planned Improvements

Fine-tuning the denoising model with more diverse training data
Implementing an improved loss function for the super-resolution component
Exploring ensemble approaches for more robust colorization
Optimizing the pipeline for better integration between components


