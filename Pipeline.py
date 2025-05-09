import gradio as gr
import numpy as np
from PIL import Image
import torch
from Super_Resolution import SuperResolution
from Colorization import Colorizer
import torchvision.transforms as transforms
from Denoising_model import DenoiseAutoencoder  # Make sure this import works!

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models (use .pth or .pt for PyTorch)
denoiser = DenoiseAutoencoder().to(device)
denoiser.load_state_dict(torch.load('/home/ccx4276/Lung-Tumor-Detection/models/dae_model.pth', map_location=device))
denoiser.eval()
sr_model = SuperResolution()
colorizer = Colorizer()

# Preprocessing for PyTorch model
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def restore_image(input_img):
    try:
        # Save original mode
        orig_mode = input_img.mode
        if orig_mode == 'L':
            input_img = input_img.convert('RGB')
        
        # Denoising (PyTorch)
        input_tensor = transform(input_img).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_tensor = denoiser(input_tensor)
        denoised_tensor = denoised_tensor.squeeze(0).cpu()
        denoised_img = transforms.ToPILImage()(denoised_tensor)
        
        # Colorization (only if original was grayscale)
        if orig_mode == 'L':
            colorized_img = colorizer.colorize(denoised_img)
        else:
            colorized_img = denoised_img
        
        # Super-resolution
        super_res_img = sr_model.enhance(colorized_img)
        
        # Final processed image (same as super_res_img in this pipeline)
        final_img = super_res_img
        
        return denoised_img, colorized_img, super_res_img, final_img
    
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

iface = gr.Interface(
    fn=restore_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Denoised Image"),
        gr.Image(type="pil", label="Colorized Image"),
        gr.Image(type="pil", label="Super-Resolved Image"),
        gr.Image(type="pil", label="Final Processed Image")
    ],
    title="AI Image Restoration",
    description="Restore images with noise removal, super-resolution, and colorization",
    examples=[["/home/ccx4276/GENAI/test.png"]]
)

if __name__ == "__main__":
    iface.launch(share=True)
