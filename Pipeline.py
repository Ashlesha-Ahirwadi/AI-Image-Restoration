import gradio as gr
import numpy as np
from PIL import Image
import torch
from tensorflow.keras.models import load_model
from Super_Resolution import SuperResolution
from GENAI.scripts.Colorization import Colorizer

# Load models
denoiser = load_model('models/denoiser.h5')
sr_model = SuperResolution()
colorizer = Colorizer()

def restore_image(input_img):
    try:
        # Convert to RGB if grayscale
        if input_img.mode == 'L':
            input_img = input_img.convert('RGB')
        
        # Denoising
        input_array = np.array(input_img.resize((128, 128))) / 255.0
        denoised = denoiser.predict(np.expand_dims(input_array, axis=0))[0]
        denoised = (denoised * 255).astype(np.uint8)
        denoised_img = Image.fromarray(denoised)
        
        # Colorization (only if original was grayscale)
        if input_img.mode == 'L':
            denoised_img = colorizer.colorize(denoised_img)
        
        # Super-resolution
        final_img = sr_model.enhance(denoised_img)
        
        return final_img
    
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=restore_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="AI Image Restoration",
    description="Restore images with noise removal, super-resolution, and colorization",
    examples=[["example_images/damaged1.jpg"], ["example_images/old_photo.jpg"]]
)

if __name__ == "__main__":
    iface.launch()