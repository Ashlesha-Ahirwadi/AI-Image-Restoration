import os
os.environ["GRADIO_TEMP_DIR"] = "/home/ccx4276/gradio_tmp"
import numpy as np
from PIL import Image
import torch
from Super_Resolution import SuperResolution
from Colorization import fake_colorize
import torchvision.transforms as transforms
from Denoising_model import DenoiseAutoencoder  
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import gradio as gr
import os

print("Starting pipeline...")
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device set")
# Load models (use .pth or .pt for PyTorch)
denoiser = DenoiseAutoencoder().to(device)
print("Denoiser model created")
denoiser.load_state_dict(torch.load('/home/ccx4276/Lung-Tumor-Detection/models/dae_model.pth', map_location=device))
print("Denoiser weights loaded")
denoiser.eval()
sr_model = SuperResolution()
print("SuperResolution model created")

# Preprocessing for PyTorch model
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def is_visually_gray(img):
    arr = np.array(img)
    if arr.ndim == 2:  # Already grayscale
        return True
    if arr.ndim == 3 and arr.shape[2] == 3:
        # If all channels are (almost) equal, it's gray
        return np.allclose(arr[..., 0], arr[..., 1]) and np.allclose(arr[..., 1], arr[..., 2])
    return False

def restore_image(input_img):
    try:
        print("In restore_image")
        orig_mode = input_img.mode
        if orig_mode in ['L', 'RGBA']:
            input_img = input_img.convert('RGB')
        print("Starting denoising")
        input_tensor = transform(input_img).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_tensor = denoiser(input_tensor)
        denoised_tensor = denoised_tensor.squeeze(0).cpu()
        denoised_img = transforms.ToPILImage()(denoised_tensor)
        print("Denoising done")
        # Use fake colorization for grayscale images
        if orig_mode == 'L' or is_visually_gray(denoised_img):
            print("Starting fake colorization")
            colorized_img = fake_colorize(denoised_img.convert('L'))
            print("Fake colorization done")
        else:
            colorized_img = denoised_img
        print("Starting super-resolution")
        super_res_img = sr_model.enhance(colorized_img)
        print("Super-resolution done")
        final_img = super_res_img
        return denoised_img, colorized_img, super_res_img, final_img
    except Exception as e:
        print(f"Exception in restore_image: {e}")
        return None, None, None, f"Error: {str(e)}"

def run_pipeline_on_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.JPEG")]
    )
    if not file_path:
        return
    try:
        img = Image.open(file_path)
        denoised, colorized, super_res, final = restore_image(img)
        # Show results in GUI
        for i, (title, im) in enumerate([
            ("Denoised", denoised),
            ("Colorized", colorized),
            ("Super-Res", super_res),
            ("Final", final)
        ]):
            if im and isinstance(im, Image.Image):
                im_disp = im.resize((256, 256))
                imtk = ImageTk.PhotoImage(im_disp)
                panels[i].config(image=imtk)
                panels[i].image = imtk
                panels[i].config(text=title)
            else:
                panels[i].config(image='', text=f"{title}\n(Error)")
        # Save outputs
        def save_outputs():
            out_prefix = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if out_prefix:
                if denoised: denoised.save(f"{out_prefix}_denoised.png")
                if colorized: colorized.save(f"{out_prefix}_colorized.png")
                if super_res: super_res.save(f"{out_prefix}_superres.png")
                if final and isinstance(final, Image.Image): final.save(f"{out_prefix}_final.png")
                messagebox.showinfo("Saved", "Images saved!")
        save_btn.config(command=save_outputs)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image:\n{e}")

def gradio_restore_image(input_img):
    # input_img is a numpy array (from Gradio)
    pil_img = Image.fromarray(input_img)
    denoised, colorized, super_res, final = restore_image(pil_img)
    # Convert outputs to numpy arrays for Gradio
    outs = []
    for im in [denoised, colorized, super_res, final]:
        if im and isinstance(im, Image.Image):
            outs.append(np.array(im))
        else:
            outs.append(None)
    return tuple(outs)

if __name__ == "__main__":
    print("Script started")
    input_path = "val_178.JPEG"
    output_prefix = "test"
    try:
        print("Opening image...")
        img = Image.open(input_path)
        print("Calling restore_image...")
        denoised, colorized, super_res, final = restore_image(img)
        print("restore_image finished")
        if denoised:
            denoised.save(f"{output_prefix}_denoised.png")
        if colorized:
            colorized.save(f"{output_prefix}_colorized.png")
        if super_res:
            super_res.save(f"{output_prefix}_superres.png")
        if final and isinstance(final, Image.Image):
            final.save(f"{output_prefix}_final.png")
        elif isinstance(final, str):
            print(f"Error: {final}")
        print("Processing complete.")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

    # Add Gradio GUI with example
    example_path = os.path.abspath(input_path)
    gr.Interface(
        fn=gradio_restore_image,
        inputs=gr.Image(type="numpy", label="Input Image"),
        outputs=[
            gr.Image(type="numpy", label="Denoised"),
            gr.Image(type="numpy", label="Colorized"),
            gr.Image(type="numpy", label="Super-Res"),
            gr.Image(type="numpy", label="Final"),
        ],
        title="Image Restoration Pipeline",
        description="Upload an image to denoise, colorize, and super-resolve it.",
        examples=[[example_path]]
    ).launch(server_name="0.0.0.0", server_port=7860)
