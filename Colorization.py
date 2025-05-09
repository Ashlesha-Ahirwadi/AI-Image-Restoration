from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer
from PIL import Image
import numpy as np

class Colorizer:
    def __init__(self):
        device.set(device=DeviceId.GPU0)
        self.colorizer = get_image_colorizer(artistic=True)
    
    def colorize(self, image):
        if isinstance(image, str):
            return self.colorizer.get_transformed_image(image, render_factor=35)
        else:
            # Convert PIL Image to file-like object
            image = image.convert('L')  # Ensure grayscale
            return self.colorizer.get_transformed_image_from_array(np.array(image))
