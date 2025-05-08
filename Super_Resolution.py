import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

class SuperResolution:
    def __init__(self):
        self.model = torch.hub.load('xinntao/Real-ESRGAN', 'RealESRGAN_x4plus', pretrained=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def enhance(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Convert to tensor
        img_tensor = ToTensor()(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sr_tensor = self.model(img_tensor)
        
        # Convert back to PIL Image
        sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())
        return sr_image