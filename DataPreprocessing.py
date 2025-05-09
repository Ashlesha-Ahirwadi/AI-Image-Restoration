import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Denoising Autoencoder
class DenoiseAutoencoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Dataset Class for Paired Images
class PairedImageDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.image_names = os.listdir(clean_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        clean_path = os.path.join(self.clean_dir, img_name)
        # Extract the index from the clean filename
        index = img_name.split('_')[-1]  # e.g., '320.png'
        noisy_img_name = 'noisy_' + index
        noisy_path = os.path.join(self.noisy_dir, noisy_img_name)
        
        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')
        
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
            
        return noisy_img, clean_img

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# DataLoader
dataset = PairedImageDataset(clean_dir='/home/ccx4276/GENAI/processed/train/clean/', noisy_dir='/home/ccx4276/GENAI/processed/train/noisy/', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

## Initialize model, loss function, optimizer
model = DenoiseAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    for data in dataloader:
        noisy_imgs, clean_imgs = data
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        # Forward pass
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
# Save Model
torch.save(model.state_dict(), '/home/ccx4276/GENAI/model/dae_model.h5')
