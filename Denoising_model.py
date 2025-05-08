import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import cv2

# -------------------------------
# Define a simple denoising autoencoder
# -------------------------------

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (B, 3, H, W)
            nn.Sigmoid(),  # To keep output in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -------------------------------
# Train function
# -------------------------------

def train_denoiser(data_dir, epochs=20, batch_size=64, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load datasets
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    print(f"Training data folder: {train_path}")
    print(f"Validation data folder: {val_path}")
    
    # Debug: List the contents of the data directories
    print("Train directory contents:", os.listdir(train_path))
    print("Val directory contents:", os.listdir(val_path))

    train_dataset = ImageFolder(root=train_path, transform=transform)
    val_dataset = ImageFolder(root=val_path, transform=transform)

    print(f"Found {len(train_dataset)} training samples.")
    print(f"Found {len(val_dataset)} validation samples.")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the train folder path.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the val folder path.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            noisy_inputs = inputs + 0.1 * torch.randn_like(inputs)  # Add Gaussian noise
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            inputs = inputs.to(device)
            noisy_inputs = noisy_inputs.to(device)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

        # Save sample output from first batch of validation data
        model.eval()
        with torch.no_grad():
            for val_inputs, _ in val_loader:
                noisy_val = val_inputs + 0.1 * torch.randn_like(val_inputs)
                noisy_val = torch.clamp(noisy_val, 0., 1.)

                val_inputs = val_inputs.to(device)
                noisy_val = noisy_val.to(device)
                recon = model(noisy_val)

                save_dir = os.path.join("outputs", f"epoch_{epoch+1}")
                os.makedirs(save_dir, exist_ok=True)
                save_image(recon.cpu(), os.path.join(save_dir, "reconstruction.png"))
                save_image(noisy_val.cpu(), os.path.join(save_dir, "noisy.png"))
                save_image(val_inputs.cpu(), os.path.join(save_dir, "original.png"))
                break  # Only one batch
    print("Training completed!")

# -------------------------------
# Argument parser
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory (must contain 'train' and 'val')")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_denoiser(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
