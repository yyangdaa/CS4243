import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------
# 1. Hyperparameters
# ---------------------------
batch_size = 64
image_size = 64
latent_dim = 100
num_epochs = 10
learning_rate = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Custom Dataset Loader (For ./data/train/*.png)
# ---------------------------
class CaptchaDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format
        if self.transform:
            image = self.transform(image)
        return image

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
])

# Create dataset and dataloader
dataset = CaptchaDataset(image_folder="./data/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# 3. Define Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ---------------------------
# 4. Define Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# ---------------------------
# 5. Initialize Models, Loss, Optimizers
# ---------------------------
netG = Generator(latent_dim).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

# ---------------------------
# 6. Training Loop
# ---------------------------
for epoch in range(num_epochs):
    for i, images in enumerate(dataloader):
        real_imgs = images.to(device)
        b_size = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(b_size, 1, device=device)
        fake_labels = torch.zeros(b_size, 1, device=device)

        # ---- Train Discriminator ----
        netD.zero_grad()
        
        # Real images
        output_real = netD(real_imgs).view(-1, 1)
        lossD_real = criterion(output_real, real_labels)

        # Fake images
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_imgs = netG(noise)
        output_fake = netD(fake_imgs.detach()).view(-1, 1)
        lossD_fake = criterion(output_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ---- Train Generator ----
        netG.zero_grad()
        
        # Try to fool the discriminator
        output_fake_for_G = netD(fake_imgs).view(-1, 1)
        lossG = criterion(output_fake_for_G, real_labels)
        lossG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

# ---------------------------
# 7. Generate & Display Captcha-Like Images
# ---------------------------
with torch.no_grad():
    sample_imgs = netG(fixed_noise).cpu()
sample_imgs = 0.5 * sample_imgs + 0.5  # Rescale from [-1,1] to [0,1]

grid = torchvision.utils.make_grid(sample_imgs, nrow=4)
plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.title("Generated Captcha Images")
plt.axis("off")
plt.show()
