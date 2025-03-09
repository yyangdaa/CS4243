import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Hyperparameters
# ---------------------------
batch_size = 64
image_size = 64
latent_dim = 100
num_classes = 10   # Example: digits 0-9
embedding_dim = 10 # Size of label embedding
num_epochs = 2
learning_rate = 0.0002

# ---------------------------
# 2. Data Loading
# ---------------------------
# Suppose your dataset is structured or you have a custom dataset returning (image, label).
# Each label is an integer in [0..9]. Adjust code if you have a different range or multi-character labels.

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# For demonstration, we'll assume you have an ImageFolder with subfolders 0,1,2...9
# that hold images. If not, adjust for a custom dataset that returns an integer label.
dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 3. Conditional Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, embedding_dim=10, feature_g=64):
        super(Generator, self).__init__()
        
        # Embedding for labels
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        # Project + reshape dimension = z_dim + embedding_dim
        self.project = nn.Sequential(
            nn.Linear(z_dim + embedding_dim, feature_g * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Transposed conv blocks
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(feature_g * 8),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_emb(labels)
        
        # Concatenate noise + embedding
        x = torch.cat((noise, label_embedding), dim=1)
        
        # Project and reshape
        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)
        
        # Pass through convolutional generator
        img = self.conv_blocks(x)
        return img

# ---------------------------
# 4. Conditional Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=10, feature_d=64):
        super(Discriminator, self).__init__()
        
        # Embedding for labels
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        # The idea: flatten the embedded label and "project" it to match image feature map size
        # We'll keep it simple and just replicate the label in a conv block or add it as a separate channel
        # For demonstration, let's just feed the combined features to a linear at the end.
        
        # Convolutional layers for the image
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, feature_d, 4, 2, 1, bias=False),
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
        )
        
        # Final classification layer
        # We'll incorporate the label embedding at this stage
        self.adv_layer = nn.Sequential(
            nn.Linear(feature_d * 8 * 4 * 4 + embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Extract image features
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        
        # Embed labels and concatenate
        label_embedding = self.label_emb(labels)
        combined = torch.cat((out, label_embedding), dim=1)
        
        # Classify real/fake
        validity = self.adv_layer(combined)
        return validity

# ---------------------------
# 5. Initialize Models, Loss, Optimizers
# ---------------------------
netG = Generator(latent_dim, num_classes, embedding_dim).to(device)
netD = Discriminator(num_classes, embedding_dim).to(device)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Prepare some fixed labels to visualize generator output evolution
fixed_noise = torch.randn(16, latent_dim, device=device)
fixed_labels = torch.randint(0, num_classes, (16,), device=device)  # random digits 0..9

# ---------------------------
# 6. Training Loop
# ---------------------------
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        labels = labels.to(device)  # ground-truth labels for the real batch
        b_size = real_imgs.size(0)

        # Real = 1, Fake = 0
        real_validity = torch.ones(b_size, 1, device=device)
        fake_validity = torch.zeros(b_size, 1, device=device)

        # ---- Train Discriminator ----
        netD.zero_grad()
        
        # Real images
        output_real = netD(real_imgs, labels)
        lossD_real = criterion(output_real, real_validity)
        
        # Fake images
        noise = torch.randn(b_size, latent_dim, device=device)
        random_labels = torch.randint(0, num_classes, (b_size,), device=device)
        fake_imgs = netG(noise, random_labels)
        
        output_fake = netD(fake_imgs.detach(), random_labels)
        lossD_fake = criterion(output_fake, fake_validity)
        
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ---- Train Generator ----
        netG.zero_grad()
        
        # Try to fool the discriminator
        output_fake_for_G = netD(fake_imgs, random_labels)
        lossG = criterion(output_fake_for_G, real_validity)
        lossG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

# ---------------------------
# 7. Generate Sample Captchas
# ---------------------------
with torch.no_grad():
    sample_imgs = netG(fixed_noise, fixed_labels).cpu()
sample_imgs = 0.5 * sample_imgs + 0.5  # from [-1,1] to [0,1]

grid = torchvision.utils.make_grid(sample_imgs, nrow=4)
plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.title("Conditional GAN - Generated Captchas")
plt.axis("off")
plt.show()
