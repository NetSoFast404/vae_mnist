import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

# =====================
# Config
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs = 50
lr = 1e-3
latent_dim = 20
save_dir = "./vae_mnist_runs"
os.makedirs(save_dir, exist_ok=True)

# =====================
# Data
# =====================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =====================
# Model
# =====================
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()

        # encoder
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # decoder
        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        x_recon = torch.sigmoid(self.fc3(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# =====================
# Loss
# =====================
def vae_loss(x_recon, x, mu, logvar):
    # reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss

# =====================
# Init
# =====================
model = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# =====================
# Save images
# =====================
@torch.no_grad()
def save_reconstruction(model, epoch):
    model.eval()
    x, _ = next(iter(test_loader))
    x = x[:16].to(device)
    x_flat = x.view(-1, 28 * 28)

    x_recon, _, _ = model(x_flat)
    x_recon = x_recon.view(-1, 1, 28, 28)

    comparison = torch.cat([x, x_recon], dim=0)
    utils.save_image(comparison.cpu(), f"{save_dir}/reconstruction_epoch_{epoch}.png", nrow=8)

@torch.no_grad()
def save_samples(model, epoch):
    model.eval()
    z = torch.randn(16, latent_dim).to(device)
    samples = model.decode(z).view(-1, 1, 28, 28)
    utils.save_image(samples.cpu(), f"{save_dir}/samples_epoch_{epoch}.png", nrow=4)

# =====================
# Train
# =====================
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for x, _ in pbar:
        x = x.to(device)
        x = x.view(-1, 28 * 28)

        optimizer.zero_grad()

        x_recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
    scheduler.step()

    avg_loss = total_loss / len(train_dataset)
    avg_recon = total_recon / len(train_dataset)
    avg_kl = total_kl / len(train_dataset)

    print(
        f"Epoch {epoch}: "
        f"loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}"
    )

    save_reconstruction(model, epoch)
    save_samples(model, epoch)

# save model
torch.save(model.state_dict(), f"{save_dir}/vae_mnist.pt")
print("Training finished.")
