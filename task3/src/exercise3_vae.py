

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

BRISC_CLASSIFICATION_TRAIN = Path(
    "/work/dldevel/momeni/datasets/brisc2025/classification_task/train"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 64
IMAGE_SIZE = 128
SEED = 42
PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 16384; divide recon by this for Ex2 units


def build_dataloaders(batch_size=64, num_workers=8):
    """Return train_loader, val_loader via the Exercise 2 pipeline."""
    from brisc import build

    _ds, _tr, _va, train_loader, val_loader, _enc = build(
        batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, val_loader


class ConvVAE(nn.Module):
    """Conv VAE for 1xHxW grayscale MRI. Encoder -> (mu, logvar); sample; decode.

    Decoder is a copy of Exercise 2 ConvAutoencoder.decoder / from_latent.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        # Same conv trunk as Exercise 2 encoder: 128x128 -> 8x8x128
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        hidden = 128 * 8 * 8
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        # Exercise 2 from_latent + decoder (unchanged)
        self.from_latent = nn.Linear(latent_dim, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(-1, 128, 8, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, diag(exp(logvar))) || N(0,I)) per sample, shape (B,)."""
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=-1)


def vae_loss(x_hat, x, mu, logvar, beta: float = 1.0):
    """Return (total, recon, kl) with recon summed over pixels then mean over batch."""
    recon = (x_hat - x).pow(2).flatten(start_dim=1).sum(dim=1).mean()
    kl = kl_divergence(mu, logvar).mean()
    total = recon + beta * kl
    return total, recon, kl


def train_vae(
    latent_dim,
    epochs,
    beta,
    tag,
    train_loader=None,
    val_loader=None,
    device=None,
    out_dir="outputs/task3",
):
    
    if device is None:
        device = DEVICE
    if train_loader is None or val_loader is None:
        train_loader, val_loader = build_dataloaders()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"best_{tag}.pt"

    model = ConvVAE(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        tr_sum = rec_sum = kl_sum = mu_sum = sig_sum = n_seen = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            total, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=beta)
            opt.zero_grad()
            total.backward()
            opt.step()
            bs = x.size(0)
            tr_sum += total.item() * bs
            rec_sum += recon.item() * bs
            kl_sum += kl.item() * bs
            mu_sum += mu.detach().abs().mean().item() * bs
            sig_sum += (0.5 * logvar.detach()).exp().mean().item() * bs
            n_seen += bs
        train_loss = tr_sum / n_seen
        recon_ep = rec_sum / n_seen
        kl_ep = kl_sum / n_seen
        mu_abs = mu_sum / n_seen
        sigma = sig_sum / n_seen

        model.eval()
        va_sum = n_val = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                total, _, _ = vae_loss(x_hat, x, mu, logvar, beta=beta)
                va_sum += total.item() * x.size(0)
                n_val += x.size(0)
        val_loss = va_sum / n_val

        row = {
            "train": train_loss,
            "val": val_loss,
            "recon": recon_ep,
            "kl": kl_ep,
            "mu_abs": mu_abs,
            "sigma": sigma,
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
        print(
            f"[{tag}] ep {epoch + 1:02d}  train {train_loss:.4f}  val {val_loss:.4f}"
            f"  recon {recon_ep:.4f} ({recon_ep / PIXELS:.5f}/px)"
            f"  kl {kl_ep:.4f}  |mu| {mu_abs:.4f}  sigma {sigma:.4f}"
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    hist_path = out / f"history_{tag}.json"
    hist_path.write_text(json.dumps({"beta": beta, "best_val": best_val, "history": history}, indent=2))
    return model, history, best_val


@torch.no_grad()
def interpolate(model: ConvVAE, x_a: torch.Tensor, x_b: torch.Tensor, steps: int = 9):
    """Straight line in latent space using posterior means (not samples)."""
    model.eval()
    device = next(model.parameters()).device
    if x_a.dim() == 3:
        x_a = x_a.unsqueeze(0)
    if x_b.dim() == 3:
        x_b = x_b.unsqueeze(0)
    mu_a, _ = model.encode(x_a.to(device))
    mu_b, _ = model.encode(x_b.to(device))
    frames = []
    for i in range(steps):
        t = i / (steps - 1)
        z = (1.0 - t) * mu_a + t * mu_b
        frames.append(model.decode(z).cpu())
    return torch.cat(frames, dim=0)


@torch.no_grad()
def sample_from_prior(model: ConvVAE, n: int = 8) -> torch.Tensor:
    """Decode z ~ N(0, I)."""
    model.eval()
    device = next(model.parameters()).device
    z = torch.randn(n, model.latent_dim, device=device)
    return model.decode(z).cpu()


def smoke_check_shapes():
    model = ConvVAE(latent_dim=8).to(DEVICE)
    x = torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    mu, logvar = model.encode(x)
    assert mu.shape == (2, 8) and logvar.shape == (2, 8)
    z = model.reparameterize(mu, logvar)
    assert z.shape == mu.shape
    x_hat = model.decode(z)
    assert x_hat.shape == x.shape
    x_hat2, mu2, logvar2 = model(x)
    assert x_hat2.shape == x.shape
    print("smoke_check_shapes: OK")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    print(f"device={DEVICE}")
    smoke_check_shapes()
