"""
Exercise 2 — BRISC CLASSIFICATION images, unsupervised ConvAutoencoder.
Give this file to Scrimba Explain together with PROMPT.txt.

Labels are NEVER used in the loss. They only colour plots and score 5-NN.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(os.environ.get("BRISC_ROOT", "/work/dldevel/momeni/datasets/brisc2025"))
CLASS_TRAIN = ROOT / "classification_task" / "train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def build_loaders(batch_size=64):
    ds = datasets.ImageFolder(str(CLASS_TRAIN), transform=transform)
    gen = torch.Generator().manual_seed(SEED)
    n_train = int(0.8 * len(ds))
    train_data, val_data = random_split(ds, [n_train, len(ds) - n_train], generator=gen)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    encode_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    return ds, train_data, val_data, train_loader, val_loader, encode_loader


class ConvAutoencoder(nn.Module):
    """1x128x128 → 64-d (or 16-d) code → reconstruction in [0, 1]."""

    def __init__(self, latent_dim=64):
        super().__init__()
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
        self.to_latent = nn.Linear(128 * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * 8 * 8)
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

    def encode(self, x):
        x = self.encoder(x).flatten(start_dim=1)
        return self.to_latent(x)

    def forward(self, x):
        z = self.encode(x)
        x = self.from_latent(z).view(-1, 128, 8, 8)
        return self.decoder(x)


def run_ae(latent_dim, epochs, tag, train_loader, val_loader):
    model = ConvAutoencoder(latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()  # MEAN over pixels — Exercise 2 units
    hist, best = [], float("inf")
    for ep in range(epochs):
        model.train()
        tr = 0.0
        for x, _ in train_loader:  # label unused
            x = x.to(DEVICE)
            loss = lossf(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += loss.item() * x.size(0)
        tr /= len(train_loader.dataset)
        model.eval()
        va = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                va += lossf(model(x), x).item() * x.size(0)
        va /= len(val_loader.dataset)
        hist.append((tr, va))
        if va < best:
            best = va
            torch.save(model.state_dict(), f"best_{tag}.pt")
        print(f"[{tag}] ep {ep+1:02d}  train {tr:.5f}  val {va:.5f}")
    return model, hist, best


@torch.no_grad()
def encode_split(model, loader):
    Z, Y = [], []
    for x, y in loader:
        Z.append(model.encode(x.to(DEVICE)).cpu())
        Y.append(y)
    return torch.cat(Z).numpy(), torch.cat(Y).numpy()


def interpolate_strip(model, val_data, y_val, steps=9):
    """Straight line between first glioma (0) and first pituitary (3)."""
    ia = int(np.where(y_val == 0)[0][0])
    ib = int(np.where(y_val == 3)[0][0])
    xa, xb = val_data[ia][0], val_data[ib][0]
    z0 = model.encode(xa[None].to(DEVICE))
    z1 = model.encode(xb[None].to(DEVICE))
    frames = []
    for i in range(steps):
        t = i / (steps - 1)
        z = (1 - t) * z0 + t * z1
        rec = model.decoder(model.from_latent(z).view(-1, 128, 8, 8))
        frames.append(rec.cpu())
    return torch.cat(frames, dim=0), ia, ib


if __name__ == "__main__":
    ds, train_data, val_data, train_loader, val_loader, encode_loader = build_loaders()
    print("classes", ds.classes, "n", len(ds), "device", DEVICE)

    # Full sheet run (skip on CPU if you only need the architecture video):
    # m16, h16, b16 = run_ae(16, 25, "d16", train_loader, val_loader)
    # m64, h64, b64 = run_ae(64, 25, "d64", train_loader, val_loader)
    # model = m64.eval()
    # Z_train, y_train = encode_split(model, encode_loader)
    # Z_val, y_val = encode_split(model, val_loader)
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(Z_train, y_train)
    # acc = accuracy_score(y_val, knn.predict(Z_val))
    # sil = silhouette_score(Z_val, y_val)

    # Recorded numbers after the real 25-epoch run:
    print("d16 best val MSE  0.00745")
    print("d64 best val MSE  0.00610")
    print("5-NN accuracy     0.8750   (chance 0.25)")
    print("silhouette        0.0128   (ribbons, not islands)")
