"""Student stubs for Exercise 3 (ConvVAE).

Implement one piece at a time; run tests/test_task3.py after each change.
"""
from __future__ import annotations

import math

import torch
from torch import nn

LATENT_DIM = 64
IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConvVAE(nn.Module):
    """1x128x128 grayscale MRI → (mu, logvar) → sample → image in [0, 1]."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        # TODO: conv trunk 1→16→32→64→128, each Conv2d(k=4, stride=2, padding=1) + ReLU
        # TODO: two Linear heads on 128*8*8: fc_mu and fc_logvar
        # TODO: from_latent Linear(latent_dim, 128*8*8)
        # TODO: decoder ConvTranspose 128→64→32→16→1, ReLU, last layer Sigmoid
        raise NotImplementedError("ConvVAE.__init__")

    def encode(self, x):
        """x: (B,1,128,128) → (mu, logvar) each (B, latent_dim)."""
        raise NotImplementedError("encode")

    def reparameterize(self, mu, logvar):
        """z = mu + exp(0.5 * logvar) * eps, eps ~ N(0,I). Use torch.randn_like."""
        raise NotImplementedError("reparameterize")

    def decode(self, z):
        """z: (B, latent_dim) → (B,1,128,128) in [0, 1]."""
        raise NotImplementedError("decode")

    def forward(self, x):
        """Return (x_hat, mu, logvar)."""
        raise NotImplementedError("forward")


def kl_divergence(mu, logvar):
    """Shape (B,). 0.5 * sum(mu^2 + exp(logvar) - logvar - 1) over latent dims."""
    raise NotImplementedError("kl_divergence")


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """Return (total, recon, kl) scalars.
    recon: sum squared error over pixels, then mean over batch.
    kl: mean of kl_divergence.
    total = recon + beta * kl.
    """
    raise NotImplementedError("vae_loss")


def interpolate(model, x_a, x_b, steps=9):
    """Straight line on posterior *means*, not samples. Return (steps,1,128,128)."""
    raise NotImplementedError("interpolate")


def sample_from_prior(model, n=8):
    """decode(randn(n, latent_dim))."""
    raise NotImplementedError("sample_from_prior")
