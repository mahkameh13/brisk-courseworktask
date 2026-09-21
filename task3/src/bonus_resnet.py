

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_feature_transform(size=224):
    """PIL image -> (3, size, size) with ImageNet normalization.

    Channel replication happens before Normalize because ImageNet stats
    differ per channel.
    """
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_frozen_resnet():
    """Pretrained ResNet-18, ImageNet head removed, frozen, eval mode."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device="cpu"):
    """Return (features, labels) numpy arrays. Autograd stays off."""
    model = model.to(device)
    model.eval()
    feats, labs = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        out = model(x)
        feats.append(out.detach().cpu().numpy())
        labs.append(y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y))
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)


def reduce_dimensions(train_features, val_features, n_components=64):
    """Fit PCA on train only, apply the same basis to val."""
    pca = PCA(n_components=n_components)
    train_reduced = pca.fit_transform(train_features)
    val_reduced = pca.transform(val_features)
    return train_reduced, val_reduced


def evaluate_features(
    train_features, train_labels, val_features, val_labels, n_neighbors=5
):
    """Same 5-NN protocol as Exercise 2."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_features, train_labels)
    pred = knn.predict(val_features)
    acc = float(accuracy_score(val_labels, pred))
    n_labels = len(np.unique(val_labels))
    if n_labels >= 2 and val_features.shape[0] > n_labels:
        sil = float(silhouette_score(val_features, val_labels))
    else:
        sil = float("nan")
    report = classification_report(val_labels, pred, zero_division=0)
    return {"accuracy": acc, "silhouette": sil, "report": report}
