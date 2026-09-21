"""
Exercise 1 — BRISC brain-tumour SEGMENTATION (U-Net).
Give this file to Scrimba Explain together with PROMPT.txt.

Data: segmentation_task images + masks.
Input: 1x256x256 grayscale. Output: binary tumour mask.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(os.environ.get(
    "BRISC_ROOT",
    "/work/dldevel/momeni/datasets/brisc2025",
)) / "segmentation_task"

IMAGE_DIR_Train_path = ROOT / "train" / "images"
MASKS_DIR_Train_path = ROOT / "train" / "masks"
IMAGE_DIR_Test_path = ROOT / "test" / "images"
MASKS_DIR_Test_path = ROOT / "test" / "masks"


# Pair images to masks by filename stem (never by list index)

def pair_by_stem(image_dir: Path, mask_dir: Path):
    images = {p.stem: p for p in sorted(image_dir.glob("*.jpg"))}
    masks = {p.stem: p for p in sorted(mask_dir.glob("*.png"))}
    keys = sorted(images.keys() & masks.keys())
    return [(k, images[k], masks[k]) for k in keys]


class SegmentationDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, image_path, mask_path = self.pairs[idx]
        image = tv_tensors.Image(Image.open(image_path).convert("L"))
        mask = tv_tensors.Mask(Image.open(mask_path).convert("L"))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image = image.float() / 255.0
        mask = (mask > 0).float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask


train_transform = v2.Compose([
    v2.Resize((256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])
eval_transform = v2.Compose([
    v2.Resize((256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True),
])



# U-Net (milesial, base 32, 1 channel, 1 class). skips=False zeros encoder
# maps but keeps concat + parameter count

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base=32, bilinear=True, skips=True):
        super().__init__()
        self.skips = skips
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16 // factor)
        self.up1 = Up(base * 16, base * 8 // factor, bilinear)
        self.up2 = Up(base * 8, base * 4 // factor, bilinear)
        self.up3 = Up(base * 4, base * 2 // factor, bilinear)
        self.up4 = Up(base * 2, base, bilinear)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if not self.skips:
            x1, x2, x3, x4 = (torch.zeros_like(t) for t in (x1, x2, x3, x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)  # logits, no sigmoid



# Losses: BCE on logits, soft Dice, combo. Tumours are ~3% of pixels so
# BCE-only can look good while missing the mass.

class SoftDiceLossV1(nn.Module):
    def __init__(self, p=1, smooth=1):
        super().__init__()
        self.p, self.smooth = p, smooth

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        return 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)


bce_loss = nn.BCEWithLogitsLoss()
dice_loss = SoftDiceLossV1()


def combo_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


def dice_iou(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice, dice / (2 - dice)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    running = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running += loss.item()
    return running / len(dataloader)


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    test_loss, dices, ious = 0.0, [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            d, i = dice_iou(pred, y)
            dices.append(d)
            ious.append(i)
    return (
        test_loss / len(dataloader),
        torch.cat(dices).mean().item(),
        torch.cat(ious).mean().item(),
    )


def run_experiment(loss_fn, train_loader, valid_loader, device, skips=True, epochs=25, seed=42, tag="bce"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = UNet(n_channels=1, n_classes=1, base=32, skips=skips).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best = -1.0
    for t in range(epochs):
        tr = train_loop(train_loader, model, loss_fn, optimizer, device)
        vl, vd, vi = test_loop(valid_loader, model, loss_fn, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_dice"].append(vd)
        history["val_iou"].append(vi)
        if vd > best:
            best = vd
            torch.save(model.state_dict(), f"best_{tag}.pt")
        print(f"epoch {t+1:2d}  train {tr:.4f}  val {vl:.4f}  dice {vd:.4f}  iou {vi:.4f}")
    return history, best


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paired = pair_by_stem(IMAGE_DIR_Train_path, MASKS_DIR_Train_path)
    train_pairs, valid_pairs = train_test_split(paired, test_size=0.2, random_state=42, shuffle=True)
    train_loader = DataLoader(
        SegmentationDataset(train_pairs, train_transform),
        batch_size=8, shuffle=True, num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        SegmentationDataset(valid_pairs, eval_transform),
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True,
    )
    print("pairs", len(paired), "train", len(train_pairs), "val", len(valid_pairs), "device", device)
    # Four runs the sheet asks for:
    # run_experiment(bce_loss, ..., tag="bce")
    # run_experiment(dice_loss, ..., tag="dice")
    # run_experiment(combo_loss, ..., tag="combo")
    # run_experiment(bce_loss, ..., skips=False, tag="noskip")
    print("UNet params", sum(p.numel() for p in UNet().parameters()))
