"""
Converted from task1.ipynb
"""

# %% [markdown] (Cell 0)
# # BRISC Brain Tumour Segmentation — Exercise 1
#

# %% (Cell 1)
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# %% (Cell 2)
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# %% [markdown] (Cell 3)
# ## Paths
#
# Kaggle input is read-only at `/kaggle/input/`. This is the only cell whose paths changed from the Colab version.

# %% (Cell 4)
from pathlib import Path
import os

ROOT = Path("/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/segmentation_task")

IMAGE_DIR_Train_path = ROOT / "train" / "images"
MASKS_DIR_Train_path = ROOT / "train" / "masks"
IMAGE_DIR_Test_path  = ROOT / "test"  / "images"
MASKS_DIR_Test_path  = ROOT / "test"  / "masks"

train_images = sorted(IMAGE_DIR_Train_path.glob("*.jpg"))
train_masks  = sorted(MASKS_DIR_Train_path.glob("*.png"))
test_images  = sorted(IMAGE_DIR_Test_path.glob("*.jpg"))
test_masks   = sorted(MASKS_DIR_Test_path.glob("*.png"))

print(len(train_images), len(train_masks), len(test_images), len(test_masks))
assert len(train_images) == 3933 and len(test_images) == 860


# %% (Cell 5)

print([path.stem for path in train_images[:5]])
print([path.stem for path in train_masks[:5]])


# %% (Cell 6)
stem_train_images = [path.stem for path in train_images]
stem_train_masks = [path.stem for path in train_masks]
print(stem_train_images[:5])
print(stem_train_masks[:5])


# %% (Cell 7)
IMAGE_DIR_Train_paths = {p.stem: p for p in train_images}
MASKS_DIR_Train_paths = {p.stem: p for p in train_masks}

print(len(IMAGE_DIR_Train_paths), len(MASKS_DIR_Train_paths))


# %% (Cell 8)
if len(IMAGE_DIR_Train_paths)==len(MASKS_DIR_Train_paths):
  print("True")
else:
  print("False")


# %% (Cell 9)


# assume IMAGE_DIR_Train_paths and MASKS_DIR_Train_paths are dicts: stem -> Path
common_keys = sorted(IMAGE_DIR_Train_paths.keys() & MASKS_DIR_Train_paths.keys())

paired = [(k, IMAGE_DIR_Train_paths[k], MASKS_DIR_Train_paths[k]) for k in common_keys]

# print explicitly so notebook won't also display the last expression value accidentally
print(len(paired), "pairs")
# if you want the notebook to display the repr instead of printing, make the paired variable the last line:
# paired


from collections import Counter


if Counter(IMAGE_DIR_Train_paths.keys()) == Counter(MASKS_DIR_Train_paths.keys()):
    print("True")
else:
    print("False")





if set(IMAGE_DIR_Train_paths) == set(MASKS_DIR_Train_paths):
  print("True")
else:
  print("False")


# %% (Cell 10)
# python


import numpy as np
from sklearn.model_selection import train_test_split

X = paired  # make sure 'paired' is defined

train_pairs , valid_pairs = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)

print("Trainv pairs" , len(train_pairs))
print("Val pairs" , len(valid_pairs))

assert len(train_pairs ) > 0 and  len(valid_pairs) > 0
assert len(train_pairs ) + len(valid_pairs) == len(X)


# %% (Cell 11)
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


class SegmentationDataset(Dataset):
  def __init__(self, pairs, transform=None):
    self.pairs = pairs
    self.transform = transform

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    _,image_path, mask_path = self.pairs[idx]

    image = tv_tensors.Image(Image.open(image_path).convert("L"))

    mask = tv_tensors.Mask(Image.open(mask_path).convert("L"))
    if self.transform is not None:
        image,mask = self.transform(image, mask)

    image = image.float()/ 255.0
    mask = (mask >0).float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)


    return image, mask

# Move the transform definition outside the class, to the global scope
train_transform = v2.Compose([
      v2.Resize((256,256), interpolation = InterpolationMode.BILINEAR , antialias = True ) ,
      v2.RandomHorizontalFlip(p=0.5) ,
      v2.RandomVerticalFlip(p=0.5) ,

  ])
eval_transform = v2.Compose([
      v2.Resize((256,256), interpolation = InterpolationMode.BILINEAR , antialias = True ) ,

  ])



# %% (Cell 12)
from torch.utils.data import DataLoader

# Instantiate the dataset
train_dataset = SegmentationDataset(train_pairs, transform=train_transform)

# Access a sample from the dataset
# The dataset's __getitem__ will now load the actual image and mask data as tensors
image, mask = train_dataset[0]

# Now let's check their types and shapes
print(f"Image type: {type(image)}")
print(f"Mask type: {type(mask)}")
print(f"Image shape: {image.shape}, min: {image.min().item()}, max: {image.max().item()}")
print(f"Mask shape: {mask.shape}, min: {mask.min().item()}, max: {mask.max().item()}")


# %% (Cell 13)
valid_dataset = SegmentationDataset(valid_pairs, transform=eval_transform)


a, _ = valid_dataset[0]
b , _ = valid_dataset[0]
assert torch.equal(a,b)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)


# %% (Cell 14)
# Quick sanity-check: visualize 4 image/mask pairs from your existing train_loader
images, masks = next(iter(train_loader))  # uses YOUR existing loader

fig, axes = plt.subplots(4, 2, figsize=(8, 14))
for i in range(4):
    axes[i, 0].imshow(images[i, 0].cpu(), cmap='gray')
    axes[i, 0].set_title(f"Image {i}"); axes[i, 0].axis('off')

    axes[i, 1].imshow(masks[i, 0].cpu(), cmap='gray')
    axes[i, 1].set_title(f"Mask {i}"); axes[i, 1].axis('off')

plt.tight_layout()
plt.show()

# Print stats
print(f"Images: {images.shape} | range [{images.min():.3f}, {images.max():.3f}]")
print(f"Masks:  {masks.shape}  | unique values: {torch.unique(masks)}")


# %% (Cell 15)
img , msk = train_dataset[0]
plt.imshow(img[0], cmap="gray")
plt.imshow(msk[0], alpha=0.4)


# %% (Cell 16)
# U-Net. Adapted from milesial/Pytorch-UNet (GPL-3.0):
#   https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
#   https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# My edits: base width 64 -> 32 (Colab), n_channels=1, n_classes=1, `skips` flag for the ablation.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base=32, bilinear=True, skips=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skips = skips

        factor = 2 if bilinear else 1
        self.inc   = DoubleConv(n_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16 // factor)
        self.up1   = Up(base * 16, base * 8 // factor, bilinear)
        self.up2   = Up(base * 8,  base * 4 // factor, bilinear)
        self.up3   = Up(base * 4,  base * 2 // factor, bilinear)
        self.up4   = Up(base * 2,  base, bilinear)
        self.outc  = OutConv(base, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ablation: same shapes, same parameter count, no information carried across
        if not self.skips:
            x1, x2, x3, x4 = (torch.zeros_like(t) for t in (x1, x2, x3, x4))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)          # raw logits, no sigmoid


# shape self-check — run this before you train anything
if True:
    m = UNet()
    out = m(torch.randn(2, 1, 256, 256))
    assert out.shape == (2, 1, 256, 256), out.shape
    print("ok", sum(p.numel() for p in m.parameters()), "params")


# %% (Cell 17)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


# %% (Cell 18)

class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


bce_loss  = nn.BCEWithLogitsLoss()
dice_loss = SoftDiceLossV1()

def combo_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


# %% (Cell 19)
mask  = (torch.rand(2, 1, 256, 256) > 0.97).float()   # ~3% positive, like a tumor
good  = mask * 20 - 10                                 # confident-correct logits
bad   = torch.full_like(mask, -10.)                    # predicts all background

print("perfect  ", bce_loss(good, mask).item(), dice_loss(good, mask).item())
print("all-bg   ", bce_loss(bad,  mask).item(), dice_loss(bad,  mask).item())


# %% (Cell 20)

from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


bce_loss = nn.BCEWithLogitsLoss()
dice_loss = SoftDiceLossV1()

def combo_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)



# %% (Cell 21)
mask = (torch.rand(4, 1, 256, 256) > 0.97).float()
good = mask * 20 -10
bad = torch.full_like(mask, -10.)

print(dice_coeff(good, mask))
print(dice_coeff(bad, mask))


# %% (Cell 22)
def dice_iou(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice, dice / (2 - dice)


# %% (Cell 23)
mask = (torch.rand(4, 1, 256, 256) > 0.97).float()
good = mask * 20 - 10
bad  = torch.full_like(mask, -10.)

d, i = dice_iou(good, mask)
assert d.mean() > 0.99 and i.mean() > 0.99, (d, i)

d, i = dice_iou(bad, mask)
assert d.mean() < 0.01 and i.mean() < 0.01, (d, i)

# cross-check against the milesial source
pred = (torch.sigmoid(good) > 0.5).float()
print(dice_iou(good, mask)[0].mean().item(),
      dice_coeff(pred, mask, reduce_batch_first=False).item())

print("metrics ok")


# %% (Cell 24)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    running = 0.0                                    # added: accumulate epoch loss
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)            # added: move to GPU

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running += loss.item()                       # added
    return running / len(dataloader)                 # added: mean epoch loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    dices, ious = [], []                             # added: replaces `correct`

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)        # added
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            d, i = dice_iou(pred, y)                 # added: replaces argmax accuracy
            dices.append(d)
            ious.append(i)

    test_loss /= num_batches
    return test_loss, torch.cat(dices).mean().item(), torch.cat(ious).mean().item()


# %% (Cell 25)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

def run_experiment(loss_fn, skips = True, epochs=25, seed = 42, tag="bce"):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  model = UNet(n_channels=1, n_classes=1, base=32, skips=skips)
  model.to(device=device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  train_losses, val_losses, val_dices, val_ious = [], [], [], []

  best = -1.0
  for t in range(epochs):
    tr = train_loop(train_loader, model, loss_fn , optimizer)
    vl, vd, vi = test_loop(valid_loader, model , loss_fn)

    train_losses.append(tr)
    val_losses.append(vl)
    val_dices.append(vd)
    val_ious.append(vi)

    if vd > best:
              best = vd
              torch.save(model.state_dict(), f"best_{tag}.pt")

    print(f"epoch {t+1:2d}  train {tr:.4f}  val {vl:.4f}  dice {vd:.4f}  iou {vi:.4f}")

  history = {"train_loss": train_losses, "val_loss": val_losses,
                "val_dice": val_dices, "val_iou": val_ious}
  return history, best


# %% (Cell 26)
hist_bce, best_bce_dice = run_experiment(bce_loss, epochs=25, tag="bce")


# %% (Cell 27)
hist_dice , best_dice_dice = run_experiment(dice_loss, epochs=25, tag="dice")
hist_combo , best_combo = run_experiment(combo_loss, epochs=25, tag="combo")



# %% (Cell 28)
histories = {
    'dice_loss_history': hist_dice,
    'combo_loss_history': hist_combo,
    "bce_loss_history": hist_bce,
}
torch.save(histories, '/kaggle/working/histories.pt')
print("saved", {k: len(v['val_dice']) for k, v in histories.items()})


# %% (Cell 29)
print(len(hist_bce['train_loss']), len(hist_bce['val_dice']), max(hist_bce['val_dice']))


# %% (Cell 30)
import matplotlib.pyplot as plt

# Access the data from the hist_bce dictionary
train_losses = hist_bce['train_loss']
val_losses = hist_bce['val_loss']
val_dices = hist_bce['val_dice']
val_ious = hist_bce['val_iou']

epochs = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_losses, label="train")
ax1.plot(epochs, val_losses, label="validation")
ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss")
ax1.set_title("Loss"); ax1.legend()

ax2.plot(epochs, val_dices, label="Dice")
ax2.plot(epochs, val_ious, label="IoU")
ax2.set_xlabel("epoch"); ax2.set_ylabel("score")
ax2.set_title("Validation metrics"); ax2.legend()

plt.tight_layout()
plt.show()


# %% (Cell 31)
# Summary using hist_bce keys
val_dices = hist_bce['val_dice']
val_ious = hist_bce['val_iou']

best = max(range(len(val_dices)), key=lambda k: val_dices[k])
print(f"best epoch {best+1}: dice {val_dices[best]:.4f}  iou {val_ious[best]:.4f}")
print(f"final epoch:  dice {val_dices[-1]:.4f}  iou {val_ious[-1]:.4f}")


# %% (Cell 32)
import numpy as np

TAIL = 10
sds = [np.std(np.array(h["val_dice"][-TAIL:]), ddof=1)
       for h in (hist_bce, hist_dice, hist_combo)]
sigma = float(np.mean(sds))

rng = np.random.default_rng(0)
spreads = []
for h in (hist_bce, hist_dice, hist_combo):
    mu = float(np.mean(h["val_dice"][-TAIL:]))
    sims = [np.max(rng.normal(mu, sigma, TAIL)) for _ in range(20000)]
    spreads.append(np.percentile(sims, 95) - np.percentile(sims, 5))

print(f"pooled within-run sd (last {TAIL} epochs): {sigma:.4f}")
print(f"2 sigma                                 : {2*sigma:.4f}")
print(f"simulated best-of-{TAIL} 5-95 spread      : {np.mean(spreads):.4f}")

NOISE = 0.04
print(f"\nNOISE = {NOISE}")

bests = {"BCE": max(hist_bce["val_dice"]),
         "Dice": max(hist_dice["val_dice"]),
         "BCE+Dice": max(hist_combo["val_dice"])}
rng_loss = max(bests.values()) - min(bests.values())
for k, v in bests.items():
    print(f"  {k:9s} {v:.4f}")
print(f"range across losses: {rng_loss:.4f}  -> "
      f"{'within noise, no loss function wins' if rng_loss < NOISE else 'a real difference'}")


# %% (Cell 33)
hist_noskip, best_noskip = run_experiment(bce_loss, skips=False, epochs=25, tag="noskip")

drop = best_bce_dice - best_noskip
print(f"skips   {best_bce_dice:.4f}")
print(f"no-skip {best_noskip:.4f}")
print(f"drop    {drop:.4f}   threshold {NOISE}")
print("VERDICT:", "skip connections help" if drop > NOISE else "within noise")


# %% (Cell 34)
torch.save({"noskip": hist_noskip}, "/kaggle/working/hist_noskip.pt")
print("saved", len(hist_noskip["val_dice"]), "epochs")


# %% (Cell 35)
plt.ioff()
hists = {"BCE": hist_bce, "Dice": hist_dice, "BCE+Dice": hist_combo,
         "BCE no-skip": hist_noskip}

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, h in hists.items():
    y = h["val_dice"]
    print(name, len(y), y[:3])          # must print 25 numbers, not 0
    ax.plot(range(1, len(y) + 1), y, marker="o", ms=3, label=name)

top = max(max(h["val_dice"]) for h in hists.values())
ax.axhspan(top - NOISE, top, color="grey", alpha=0.15, label=f"noise band ({NOISE}, converged-regime)")
ax.set_xlabel("epoch"); ax.set_ylabel("validation Dice"); ax.legend(); ax.grid(alpha=0.3)
plt.show()


# %% (Cell 36)
model = UNet(n_channels=1, n_classes=1, base=32, skips=True)
model.load_state_dict(torch.load("best_bce.pt", map_location=device))
model.to(device).eval()

scores, imgs, gts, probs = [], [], [], []
with torch.no_grad():
    for X, y in valid_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        d, _ = dice_iou(logits, y)
        scores.append(d.cpu()); imgs.append(X.cpu())
        gts.append(y.cpu());   probs.append(torch.sigmoid(logits).cpu())

scores = torch.cat(scores); imgs = torch.cat(imgs)
gts    = torch.cat(gts);    probs = torch.cat(probs)

print(f"n = {len(scores)}")
print(f"mean {scores.mean():.4f}   median {scores.median():.4f}")
print(f"min  {scores.min():.4f}    max    {scores.max():.4f}")
print(f"below 0.5: {(scores < 0.5).sum().item()} images")


# %% (Cell 37)
order = torch.argsort(scores)
picks = [(order[-1].item(), "best"),
         (order[len(order)//2].item(), "median"),
         (order[0].item(), "worst")]

fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
for r, (idx, label) in enumerate(picks):
    pred = (probs[idx, 0] > 0.5).float()
    axes[r, 0].imshow(imgs[idx, 0], cmap="gray")
    axes[r, 0].set_ylabel(f"{label}\nDice {scores[idx]:.3f}", fontsize=10)
    axes[r, 1].imshow(gts[idx, 0], cmap="gray")
    axes[r, 2].imshow(pred, cmap="gray")
    for c in range(3):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
for c, t in enumerate(["input MRI", "ground truth", "prediction"]):
    axes[0, c].set_title(t)
plt.tight_layout(); plt.show()


# %% (Cell 38)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (idx, label) in zip(axes, picks):
    ax.imshow(imgs[idx, 0], cmap="gray")
    ax.contour(gts[idx, 0],   levels=[0.5], colors="lime", linewidths=1.4)
    ax.contour(probs[idx, 0], levels=[0.5], colors="red",  linewidths=1.2)
    ax.set_title(f"{label}  Dice {scores[idx]:.3f}", fontsize=10)
    ax.axis("off")
fig.suptitle("green = ground truth, red = prediction", fontsize=10)
plt.tight_layout(); plt.show()


# %% (Cell 39)
model_ns = UNet(n_channels=1, n_classes=1, base=32, skips=False)   # skips=False
model_ns.load_state_dict(torch.load("best_noskip.pt", map_location=device))
model_ns.to(device).eval()

with torch.no_grad():
    probs_ns = torch.sigmoid(model_ns(imgs[[i for i, _ in picks]].to(device))).cpu()

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for k, (ax, (idx, label)) in enumerate(zip(axes, picks)):
    ax.imshow(imgs[idx, 0], cmap="gray")
    ax.contour(gts[idx, 0],    levels=[0.5], colors="lime",   linewidths=1.4)
    ax.contour(probs_ns[k, 0], levels=[0.5], colors="orange", linewidths=1.2)
    ax.set_title(f"{label} (no skips)", fontsize=10)
    ax.axis("off")
fig.suptitle("green = ground truth, orange = no-skip prediction", fontsize=10)
plt.tight_layout(); plt.show()


# %% [markdown] (Cell 40)
# ## Test-set evaluation
#
# Model selection is complete at this point. The test set is evaluated **once**, after
# selection, and no model, loss function or threshold is adjusted in response to it.
#
# Test pairs are matched by filename stem, as in training, so a mismatch between the two
# directories cannot silently misalign an image with another image's mask.

# %% (Cell 41)
test_img_paths  = {p.stem: p for p in test_images}
test_mask_paths = {p.stem: p for p in test_masks}
common_test = sorted(test_img_paths.keys() & test_mask_paths.keys())
test_pairs  = [(k, test_img_paths[k], test_mask_paths[k]) for k in common_test]

print(f"{len(test_images)} images, {len(test_masks)} masks -> {len(test_pairs)} pairs")
assert len(test_pairs) > 0

test_dataset = SegmentationDataset(test_pairs, transform=eval_transform)   # eval, no flips
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False,
                          num_workers=4, pin_memory=True)


# %% (Cell 42)
def evaluate(ckpt, loss_fn, skips=True):
    m = UNet(n_channels=1, n_classes=1, base=32, skips=skips)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.to(device).eval()
    return test_loop(test_loader, m, loss_fn)

runs = [("BCE",         "best_bce.pt",    bce_loss,   True),
        ("Dice",        "best_dice.pt",   dice_loss,  True),
        ("BCE+Dice",    "best_combo.pt",  combo_loss, True),
        ("BCE no-skip", "best_noskip.pt", bce_loss,   False)]   # skips=False

print(f"{'model':12s} {'loss':>8s} {'Dice':>8s} {'IoU':>8s}")
test_results = {}
for name, ckpt, fn, sk in runs:
    try:
        loss, d, i = evaluate(ckpt, fn, skips=sk)
        test_results[name] = (d, i)
        print(f"{name:12s} {loss:8.4f} {d:8.4f} {i:8.4f}")
    except FileNotFoundError:
        print(f"{name:12s}   [checkpoint not found]")

print()
print(f"BCE validation Dice {best_bce_dice:.4f}")
if "BCE" in test_results:
    print(f"BCE test       Dice {test_results['BCE'][0]:.4f}")
    print(f"gap                 {best_bce_dice - test_results['BCE'][0]:+.4f}")


# %% [markdown] (Cell 43)
# # Q1 : What do skip connections contribute  
#    The ablation zeroes the encoder maps `x1` to `x4` inside `UNet.forward`. Concatenation, layer
#    shapes and the 4,317,825 parameters stay as they are, so the decoder loses information and keeps
#    its capacity. The conventional ablation deletes the concatenation and halves the decoder's input
#    width, which mixes those two effects together.
#
#    The BCE baseline peaked at 0.7833 validation Dice on epoch 23. Zeroed, it peaked at 0.7421 on
#    epoch 21. The drop is 0.0412 against the 0.04 threshold fixed before the run, so it clears by
#    0.0012, and the noise-floor section put the best-of-ten spread at 0.086. One run cannot settle
#    it. The test set, evaluated once on 860 images, gives 0.7913 against 0.7533, a drop of 0.0380 in
#    the same direction and of the same size. The skip connections contribute about four Dice points.
#
#    The cause is spatial. Four `MaxPool2d` steps take 256x256 down to a 16x16 bottleneck, and each
#    step buys receptive field by discarding where a feature sat. At `x5` the network knows a tumour
#    is present and roughly where, but not which pixel holds the edge. The skips pass the pre-pooling
#    maps to the decoder at matching resolutions and return that precision. Ronneberger et al. call
#   
#
#    Four points fits that cause. The no-skip model still finds the tumour and still scores 0.7533 on
#    the test set. It loses the boundary, where a few pixels move Dice by a few points. The smallest
#    tumours should suffer most, since their extent approaches the 16-pixel resolution of the
#    bottleneck. The overlays above show both models on the same three slices.

# %% [markdown] (Cell 44)
#
#    ## Q2 :  Dice and IoU both measure overlap. Why can accuracy be high while Dice is low?
#
#    Take TP, FP, FN and TN as pixel counts:
#
#    $$\text{Dice} = \frac{2\,TP}{2\,TP + FP + FN}, \qquad \text{IoU} = \frac{TP}{TP + FP + FN}, \qquad \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
#
#    Only accuracy contains TN. The tumour covers a small part of each BRISC slice, so TN dominates.
#    A model that predicts background everywhere has TP = 0 and FP = 0. Its Dice and IoU are 0. Its
#    accuracy is above 95%, earned by ignoring the region nobody asked about. Segmentation is scored
#    with Dice and IoU for that reason.
#
#    The two overlap metrics are one measurement. Eliminating TP + FP + FN gives
#
#    $$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$
#
#    and `dice_iou` returns `dice / (2 - dice)` instead of counting twice. The map is monotonic on
#    [0,1], so the two rank models identically. They differ in scale, since Dice counts the
#    intersection twice in its numerator and punishes an error less.
#
#    Both are computed per image with reduction over `dim=(1,2,3)`, then averaged. A global
#    intersection and union would let large tumours set the score. The per-image mean weights every
#    patient equally and is the stricter number.

# %% [markdown] (Cell 45)
#  ## Q3 — Must geometric augmentation be applied identically to image and mask?
#
#    Yes, necessarily. A mask is a per-pixel label in the image's own coordinate frame, and each
#    entry states that pixel (i, j) belongs to the tumour. A geometric transform remaps that frame,
#    so transforming one tensor and not the other makes every label in the pair false.
#
#    Flip the image alone and the network is told to find tumour where the image now shows healthy
#    tissue. Each flip is sampled independently with p = 0.5, so about half the dataset carries
#    inverted supervision. No weights satisfy both halves. Gradient descent converges to the
#    conditional mean, a diffuse prediction near the average mask position, and Dice collapses. No
#    error is raised in the training loop, so the run looks healthy while it happens.
#
#    The dataset wraps its tensors as `tv_tensors.Image` and `tv_tensors.Mask` and passes both
#    through one `v2.Compose` call. The v2 API samples the random parameters once per call and
#    dispatches them to every tensor in that call, so image and mask share one transform. Two
#    separate `Compose` objects would draw two independent random numbers, even when configured
#    identically, and the bug would return.
#
#    Interpolation needs the same care. The resize uses `InterpolationMode.BILINEAR` with
#    antialiasing, which suits continuous intensities and puts intermediate values along every mask
#    boundary. The dataset re-binarises with `mask = (mask > 0).float()` afterwards, restoring the
#    strict {0, 1} target that `BCEWithLogitsLoss` requires. Augmentation appears only in
#    `train_transform`. `eval_transform` resizes and nothing else, so validation and test scores stay
#    deterministic and comparable across epochs and models.

# %% [markdown] (Cell 46)
#  ## Q4 — The no_tumor class has no mask. How was it handled?
#
#    The `segmentation_task/` split ships masks for the three tumorous classes only, with no
#    all-black masks for `no_tumor`. No filter was needed. The pairing step drops them by
#    construction:
#
#        common_keys = sorted(IMAGE_DIR_Train_paths.keys() & MASKS_DIR_Train_paths.keys())
#        paired = [(k, IMAGE_DIR_Train_paths[k], MASKS_DIR_Train_paths[k]) for k in common_keys]
#
#    A stem enters `paired` only when both the `.jpg` and the `.png` exist, so an unmasked slice
#    cannot survive the intersection. A written rule could be dropped in a refactor; a set
#    intersection cannot. It also blocks a quieter failure, since zipping two sorted lists by
#    position would pair one patient's image with another patient's mask and report nothing.
#
#    Including those slices would require an all-zero mask. The label is well-formed and the metric
#    is not, because Dice on an empty ground truth evaluates as 0/0. The smoothing constant decides
#    the outcome. With prediction and target both empty, `dice = (2*0 + eps) / (0 + eps) = 1.0`,
#    which scores a correct negative as perfect, and any false positive drives it toward 0. The
#    convention is sound and needs stating, since a different placement of eps would return 0 for the
#    same case and penalise correct behaviour.
#
#    There is a statistical cost as well. Tumour-free slices raise the background fraction and worsen
#    an imbalance that is already severe, pushing a BCE-trained model further toward the
#    all-background solution from Q2. Splitting the problem avoids it. A classifier decides whether a
#    tumour is present, and the segmentation model runs on positive cases only. BRISC implies that
#    split, keeping `classification_task/` and `segmentation_task/` as separate problems.

# %% (Cell 47)
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))    # want (7,5) for T4
print(torch.cuda.get_arch_list())             # sm_60 absent on P100 = the bug


