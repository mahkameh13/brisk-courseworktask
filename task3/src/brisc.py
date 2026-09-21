
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(os.environ.get("BRISC_ROOT", "/work/dldevel/momeni/datasets/brisc2025"))
CLASS_TRAIN = ROOT / "classification_task" / "train"
SEG_TRAIN = ROOT / "segmentation_task" / "train"

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def build(batch_size=64, num_workers=8):
    """Return (dataset, train_data, val_data, train_loader, val_loader, encode_loader)."""
    classification_data_transform = datasets.ImageFolder(
        root=str(CLASS_TRAIN), transform=transform
    )

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(classification_data_transform))
    val_size = len(classification_data_transform) - train_size
    train_data, val_data = random_split(
        classification_data_transform, [train_size, val_size], generator=generator
    )

    
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    encode_loader = DataLoader(train_data, shuffle=False, **common)

    return (classification_data_transform, train_data, val_data,
            train_loader, val_loader, encode_loader)


if __name__ == "__main__":
    ds, tr, va, trl, val, enc = build()
    print("root      ", ROOT)
    print("classes   ", ds.classes)
    print("images    ", len(ds), "->", len(tr), "train /", len(va), "val")
    x, y = next(iter(trl))
    print("batch     ", tuple(x.shape), x.min().item(), x.max().item())
    print("device    ", device)
