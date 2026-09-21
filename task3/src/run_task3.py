"""Train two VAE betas, save figures, run frozen ResNet-18 bonus."""
#!/usr/bin/env python3
"""Train two VAE betas, save interpolation/prior figures, score frozen ResNet-18."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from bonus_resnet import (
    build_feature_transform,
    build_frozen_resnet,
    evaluate_features,
    extract_features,
    reduce_dimensions,
)
from brisc import CLASS_TRAIN, build, device as brisc_device
from exercise3_vae import (
    PIXELS,
    ConvVAE,
    interpolate,
    sample_from_prior,
    train_vae,
    vae_loss,
)

OUT = Path("outputs/task3")
EX2_D64_MSE = 0.00610
EX2_KNN_ACC = 0.8750
EX2_SIL = 0.0128


def _pair_from_val(val_data):
    """First glioma (0) and first pituitary (3) in the Exercise 2 val split."""
    ia = ib = None
    for i in range(len(val_data)):
        _, y = val_data[i]
        y = int(y)
        if y == 0 and ia is None:
            ia = i
        if y == 3 and ib is None:
            ib = i
        if ia is not None and ib is not None:
            break
    xa = val_data[ia][0]
    xb = val_data[ib][0]
    return xa, xb, ia, ib


def _save_strip(frames, path, titles=None):
    n = frames.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.2))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(frames[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=7)
        else:
            ax.set_title(f"a={i / (n - 1):.2f}" if n > 1 else "prior", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _val_recon_units(model, val_loader, device, beta):
    model.eval()
    rec_sum = n = 0.0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            _, recon, _ = vae_loss(x_hat, x, mu, logvar, beta=beta)
            rec_sum += recon.item() * x.size(0)
            n += x.size(0)
    recon_sum = rec_sum / n
    return recon_sum, recon_sum / PIXELS


def _scatter(ax, xy, labels, class_names, title):
    for k, name in enumerate(class_names):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=8, label=name)
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--smoke", action="store_true", help="1 epoch, skip ResNet")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-resnet", action="store_true")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1
        args.skip_resnet = True

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "| brisc", brisc_device)

    ds, train_data, val_data, train_loader, val_loader, encode_loader = build()
    print("classes", ds.classes, "n", len(ds), "train", len(train_data), "val", len(val_data))

    metrics = {
        "exercise2": {
            "d64_val_mse_per_pixel": EX2_D64_MSE,
            "d64_val_mse_sum_pixels": EX2_D64_MSE * PIXELS,
            "knn_acc": EX2_KNN_ACC,
            "silhouette": EX2_SIL,
        }
    }

    trained = {}
    if not args.skip_train:
        for beta, tag in [(0.01, "b001"), (1.0, "b1")]:
            model, hist, best = train_vae(
                64,
                args.epochs,
                beta,
                tag,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                out_dir=str(out),
            )
            recon_sum, recon_px = _val_recon_units(model, val_loader, device, beta)
            last = hist[-1]
            metrics[tag] = {
                "beta": beta,
                "best_val_total": best,
                "last_mu_abs": last["mu_abs"],
                "last_sigma": last["sigma"],
                "val_recon_sum_pixels": recon_sum,
                "val_recon_per_pixel": recon_px,
            }
            trained[tag] = model
            print(
                f"{tag}: recon {recon_sum:.2f} sum-px  {recon_px:.5f}/px  "
                f"(Ex2 d64 was {EX2_D64_MSE:.5f}/px = {EX2_D64_MSE * PIXELS:.1f} sum-px)"
            )
    else:
        for tag, beta in [("b001", 0.01), ("b1", 1.0)]:
            path = out / f"best_{tag}.pt"
            if not path.exists():
                continue
            model = ConvVAE(64).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            trained[tag] = model

    model = trained.get("b1") or trained.get("b001")
    if model is not None:
        xa, xb, ia, ib = _pair_from_val(val_data)
        frames = interpolate(model, xa, xb, steps=9)
        _save_strip(frames, out / "interpolate_vae.png")
        priors = sample_from_prior(model, n=8)
        _save_strip(
            priors,
            out / "prior_samples.png",
            titles=[f"z~N(0,I) {i}" for i in range(8)],
        )
        metrics["interpolation"] = {"ia": ia, "ib": ib, "steps": 9}

    if not args.skip_resnet:
        resnet_tf = build_feature_transform()
        ds_res = datasets.ImageFolder(str(CLASS_TRAIN), transform=resnet_tf)
        train_res = Subset(ds_res, train_data.indices)
        val_res = Subset(ds_res, val_data.indices)
        common = dict(batch_size=64, num_workers=8, pin_memory=device == "cuda", shuffle=False)
        tr_loader = DataLoader(train_res, **common)
        va_loader = DataLoader(val_res, **common)
        extractor = build_frozen_resnet()
        print("extracting ResNet-18 features")
        tr_f, tr_y = extract_features(extractor, tr_loader, device=device)
        va_f, va_y = extract_features(extractor, va_loader, device=device)
        raw = evaluate_features(tr_f, tr_y, va_f, va_y, n_neighbors=5)
        tr64, va64 = reduce_dimensions(tr_f, va_f, n_components=64)
        red = evaluate_features(tr64, tr_y, va64, va_y, n_neighbors=5)
        metrics["resnet"] = {
            "width_512": {"accuracy": raw["accuracy"], "silhouette": raw["silhouette"]},
            "width_64": {"accuracy": red["accuracy"], "silhouette": red["silhouette"]},
            "report_512": raw["report"],
            "report_64": red["report"],
        }
        print(f"ResNet 512  acc {raw['accuracy']:.4f}  sil {raw['silhouette']:.4f}")
        print(f"ResNet  64  acc {red['accuracy']:.4f}  sil {red['silhouette']:.4f}")
        print(f"AE d64      acc {EX2_KNN_ACC:.4f}  sil {EX2_SIL:.4f}")

        P = PCA(n_components=2).fit_transform(va_f)
        T = TSNE(n_components=2, random_state=42).fit_transform(va_f)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        names = ds.classes
        _scatter(axes[0], P, va_y, names, "PCA of ResNet-18 val features")
        _scatter(axes[1], T, va_y, names, "t-SNE of ResNet-18 val features")
        fig.tight_layout()
        fig.savefig(out / "resnet_pca_tsne.png", dpi=140)
        plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
