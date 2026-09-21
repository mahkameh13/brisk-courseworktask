# Task 3 + Bonus 4 results

Job 1770494 completed (exit 0) in 1:06:05 on gpu2 / A100. Tests 59/59. Numbers: `results/metrics.json`.

## Reconstruction (AE units)

Divide VAE recon by 16384 before comparing to 0.00610.

| Model | beta | val recon (sum) | / pixel | vs Ex2 d64 |
|---|---|---|---|---|
| AE d64 | — | 99.9 | 0.00610 | baseline |
| VAE d64 | 0.01 | 107.44 | 0.00656 | +7.5% |
| VAE d64 | 1.0 | 138.19 | 0.00843 | +38% |

## Collapse check (last epoch)

| Run | mean \|μ\| | mean σ |
|---|---|---|
| beta 0.01 | 0.837 | 0.029 |
| beta 1.0 | 0.377 | 0.726 |

## Figures

- `results/interpolate_vae.png` — glioma → pituitary (val 5→7)
- `results/prior_samples.png` — z ~ N(0,I)
- `results/resnet_pca_tsne.png` — Bonus 4

## Bonus 4 (frozen ResNet-18, k=5)

| Representation | width | 5-NN | silhouette |
|---|---|---|---|
| AE d64 | 64 | 0.875 | 0.0128 |
| ResNet-18 ImageNet | 512 | 0.904 | 0.0299 |
| ResNet → PCA | 64 | 0.900 | 0.0400 |

ResNet ≈ 0.90: classes are in the pixels; AE/VAE gap is the training objective, not missing signal. PCA to 64 barely drops accuracy.
