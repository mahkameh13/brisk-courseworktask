# Exercise 3 + Bonus 4 answers

## Exercise 2 anchors

| Quantity | Value |
|---|---|
| d16 best val MSE / pixel | 0.00745 |
| d64 best val MSE / pixel | 0.00610 |
| d64 as sum over pixels | 0.00610 × 16384 ≈ 99.9 |
| 5-NN on d64 codes | 0.8750 |
| Silhouette on d64 codes | 0.0128 |

VAE recon in `vae_loss` is summed over 16384 pixels — divide by 16384 to compare with 0.00610. A constrained VAE should not beat the unconstrained AE.

## What KL does

KL pushes each image’s posterior toward N(0, I): means near 0, variances near 1, no collapse to a delta. Sampling + overlapping Gaussians fill gaps between AE point codes, so interpolations look less like average-blur.

## Beta

- Small beta (~0.01): closer to AE (sharper recon, tighter latents).
- Large beta (~1): stronger prior (smoother latent, worse recon). Watch mean |μ| and σ for posterior collapse.
- Sheet asks for the comparison, not one winner (`history_b001.json`, `history_b1.json`).

## Generated MRI

Prior samples / interpolations are decoder outputs, not patients. Do not use as medical evidence or labels.

## Bonus 4

Same split and k=5; only the feature extractor changes.

| Representation | 5-NN | Silhouette |
|---|---|---|
| AE d64 | 0.875 | 0.0128 |
| ResNet-18 frozen (512) | see `metrics.json` | see `metrics.json` |
| ResNet → 64 (PCA) | see `metrics.json` | see `metrics.json` |

If ResNet kNN is clearly higher, classes are separable and the AE/VAE limit is the objective (recon ≠ classification). ImageNet mid-level filters (edges, mass, texture) still respond on MRI even without MRI training.
