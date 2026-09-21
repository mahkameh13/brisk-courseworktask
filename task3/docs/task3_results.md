# Task 3 and Bonus 4 results

Job **1770494** finished **COMPLETED** (exit 0) in **1:06:05** on `gpu2` / A100.  
Tests: **59/59** passed.  
Raw numbers: `outputs/task3/metrics.json`

## Reconstruction (same units as Exercise 2)

Divide VAE `recon` by 16384 before comparing to 0.00610.

| Model | beta | val recon (sum pixels) | val recon / pixel | vs Ex2 d64 |
|---|---|---|---|---|
| AE d64 (Exercise 2) | — | 99.9 | **0.00610** | baseline |
| VAE d64 | 0.01 | 107.44 | **0.00656** | +7.5% worse |
| VAE d64 | 1.0 | 138.19 | **0.00843** | +38% worse |

Constraint cannot beat unconstrained MSE. Both VAEs are worse than 0.00610, as required.

## Collapse check (last epoch)

| Run | mean \|mu\| | mean sigma | reading |
|---|---|---|---|
| beta 0.01 | 0.837 | 0.029 | clouds tight; close to a plain AE |
| beta 1.0 | 0.377 | 0.726 | stronger prior; not collapsed to one average brain |

Histories: `outputs/task3/history_b001.json`, `outputs/task3/history_b1.json`

## Figures (Task 3 proofs)

| File | What |
|---|---|
| `outputs/task3/interpolate_vae.png` | 9-step glioma → pituitary (val indices 5 → 7, posterior means) |
| `outputs/task3/prior_samples.png` | 8 scans from `z ~ N(0,I)`, no input image |
| `outputs/task3/best_b001.pt` / `best_b1.pt` | checkpoints |

## Bonus 4 — frozen ResNet-18, same split, k=5

| Representation | width | 5-NN acc | silhouette |
|---|---|---|---|
| AE d64 codes (Exercise 2) | 64 | 0.875 | 0.0128 |
| ResNet-18 ImageNet, frozen | 512 | **0.904** | 0.0299 |
| ResNet-18 → PCA (train-fit only) | 64 | **0.900** | 0.0400 |

Plot: `outputs/task3/resnet_pca_tsne.png`

ResNet at 0.90 matches the sheet’s “separates well (~0.90)” case: the four classes **are** in the pixels. The AE/VAE gap is the objective (reconstruction ≠ classification), not a dead dataset.

PCA to 64 barely drops accuracy (0.904 → 0.900), so the AE’s 0.875 is not “only 64 dimensions.”

## What to put in the write-up

1. **KL:** penalises off-centre means, too-wide clouds, and collapse-to-a-point. Sampling + overlapping clouds fill the holes the AE left between codes.
2. **Beta:** 0.01 ≈ AE recon (0.00656 vs 0.00610) and tiny sigma. 1.0 pays 0.00843 recon for a prior-shaped latent (sigma 0.73).
3. **Generated scans:** decoder toys, not patients. Do not use as medical data.
4. **ResNet vs AE:** ImageNet features win (0.90 vs 0.875) because mid-level filters still fire on mass/texture. Neither loss said “separate BRISC types”; class geometry is a side effect.

## Paths

```
/work/dldevel/momeni/task3_results.md          ← this file
/work/dldevel/momeni/task3_answers.md          ← question prose
/work/dldevel/momeni/outputs/task3/metrics.json
/work/dldevel/momeni/outputs/task3/*.png
/work/dldevel/momeni/logs/task3-1770494.out
```
