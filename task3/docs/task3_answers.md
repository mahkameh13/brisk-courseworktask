# Exercise 3 and Bonus 4 write-up

Exercise 2 anchors (from `notebook7bea9fcea6.ipynb`):

| Quantity | Value |
|---|---|
| d16 best val MSE (per pixel) | 0.00745 |
| d64 best val MSE (per pixel) | 0.00610 |
| d64 in sum-over-pixels units | 0.00610 × 16384 = 99.9 |
| 5-NN accuracy on d64 codes | 0.8750 |
| Silhouette on d64 codes | 0.0128 |

VAE `vae_loss` reconstruction is a **sum over 16384 pixels**. Divide by 16384
before sitting it next to 0.00610. After a GPU run, read the converted
number from `outputs/task3/metrics.json` (`val_recon_per_pixel`). Expect
it **above** 0.00610: a constraint cannot beat the unconstrained MSE.

## Exercise 3 questions

### What the KL term penalises, and how that removes the holes

KL is the closed-form penalty of each image's posterior Gaussian against
N(0, I). It charges three things: drifting the mean off the origin,
spreading the cloud too wide, and shrinking it to a point. The last one
stops the VAE from collapsing back into a plain autoencoder.

A plain AE stores each image as an isolated point. Nothing in MSE trains
the decoder on the space between points, so interpolation is a blurry
average. KL plus sampling turns each point into a cloud. Clouds overlap,
the decoder sees neighbourhoods, and the holes close. The **aggregate**
posterior has to cover the prior; individual clouds can stay small.

### What beta trades off

Beta is the exchange rate between reconstruction and that penalty.

- **beta near 0**: the model behaves like the Exercise 2 AE. Reconstructions
  sharpen, latent codes scatter, interpolation middles go blurry again.
- **beta very large**: the posterior is forced onto N(0, I). Mean |mu|
  drops, sigma heads toward 1, every reconstruction looks like the same
  average brain. That is posterior collapse. Loss still falls, because
  both halves of the objective look like progress. Log `|mu|` and `sigma`
  or you will miss it.
- **beta = 1 vs 0.01**: the sheet wants the comparison, not a single
  champion. Use `history_b001.json` and `history_b1.json`.

### Risks of treating generated scans as real medical data

Prior samples and interpolation frames are decoder outputs from a
64-dimensional Gaussian fitted to this dataset. They are not
acquisitions, they have no patient, they can invent tumours or erase
them, and they copy dataset artefacts. Using them as radiology evidence,
training labels, or "synthetic patients" is unsafe. They are figures for
this assignment only.

## Bonus 4

Hold everything else fixed: same split indices, same k=5, same metric.
Change only the extractor.

| Representation | What it saw | What to read |
|---|---|---|
| AE d64 codes | Unlabelled 128×128 MRI, MSE | 5-NN 0.8750, silhouette 0.0128 (local neighbourhoods, no blobs) |
| ResNet-18 512 | ImageNet labels, frozen | see `metrics.json` after the GPU job |
| ResNet-18 → 64 | Same, width matched to d | see `metrics.json` |

If ResNet kNN is clearly above the AE (the sheet's "say 0.90"), the
four classes **are** separable at this resolution and the AE/VAE
objective is the limit: reconstruction is not classification. If ResNet
is also poor (the sheet's "say 0.70"), the signal is weak at 128×128
and no unsupervised method on these pixels will rescue it.

### Why ImageNet features still organise brain scans by tumour type

ResNet-18 never saw an MRI. It did see edges, blobs, texture, and
left/right mass on millions of natural images. Tumour type in BRISC
shows up as location, size, and texture of a mass on a skull-shaped
background. Those mid-level filters still fire. Transfer is not
"it knows glioma"; it is "the same visual factors that define the
classes are factors ImageNet training already had to represent".

### Neither objective was "separate BRISC tumour types"

MSE keeps variation that is large in pixel space. ImageNet training
keeps variation that was useful for 1000 object classes. Tumour type
correlates with both: a large bright mass is both high-MSE structure
and a blob-like object. Geometry in the codes is a side effect, which
is why 5-NN can work while silhouette stays near zero for the AE
(ribbons, not islands). Bonus 4 exists to tell apart "the model is
weak" from "the images do not carry the class".
