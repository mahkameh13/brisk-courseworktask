# Task 3 — ConvVAE on BRISC (+ Bonus ResNet)

Bachelor coursework (Goethe University): variational autoencoder on BRISC MRI classification images, with Bonus 4 frozen ImageNet ResNet-18 feature comparison.

## Layout

- `notebooks/` — `my_task3.ipynb` and bonus notebook
- `src/` — ConvVAE, training driver, ResNet bonus helpers, BRISC loader
- `scripts/` — Slurm GPU jobs and data download
- `docs/` — results write-up and sheet answers
- `results/` — metrics, histories, interpolation / prior / ResNet figures
- `tests/` — automated checks

## Note

Large checkpoints (`*.pt`) and the raw dataset are not included. Code and figures live under `/work/dldevel/momeni` on the Goethe cluster.
