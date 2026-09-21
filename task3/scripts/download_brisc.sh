#!/bin/bash
# Fetch BRISC2025 onto the cluster. Run on the LOGIN node (compute nodes have no
# internet). Needs ~/.kaggle/kaggle.json with your API token, chmod 600.
set -euo pipefail

source /work/dldevel/momeni/miniconda3/etc/profile.d/conda.sh
conda activate bachelor

DEST=/work/dldevel/momeni/datasets
mkdir -p "$DEST"

kaggle datasets download -d briscdataset/brisc2025 -p "$DEST" --unzip

echo "--- layout ---"
find "$DEST" -maxdepth 4 -type d | head -30
