#!/bin/bash
#SBATCH --job-name=jupyter-gpu2
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/dldevel/momeni/logs/jupyter-gpu2-%j.out

source /work/dldevel/momeni/miniconda3/etc/profile.d/conda.sh
conda activate bachelor-cu

PORT=$(shuf -i 8000-9999 -n 1)
echo "==============================================="
echo "node : $(hostname)"
echo "port : $PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "tunnel from your laptop:"
echo "  ssh -N -L ${PORT}:$(hostname):${PORT} ${USER}@login01.hhlr-gu.hhlr.de"
echo "then open the http://127.0.0.1:${PORT}/... URL printed below"
echo "==============================================="

cd /work/dldevel/momeni
jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT}
