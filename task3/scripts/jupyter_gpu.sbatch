#!/bin/bash
#SBATCH --job-name=jupyter-gpu
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/dldevel/momeni/logs/jupyter-%j.out

source /work/dldevel/momeni/miniconda3/etc/profile.d/conda.sh
conda activate bachelor

PORT=$(shuf -i 8000-9999 -n 1)
echo "==============================================="
echo "node : $(hostname)"
echo "port : $PORT"
echo "tunnel from your laptop:"
echo "  ssh -N -L ${PORT}:$(hostname):${PORT} ${USER}@<login-node>"
echo "then open the http://127.0.0.1:${PORT}/... URL printed below"
echo "==============================================="

cd /work/dldevel/momeni
jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT}
