#!/bin/bash
#SBATCH --job-name=vae-task3
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/dldevel/momeni/logs/task3-%j.out
#SBATCH --error=/work/dldevel/momeni/logs/task3-%j.err

source /work/dldevel/momeni/miniconda3/etc/profile.d/conda.sh
conda activate bachelor-cu

cd /work/dldevel/momeni
export BRISC_ROOT=/work/dldevel/momeni/datasets/brisc2025
mkdir -p logs outputs/task3

python -c "import torch; print('gpu:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python brisc.py
python tests/test_task3.py
python run_task3.py --epochs 25
