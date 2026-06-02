#!/bin/bash
#SBATCH --job-name=half_dac_pipeline
#SBATCH --partition=ifn
##SBATCH --gres=gpu:a100_80gb:1
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu01               # <--- FORCE SLURM TO SKIP THE BROKEN NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=192:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Clean path configurations 
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH

cd /beegfs/work_fast/singh/Half_Dac_Transformer
source /beegfs/work_fast/singh/dac_clean/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Running on Node: $(hostname) ==="
python -c "import torch; torch.cuda.init(); print('CUDA Functional?:', torch.cuda.is_available())"



nvidia-smi




echo 'Starting training...'
python train.py

echo 'Starting inference...'
python inference.py

echo 'Starting evaluation...'
python evaluation.py