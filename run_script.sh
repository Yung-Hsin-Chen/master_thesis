#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --gpus=A100:1
cd data/yunche/master_thesis
module load mamba
source activate my_env
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit
export CUDA_VISIBLE_DEVICES=0
python ./src/main.py