#!/bin/bash
#SBATCH --time=336:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400GB
#SBATCH --gpus=A100:5
cd data/yunche/master_thesis
module load mamba
source activate my_env
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python src/train/train_charbert_small.py --model_type roberta --model_name_or_path roberta-base --do_train --do_eval --train_data_file data/wiki/enwiki_train.txt --eval_data_file data/wiki/enwiki_val.txt --term_vocab src/models/charbert/data/dict/term_vocab --learning_rate 3e-5 --num_train_epochs 3 --char_vocab src/models/charbert/data/dict/roberta_char_vocab --mlm_probability 0.10 --input_nraws 1000 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --save_steps 10000 --block_size 384 --overwrite_output_dir --mlm --output_dir results/charbert_small/