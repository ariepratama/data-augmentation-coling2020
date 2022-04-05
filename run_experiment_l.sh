#!/bin/bash
#SBATCH --job-name=da-coling2020-l
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --nodes=1
#SBATCH --output=run_experiment_l.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ariesutiono@email.arizona.edu
#SBATCH --account=clu-ling
#SBATCH --gres=gpu:1

module load anaconda/2020
conda init bash
source ~/.bashrc
conda activate py38
python main.py \
--data_folder /home/u26/ariesutiono/da-coling2020/data/L-sent \
--embedding_type bert \
--pretrained_dir allenai/scibert_scivocab_cased \
--result_filepath /home/u26/ariesutiono/da-coling2020/L-sent.log
