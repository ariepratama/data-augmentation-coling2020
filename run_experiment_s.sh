#!/bin/bash
#SBATCH --job-name=da-coling2020-s
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --nodes=1
#SBATCH --output=/home/u26/ariesutiono/da-coling2020/logs/run_experiment_s.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ariesutiono@email.arizona.edu
#SBATCH --account=clu-ling
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module load anaconda/2020
conda init bash
source ~/.bashrc
conda activate py38
python main.py \
--data_folder /home/u26/ariesutiono/da-coling2020/data/S-sent \
--embedding_type bert \
--pretrained_dir allenai/scibert_scivocab_cased \
--result_filepath /home/u26/ariesutiono/da-coling2020/logs/S-sent.log