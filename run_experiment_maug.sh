#!/bin/bash
#SBATCH --job-name=da-coling2020-maug
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --nodes=1
#SBATCH --output=/home/u26/ariesutiono/da-coling2020/logs/run_experiment_maug.log
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
--data_folder data/M-sent \
--embedding_type bert \
--pretrained_dir allenai/scibert_scivocab_cased \
--result_filepath logs/M-sent-maug.log \
--augmentation GR \
--replaced_non_terminal NP \
--num_generated_samples 5 \

