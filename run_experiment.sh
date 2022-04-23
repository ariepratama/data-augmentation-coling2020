#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=clu-ling
#SBATCH --mail-user=ariesutiono@email.arizona.edu
#SBATCH --mail-type=ALL

echo "data dir: ${DATA_DIR}"
echo "log dir: ${LOG_DIR}"
echo "data size: ${DATA_SIZE}"
echo "run datetime: ${RUN_DATETIME}"
echo "repalced non terminal: ${REPLACED_NON_TERMINAL}"
echo "num generated sentences: ${NUM_GENERATED_SENTENCES}"
echo "augmentation: ${AUGMENTATION}"


module load anaconda/2020
conda init bash
source ~/.bashrc
conda activate py38
echo "running main script..."

python main.py \
--data_folder ${DATA_DIR}/${DATA_SIZE}-sent \
--embedding_type bert \
--pretrained_dir allenai/scibert_scivocab_cased \
--result_filepath ${LOG_DIR}/${DATA_SIZE}-sent-${AUGMENTATION}-${RUN_DATETIME}.log \
--augmentation ${AUGMENTATION} \
--replaced_non_terminal ${REPLACED_NON_TERMINAL} \
--num_generated_samples ${NUM_GENERATED_SENTENCES}