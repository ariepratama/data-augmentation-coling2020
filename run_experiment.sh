#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=clu-ling
#SBATCH --mail-user=ariesutiono@email.arizona.edu
#SBATCH --mail-type=ALL

echo "data dir: ${DATA_DIR}"
echo "log dir: ${LOG_DIR}"
echo "result dir: ${RESULT_DIR}"
echo "data size: ${DATA_SIZE}"
echo "run datetime: ${RUN_DATETIME}"
echo "replaced non terminal: ${REPLACED_NON_TERMINAL}"
echo "num generated sentences: ${NUM_GENERATED_SENTENCES}"
echo "n replaced non terminal: ${N_REPLACED_NON_TERMINAL}"
echo "augmentation: ${AUGMENTATION}"
echo "seed: ${RANDOM_SEED}"
echo "pretrined_dir: ${PRETRAINED_DIR}"
export OUTPUT_FILE="${DATA_SIZE}-${AUGMENTATION}-${NUM_GENERATED_SENTENCES}-${REPLACED_NON_TERMINAL}-${N_REPLACED_NON_TERMINAL}-${RANDOM_SEED}-${RUN_DATETIME}"
export OUTPUT_DIR="/home/u26/ariesutiono/da-coling2020-run/out/${OUTPUT_FILE}"
mkdir -p ${OUTPUT_DIR}
echo "storing output of models at ${OUTPUT_DIR}"


module load anaconda/2020
conda init bash
source ~/.bashrc
conda activate py38
echo "running main script..."

python main.py \
--data_folder ${DATA_DIR}/${DATA_SIZE}-sent \
--embedding_type bert \
--pretrained_dir allenai/scibert_scivocab_cased \
--result_filepath ${RESULT_DIR}/sent-${OUTPUT_FILE}.log \
--augmentation ${AUGMENTATION} \
--replaced_non_terminal ${REPLACED_NON_TERMINAL} \
--num_generated_samples ${NUM_GENERATED_SENTENCES} \
--n_replaced_non_terminal ${N_REPLACED_NON_TERMINAL} \
--seed ${RANDOM_SEED} \
--output_dir ${OUTPUT_DIR} \
--log_filepath ${OUTPUT_DIR}/development.log