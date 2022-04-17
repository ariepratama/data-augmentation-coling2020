#!/bin/bash

module load anaconda/2020
conda init bash
source ~/.bashrc
conda activate py38
python main.py \
--data_folder ${DATA_DIR}/${DATA_SIZE}-sent \
--embedding_type bert \
--pretrained_dir allenai/scibert_scivocab_cased \
--result_filepath ${LOG_DIR}/${DATA_SIZE}-sent-${RUN_DATETIME}.log \
--augmentation GR \
--replaced_non_terminal ${REPLACED_NON_TERMINAL} \
--num_generated_samples ${NUM_GENERATED_SENTENCES}