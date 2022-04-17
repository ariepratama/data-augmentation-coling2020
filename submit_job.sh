#!/bin/bash

export SIZE=$1
export RUN_DATETIME=$2
export REPLACED_NON_TERMINAL=$3
export NUM_GENERATED_SENTENCES=$4
export DATA_DIR=/home/u26/ariesutiono/da-coling2020/data
export LOG_DIR=/home/u26/ariesutiono/da-coling2020/logs

sbatch --job-name=da-coling2020-${SIZE}-${RUN_DATETIME} \
--output=/home/u26/ariesutiono/da-coling2020/logs/${SIZE}-${RUN_DATETIME}.sbatch.log \
--time=00:30:00 \
--partition=standard \
--account=clu-ling \
--mail-type=ALL \
--mail-user=ariesutiono@email.arizona.edu \
--ntasks=1 \
--cpus-per-task=1 \
--gres=gpu:1 \
--mem=8GB \
--export=DATA_DIR=${DATA_DIR},LOG_DIR=${LOG_DIR},DATA_SIZE=${SIZE},RUN_DATETIME=${RUN_DATETIME},REPLACED_NON_TERMINAL=${REPLACED_NON_TERMINAL},NUM_GENERATED_SENTENCES=${NUM_GENERATED_SENTENCES} \
run_experiment.sh