#!/bin/bash

#SBATCH --job-name=da-coling2020-l
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --nodes=1
#SBATCH --output=/home/u26/ariesutiono/da-coling2020/logs/run_experiment_l.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ariesutiono@email.arizona.edu
#SBATCH --account=clu-ling
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
export SIZE=$1
export RUN_DATETIME=$2
export REPLACED_NON_TERMINAL=$3
export NUM_GENERATED_SENTENCES=$4
export DATA_DIR=/home/u26/ariesutiono/da-coling2020/data
export LOG_DIR=/home/u26/ariesutiono/da-coling2020/logs

sbatch run_experiment.sh --job-name=arie-da-coling2020-${SIZE} \
--output=/home/u26/ariesutiono/${SIZE}-${RUN_DATETIME}.sbatch.log \
--time=00:30:00 \
--partition=standard \
--account=clu-ling \
--mail-type=ALL \
--mail-user=ariesutiono@email.arizona.edu \
--ntasks=1 \
--cpus-per-task=1 \
--mem=8GB \
--export=DATA_DIR=${DATA_DIR},LOG_DIR=${LOG_DIR},DATA_SIZE=${DATA_SIZE},RUN_DATETIME=${RUN_DATETIME},REPLACED_NON_TERMINAL=${REPLACED_NON_TERMINAL},NUM_GENERATED_SENTENCES=${NUM_GENERATED_SENTENCES}