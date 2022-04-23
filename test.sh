python main.py \
--data_folder data/S-sent \
--embedding_type bert \
--pretrained_dir allenai/scibert_scivocab_cased \
--train_bs 1 \
--eval_bs 1 \
--result_filepath S-sent.log \
--augmentation SYN