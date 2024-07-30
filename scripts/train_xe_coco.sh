#!/bin/bash

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate dico

export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=5625

# With 2 GPUs we evaluate every quarter of epoch
eval_steps=200
save_steps=200

max_steps=10000
train_batch_size=64
eval_batch_size=64

torchrun --nproc_per_node ${WORLD_SIZE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} main.py \
--do_train \
--output_dir results/XE \
--run_name  XE \
--train_dataset coco_training_xe \
--validation_dataset coco_validation \
--test_dataset coco_test \
--evaluation_strategy steps \
--eval_steps ${eval_steps} \
--save_steps ${save_steps} \
--max_steps ${max_steps} \
--logging_steps 5 \
--remove_unused_columns False \
--ignore_data_skip \
--generation_max_length 30 \
--generation_num_beams 5 \
--per_device_train_batch_size ${train_batch_size} \
--per_device_eval_batch_size ${eval_batch_size} \
--fp16 --fp16_full_eval \
--deepspeed configs/config_adam_zero2.json \
--learning_rate 2.5e-4 \
--lr_scheduler_type constant_with_warmup \
--warmup_ratio 0.05 \
--gradient_accumulation_steps 8 \
--metric_for_best_model cider \
--load_best_model_at_end \
--save_total_limit 2 \
--dataloader_num_workers 2 \
--predict_with_generate \
--report_to wandb