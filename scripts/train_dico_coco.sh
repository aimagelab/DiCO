#!/bin/bash

conda activate dico

export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=5625

# PAC-S Reference repo: https://github.com/aimagelab/pacscore
# Download from https://drive.google.com/drive/folders/15Da_nh7CYv8xfryIdETG6dPFSqcBiqpd?usp=sharing
PACS_checkpoint=clip_ViT-B-32.pth

# With 2 GPUs we evaluate every quarter of epoch
eval_steps=1770
save_steps=1770

max_steps=250000
train_batch_size=8
eval_batch_size=8
resume_from_checkpoint=/path/to/xe/checkpoint-1234

torchrun --nproc_per_node ${WORLD_SIZE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} main.py \
--do_train \
--dico \
--pacs_checkpoint ${PACS_checkpoint} \
--output_dir results/DiCO \
--run_name  DiCO \
--train_dataset coco_training_dico \
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
--learning_rate 1e-6 \
--lr_scheduler_type constant \
--gradient_accumulation_steps 1 \
--metric_for_best_model ref-PACScore \
--load_best_model_at_end \
--save_total_limit 2 \
--dataloader_num_workers 2 \
--predict_with_generate \
--resume_from_checkpoint ${resume_from_checkpoint} \
--report_to wandb \