#!/bin/bash

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate dico

export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=5626

# PAC-S Reference repo: https://github.com/aimagelab/pacscore
# Download from https://drive.google.com/drive/folders/15Da_nh7CYv8xfryIdETG6dPFSqcBiqpd?usp=sharing
PACS_checkpoint=clip_ViT-B-32.pth

# DiCO Reference repo coming soon
DiCO_checkpoint=dico-ViTL14

torchrun --nproc_per_node ${WORLD_SIZE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} main.py \
--do_predict \
--output_dir results/inference  \
--run_name coco_inference \
--test_dataset coco_test \
--logging_steps 10  \
--remove_unused_columns False  \
--generation_max_length 30 \
--generation_num_beams 5 \
--per_device_eval_batch_size 16 \
--fp16 --fp16_full_eval  \
--predict_with_generate \
--pacs_checkpoint ${PACS_checkpoint} \
--resume_from_checkpoint ${DiCO_checkpoint}