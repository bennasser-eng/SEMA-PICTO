#!/bin/bash

source ../picto/env/bin/activate

python train_lora.py \
    --train_data_dir ./dataset_picto/train \
    --val_data_dir ./dataset_picto/val \
    --test_data_dir ./dataset_picto/test \
    --output_dir ./lora_picto_model \
    --batch_size 4 \
    --num_epochs 10 \
    --lambda_reward 10.0
