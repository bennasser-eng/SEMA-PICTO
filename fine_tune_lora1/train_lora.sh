#!/bin/bash

source ../picto/env/bin/activate

python train_lora.py \
    --train_data_dir ./dataset_picto/dataset_final/train \
    --val_data_dir ./dataset_picto/dataset_final/val \
    --test_data_dir ./dataset_picto/dataset_final/test \
    --output_dir ./epochs500_rank64_alpha64_lamda5_loss_avec_-score \
    --batch_size 16 \
    --num_epochs 500 \
    --lambda_reward 5.0 \
    --kid_samples 200
