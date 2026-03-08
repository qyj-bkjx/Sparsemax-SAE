#! /bin/bash

PYTHONPATH=./ python tasks/classification_with_top_k_masking.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path  path to sae model \
    --cls_wise_sae_activation_path path to cnt file \
    --vit_type base \
    --device cuda:0 \
    --batch_size 32 \
    --save_name  path to save results \

