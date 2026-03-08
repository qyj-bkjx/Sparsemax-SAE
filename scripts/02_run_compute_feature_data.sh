#! /bin/bash

PYTHONPATH=./  python -u tasks/compute_sae_feature_data.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path path to sae model \
    --vit_type base \
    --device cuda:0


