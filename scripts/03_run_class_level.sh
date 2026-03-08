#! /bin/bash

PYTHONPATH=./  python -u tasks/compute_class_wise_sae_activation.py \
    --root_dir ./ \
    --dataset_name eurosat \
    --sae_path path to sae model \
    --vit_type base \
    --savename outoursaens49152 \
    --device cuda:0 \
    --model_path /root/jsz/clip \


