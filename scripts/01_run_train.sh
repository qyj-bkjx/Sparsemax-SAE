#! /bin/bash

PYTHONPATH=./ python -u tasks/train_sae_vit.py \
  --batch_size 32 \
  --checkpoint_path path to sae checkpoints \
  --dataset_name imagenet \
  --n_checkpoints 10 \
  --use_ghost_grads \
  --gpu 7
