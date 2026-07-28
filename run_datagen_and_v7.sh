#!/bin/bash
set -e

echo "=== Deleting old dataset ==="
rm -rf data/cartpole_visual_fs5_near_eq_longhorizon

echo "=== Generating dataset ==="
python experiments/generate_cartpole_dataset.py \
    --output-dir      data/cartpole_visual_fs5_near_eq_longhorizon \
    --num-transitions 500000 \
    --image-size 64 \
    --frame-skip 5 \
    --seed 2 \
    --use-continuous-env \
    --theta-threshold 5.0 \
    --frac-passive 1.0 \
    --pole-angle-range 0.002

echo "=== Starting v7 training ==="
python experiments/train_jepa_pred_inv_random.py \
    --data             data/cartpole_visual_fs5_near_eq_longhorizon \
    --extra-data       data/cartpole_visual_fs5_passive_long \
    --config           configs/cartpole_jepa_sf_w3_fs5_finetune.yaml \
    --init-checkpoint  results/jepa_sf_w3_fs5_finetune_v5/checkpoints/checkpoint_epoch0010.pt \
    --save-dir         results/jepa_sf_w3_fs5_finetune_v7 \
    --epochs 150 \
    --device cuda:1
