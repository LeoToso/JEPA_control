#!/bin/bash
set -e

echo "=== Deleting old dataset ==="
rm -rf data/cartpole_visual_fs5_near_eq_v8

echo "=== Generating dataset ==="
python experiments/generate_cartpole_dataset.py \
    --output-dir      data/cartpole_visual_fs5_near_eq_v8 \
    --num-transitions 500000 \
    --image-size 64 \
    --frame-skip 5 \
    --seed 3 \
    --use-continuous-env \
    --theta-threshold 5.0 \
    --frac-passive 0.5 \
    --frac-random 0.5 \
    --pole-angle-range 0.05

echo "=== Starting v8 training ==="
python experiments/train_jepa_pred_inv_random.py \
    --data             data/cartpole_visual_fs5_near_eq_v8 \
    --extra-data       data/cartpole_visual_fs5_passive_long \
    --config           configs/cartpole_jepa_sf_w3_fs5_finetune_v8.yaml \
    --init-checkpoint  results/jepa_sf_w3_fs5_finetune_v5/checkpoints/checkpoint_epoch0010.pt \
    --save-dir         results/jepa_sf_w3_fs5_finetune_v8 \
    --epochs 150 \
    --device cuda:1
