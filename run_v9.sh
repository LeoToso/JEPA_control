#!/bin/bash
set -e

echo "=== Starting v9 training (from scratch, lambda_fp=0.1) ==="
python experiments/train_jepa_pred_inv_random.py \
    --data             data/cartpole_visual_fs5_passive_long \
    --extra-data       data/cartpole_visual_fs5_near_eq_v8 \
    --config           configs/cartpole_jepa_sf_w3_fs5_v9.yaml \
    --save-dir         results/jepa_sf_w3_fs5_v9 \
    --epochs 500 \
    --device cuda:2
