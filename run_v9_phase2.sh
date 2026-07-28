#!/bin/bash
set -e

echo "=== Starting v9 phase 2 (predictor-only, encoder+action frozen) ==="
python experiments/train_jepa_pred_inv_random.py \
    --data             data/cartpole_visual_fs5_passive_long \
    --config           configs/cartpole_jepa_sf_w3_fs5_v9_phase2.yaml \
    --init-checkpoint  results/jepa_sf_w3_fs5_v9/checkpoints/checkpoint_epoch0100.pt \
    --save-dir         results/jepa_sf_w3_fs5_v9_phase2 \
    --epochs 300 \
    --device cuda:2
