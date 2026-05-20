#!/usr/bin/env bash
# Overnight training pipeline — runs two experiments sequentially.
#
#   Exp A (baseline):   cartpole_v2.yaml         — unstable-mode spectral only
#   Exp B (fullspec):   cartpole_v2_fullspec.yaml — unstable + marginal modes
#
# Both use the same dataset (generated once), warm-start CEM, and 100 epochs.
# Results land in results/<exp_name>/<timestamp>/.
#
# Usage:
#   nohup bash run_overnight.sh > /tmp/overnight.log 2>&1 &

set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
LOG_DIR=/tmp/jepa_overnight
mkdir -p "$LOG_DIR"

echo "========================================================"
echo " JEPA overnight pipeline — $(date)"
echo "========================================================"

# ── Experiment A: baseline (cartpole_v2.yaml) ───────────────────────────────
echo ""
echo "[overnight] === Exp A: baseline (unstable spectral only) ==="
echo "[overnight] Start: $(date)"
$PYTHON experiments/run_experiment.py \
    --variant E-full \
    --dataset mixed \
    --seed 42 \
    --config configs/cartpole_v2.yaml \
    --force \
    2>&1 | tee "$LOG_DIR/expA_baseline.log"
echo "[overnight] Exp A done: $(date)"

# ── Experiment B: full spectral (cartpole_v2_fullspec.yaml) ─────────────────
echo ""
echo "[overnight] === Exp B: full spectral (unstable + marginal modes) ==="
echo "[overnight] Start: $(date)"
$PYTHON experiments/run_experiment.py \
    --variant E-full \
    --dataset mixed \
    --seed 42 \
    --config configs/cartpole_v2_fullspec.yaml \
    --force \
    2>&1 | tee "$LOG_DIR/expB_fullspec.log"
echo "[overnight] Exp B done: $(date)"

echo ""
echo "========================================================"
echo " All experiments complete — $(date)"
echo "========================================================"
echo "Logs in: $LOG_DIR"
echo "Results in: results/"
