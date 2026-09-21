# JEPA World Models for Visual Control

This repository implements and evaluates **Joint Embedding Predictive Architecture (JEPA) world models** for visual control tasks. We build on the **Sensorimotor World Model (SMWM)** architecture from [Ivashkov et al.](https://github.com/petr-ivashkov/sensorimotor-world-model) and compare different visual encoders across three control benchmarks: CartPole, PointMaze, and Walker2D.

We also compare against **DINO-WM** (Zen et al., 2024), an external baseline using a frozen DINOv2 encoder with a separately learned predictor.

---

## Tasks

### CartPole

Visual pole-balancing. The agent observes **128×128 RGB images** plus a 4D proprioceptive state (cart position, cart velocity, pole angle, pole angular velocity). Actions are scalar forces; **frame skip = 5**.

Dataset: **985 trajectories** (75% open-loop exploration, 25% LQR-stabilized).

**Control methods:**
| Planner | Description |
|---------|-------------|
| **LQR** | Linearize learned latent dynamics at the operating point; solve the discrete-time algebraic Riccati equation. |
| **CEM** | Cross-entropy method over action sequences in the latent space. |
| **GBP** | Directly optimize the action sequence via backpropagation through the differentiable latent rollout, minimizing distance to the goal latent. |

---

### PointMaze

Goal-conditioned navigation in a **U-shaped maze**. The agent observes **64×64 RGB images** (196×196 for DINOv2/iBOT) plus a 4D proprioceptive state (x, y, vx, vy). Each action encodes **5 sub-actions × 2D = 10D** with frame skip = 5.

Dataset: **2000 episodes** collected with a scripted expert policy.

**External baseline:** DINO-WM (Zen et al., 2024) — frozen DINOv2 with a separately trained predictor; reports **98% success** on D4RL U-maze with mujoco_py rendering.

**Control methods:**
| Planner | Description |
|---------|-------------|
| **CEM** | Plan goal-directed trajectories using latent distance-to-goal as cost. |
| **LQR** | Linearize latent dynamics around a nominal trajectory toward the goal. |
| **GBP** | Directly optimize the action sequence via backpropagation through the differentiable latent rollout, minimizing distance to the goal latent. |

---

### Walker2D 

Locomotion control of a **bipedal walker**. The agent observes 84×84 RGB images along with joint proprioception. 


---

## Architecture

The **Sensorimotor World Model (SMWM)** encodes each observation into a latent vector `z_t` and learns a predictor `f(z_t, a_t) → z_{t+1}`. The visual encoder can be:

| Encoder | Type | Description |
|---------|------|-------------|
| **Diff** | Learned | Differentiable patch-difference encoder, trained end-to-end |
| **DINOv2** | Frozen | ViT-S/14 self-supervised features (384D per patch token) |
| **iBOT** | Frozen | ViT-S/16 masked-image-modeling features (384D per patch token) |

Training minimizes a combination of objectives:

| Loss | Symbol | Description |
|------|--------|-------------|
| Forward prediction | L_fwd | One-step MSE in latent space |
| Multi-step rollout | L_rollout | Prediction over H-step horizon |
| Inverse dynamics | L_inverse | Action reconstruction from (z_t, z_{t+1}) |
| SIGReg. | L_sigreg | https://github.com/galilai-group/lejepa |

---

## Installation

### Prerequisites

- Python 3.9+
- MuJoCo (for physics simulation and rendering)
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/LeoToso/JEPA_control.git
cd JEPA_control

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install the package and core dependencies
pip install -e .
pip install -r requirements.txt

# For PointMaze (gymnasium-robotics U-maze)
pip install gymnasium-robotics

# For frozen encoders (DINOv2 / iBOT)
pip install timm
```

### Environment variables

```bash
export DATASET_DIR=/path/to/datasets    # Root for storing generated data
export RESULTS_DIR=/path/to/results     # Root for model checkpoints
```

### DINO-WM baseline *(optional)*

```bash
git clone https://github.com/gaoyuezhou/dino_wm ~/dino_wm
# Follow ~/dino_wm/README.md for environment setup (requires mujoco_py + d4rl)
```

> **Note:** DINO-WM requires a separate conda environment (`py38`) with `mujoco_py`, `d4rl`, and `gym==0.23.1`. The JEPA venv and DINO-WM environments are incompatible and must be run separately.

---

## Data Generation

### CartPole

Data generation produces a flat HDF5 file, then converts it to the split-episode format expected by the trainer.

**Step 1 — collect episodes:**
```bash
python experiments/generate_data.py \
    --config configs/cartpole_jepa_recovery_vit_128_diff_latent64_excitation_depth4.yaml \
    --output data/cartpole_excitation_depth4_fs5_128_flat.h5 \
    --seed 42
```

**Step 2 — convert to split format:**
```bash
python experiments/convert_flat_to_split_hdf5.py \
    --src data/cartpole_excitation_depth4_fs5_128_flat.h5 \
    --out data/cartpole_excitation_depth4_fs5_128
```

The resulting directory `cartpole_excitation_depth4_fs5_128/` is passed as `--data` to `train_sensorimotor.py`.

### PointMaze

```bash
MUJOCO_GL=osmesa python experiments/generate_pointmaze_dataset.py \
    --output-dir data/pointmaze_u_fs5_ma_64 \
    --n-episodes 2000 --episode-steps 100 \
    --image-size 64 --maze-map U --seed 0 \
    --frame-skip 5 --multi-action
```

---

## Training

All models are trained using `train_sensorimotor.py` with a YAML config file specifying the encoder, losses, and hyperparameters.

### CartPole Models

```bash
# 1. SIGReg only (one-step prediction + singular value regularization)
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg.yaml \
    --save-dir $RESULTS_DIR/cartpole/diff_proprio_sigreg \
    --seed 42 --device cuda

# 2. Action reconstruction (one-step prediction + one-step action reconstruction)
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_diff_proprio.yaml \
    --save-dir $RESULTS_DIR/cartpole/diff_proprio_ar \
    --seed 42 --device cuda

# 3. One-step prediction + one-step AR + multi-step AR
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_diff_proprio_ar_1step_ms.yaml \
    --save-dir $RESULTS_DIR/cartpole/diff_proprio_ar_1step_ms \
    --seed 42 --device cuda

# 4. Multi-step prediction + multi-step AR
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_diff_proprio_ar_rollout.yaml \
    --save-dir $RESULTS_DIR/cartpole/diff_proprio_rollout_ms \
    --seed 42 --device cuda

# 5. SIGReg + multi-step prediction
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml \
    --save-dir $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout \
    --seed 42 --device cuda

# 6. Multi-step prediction + multi-step AR + SIGReg
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout_ms.yaml \
    --save-dir $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout_ms \
    --seed 42 --device cuda

# 7. Frozen DINOv2 encoder + multi-step prediction
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_dinov2_rollout.yaml \
    --save-dir $RESULTS_DIR/cartpole/dinov2_rollout \
    --seed 42 --device cuda

# 8. Frozen iBOT encoder + multi-step prediction
python experiments/train_sensorimotor.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_sensorimotor_world_model_ibot_rollout.yaml \
    --save-dir $RESULTS_DIR/cartpole/ibot_rollout \
    --seed 42 --device cuda
```

### PointMaze Models

```bash
# Diff+proprio+rollout (frame_skip=5)
MUJOCO_GL=egl python experiments/train_sensorimotor.py \
    --config configs/pointmaze_smwm_rollout_ms_fs5.yaml \
    --data-dir $DATASET_DIR/pointmaze_umaze_fs5 \
    --save-dir $RESULTS_DIR/pointmaze/rollout_ms_fs5

# Diff+proprio+sigreg+rollout (frame_skip=5)
MUJOCO_GL=egl python experiments/train_sensorimotor.py \
    --config configs/pointmaze_smwm_sigreg_rollout_fs5.yaml \
    --data-dir $DATASET_DIR/pointmaze_umaze_fs5 \
    --save-dir $RESULTS_DIR/pointmaze/sigreg_rollout_fs5

# Frozen DINOv2 encoder + rollout (frame_skip=5)
MUJOCO_GL=egl python experiments/train_sensorimotor.py \
    --config configs/pointmaze_smwm_rollout_dinov2_fs5.yaml \
    --data-dir $DATASET_DIR/pointmaze_umaze_fs5 \
    --save-dir $RESULTS_DIR/pointmaze/dinov2_rollout_fs5

# Frozen iBOT encoder + rollout (frame_skip=5)
MUJOCO_GL=egl python experiments/train_sensorimotor.py \
    --config configs/pointmaze_smwm_rollout_ibot_fs5.yaml \
    --data-dir $DATASET_DIR/pointmaze_umaze_fs5 \
    --save-dir $RESULTS_DIR/pointmaze/ibot_rollout_fs5

# Autoregressive 1-step multi-step (frame_skip=5)
MUJOCO_GL=egl python experiments/train_sensorimotor.py \
    --config configs/pointmaze_smwm_ar_1step_ms_fs5.yaml \
    --data-dir $DATASET_DIR/pointmaze_umaze_fs5 \
    --save-dir $RESULTS_DIR/pointmaze/ar_1step_ms_fs5

# 1-step prediction + 1-step AR only (frame_skip=5)
SAVE=/mnt/t7shield/jepa_results/pointmaze_ar_1step_fs5_200ep_seed42
DATA=data/pointmaze_u_fs5_ma_64
mkdir -p "$SAVE/logs"
MUJOCO_GL=osmesa python experiments/train_sensorimotor.py \
    --data "$DATA" \
    --config configs/pointmaze_smwm_ar_1step_fs5.yaml \
    --save-dir "$SAVE" \
    --seed 42 \
    --device cuda \
    2>&1 | tee "$SAVE/logs/train.log"
```

---

## Evaluation

### CartPole — LQR

LQR protocol: 10 trials, 300 steps, success threshold = 0.7 rad (pole angle), hold = 10 steps.

```bash
# All models evaluated individually — replace --ckpts / --cfgs for each variant
python experiments/compare_lqr_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/<variant>/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_<variant>.yaml \
    --trials 10 \
    --n-steps 300 \
    --success-threshold 0.7 \
    --success-hold-steps 10 \
    --q-scale 1.0 \
    --r-scale 1.0 \
    --device cuda \
    --output results/lqr_<variant>.json
```

Example for each model:

```bash
# 1-step AR
python experiments/compare_lqr_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_ar/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_1sp_AR.json

# 1-step SIGReg
python experiments/compare_lqr_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_sigreg/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_1sp_sigreg.json

# AR-1step+MS
python experiments/compare_lqr_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_ar_1step_ms/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio_ar_1step_ms.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_ms_AR.json

# MS pred + MS AR
python experiments/compare_lqr_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_rollout_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_ar_rollout.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_ms_pred_ms_AR.json

# SIGReg + MS pred
python experiments/compare_lqr_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_ms_sigreg.json

# MS pred + MS AR + SIGReg
python experiments/compare_lqr_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout_ms.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_ms_pred_ms_AR_sigreg.json

# DINOv2 + MS pred
python experiments/compare_lqr_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/dinov2_rollout/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_dinov2_rollout.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_ms_dinov2.json

# iBOT + MS pred
python experiments/compare_lqr_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/ibot_rollout/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_ibot_rollout.yaml \
    --trials 10 --n-steps 300 --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 --device cuda --output results/lqr_ms_IBOT.json
```

### CartPole — CEM

CEM protocol: 10 trials, 300 steps, H=10, K=1 (executed steps), population=300, elites=30, iters=30.

```bash
# Template — replace --ckpts / --cfgs / --output for each variant
python experiments/compare_paper_cem_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/<variant>/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_<variant>.yaml \
    --trials 10 \
    --primitive-budget 300 \
    --planning-horizon 10 \
    --executed-steps 1 \
    --cem-population 300 \
    --cem-elites 30 \
    --cem-iters 30 \
    --cem-initial-variance 1.0 \
    --success-threshold 0.7 \
    --seed 123 \
    --device cuda \
    --output results/CEM_<variant>.json
```

Example for each model:

```bash
# 1-step AR
python experiments/compare_paper_cem_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_ar/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_1sp_AR_steps1.json

# 1-step SIGReg
python experiments/compare_paper_cem_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_sigreg/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_1sp_SIGreg_1step.json

# AR-1step+MS
python experiments/compare_paper_cem_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_ar_1step_ms/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio_ar_1step_ms.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_ms_AR.json

# MS pred + MS AR
python experiments/compare_paper_cem_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_rollout_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_ar_rollout.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_ms_pred_ms_AR.json

# SIGReg + MS pred
python experiments/compare_paper_cem_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_ms_SIGreg_1step.json

# MS pred + MS AR + SIGReg
python experiments/compare_paper_cem_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout_ms.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_ms_pred_ms_AR_sigreg.json

# DINOv2 + MS pred
python experiments/compare_paper_cem_smwm.py \
    --ckpts $RESULTS_DIR/cartpole/dinov2_rollout/model_final.pt \
    --cfgs  configs/cartpole_sensorimotor_world_model_dinov2_rollout.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_ms_dinov2_1step.json

# iBOT + MS pred
python experiments/compare_paper_cem_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/ibot_rollout/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_ibot_rollout.yaml \
    --trials 10 --primitive-budget 300 --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 --cem-initial-variance 1.0 \
    --success-threshold 0.7 --seed 123 --device cuda --output results/CEM_ms_IBOT.json
```

### PointMaze — CEM

```bash
# Ground-truth encoder (oracle upper bound)
MUJOCO_GL=osmesa python experiments/gt_cem_gymnasium_pointmaze.py \
    --trials 10 --n-steps 200 \
    --planning-horizon 25 --executed-steps 25 \
    --cem-population 300 --cem-elites 30 --cem-iters 10 \
    --frame-skip 5 \
    --output results/gt_oracle_cem_gymnasium_H25_fs5.json

# DINO-WM baseline (original checkpoint; requires py38 conda env — see Installation)
conda activate py38
cd ~/dino_wm
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export DATASET_DIR=/path/to/dino_bsmpc/data
WANDB_MODE=disabled MUJOCO_GL=egl python plan.py \
    --config-name plan_local \
    model_name=point_maze \
    ckpt_base_path=/path/to/dinowm_checkpoints \
    n_evals=10 2>&1 | tee /tmp/plan_live.txt

# Trained models (SIGReg+rollout, AR-1step+MS, Rollout+MS)
MUJOCO_GL=egl python experiments/compare_cem_dinowm_pointmaze.py \
    --ckpts \
        $RESULTS_DIR/pointmaze/sigreg_rollout_fs5/model_final.pt \
        $RESULTS_DIR/pointmaze/ar_1step_ms_fs5/model_final.pt \
        $RESULTS_DIR/pointmaze/rollout_ms_fs5/model_final.pt \
    --cfgs \
        configs/pointmaze_smwm_sigreg_rollout_fs5.yaml \
        configs/pointmaze_smwm_ar_1step_ms_fs5.yaml \
        configs/pointmaze_smwm_rollout_ms_fs5.yaml \
    --trials 10 --n-steps 200 \
    --planning-horizon 25 --executed-steps 25 \
    --cem-population 300 --cem-elites 30 --cem-iters 10 \
    --frame-skip 5 \
    --output results/cem_fs5_pointmaze.json
```

### PointMaze — LQR

```bash
# Ground-truth encoder (oracle upper bound)
MUJOCO_GL=osmesa python experiments/gt_lqr_gymnasium_pointmaze.py \
    --trials 10 --n-steps 200 \
    --frame-skip 5 \
    --output results/gt_oracle_lqr_gymnasium_fs5.json

# DINO-WM baseline
MUJOCO_GL=osmesa python experiments/lqr_dinowm_pointmaze.py \
    --dino-wm-dir ~/dino_wm \
    --ckpt /path/to/dinowm/model_latest.pth \
    --trials 10 --n-steps 200 --seed 0 \
    --output results/lqr_dinowm_pointmaze_fs5.json

# Trained models (SIGReg+rollout, AR-1step+MS, Rollout+MS)
MUJOCO_GL=egl python experiments/compare_lqr_pointmaze.py \
    --ckpts \
        $RESULTS_DIR/pointmaze/sigreg_rollout_fs5/model_final.pt \
        $RESULTS_DIR/pointmaze/ar_1step_ms_fs5/model_final.pt \
        $RESULTS_DIR/pointmaze/rollout_ms_fs5/model_final.pt \
    --cfgs \
        configs/pointmaze_smwm_sigreg_rollout_fs5.yaml \
        configs/pointmaze_smwm_ar_1step_ms_fs5.yaml \
        configs/pointmaze_smwm_rollout_ms_fs5.yaml \
    --trials 10 --n-steps 200 \
    --success-threshold 0.5 --success-hold-steps 1 \
    --frame-skip 5 --seed 0 \
    --output results/lqr_fs5_pointmaze.json
```

---

## Probe Scripts

Probe scripts analyze latent space geometry and control-theoretic properties without running full rollouts.

### CartPole Probes

#### Ground-truth planning cost

```bash
python experiments/plot_gt_planning_cost.py \
    --cfg configs/cartpole_sensorimotor_world_model_diff_proprio.yaml \
    --out results/gt_planning_cost.pdf
```

#### Per-model latent landscape summary (5 panels)

`plot_checkpoint_summary_smwm.py` produces a 5-panel figure per model:

- **Panel 1 — Phase portrait**: grid of (θ, θ̇) states; GT arrow vs learned arrow (encode → predict one step → decode via ridge probe).
- **Panel 2 — H=3 prediction error**: open-loop roll H=3 steps with zero action; error = ‖ẑ_H − z_GT_encoded‖.
- **Panel 3 — Planning cost**: log₁₀‖z_H − z_goal‖² (z_goal = encoded equilibrium); blue = model thinks state is near equilibrium after H steps.
- **Panel 4 — Latent norm divergence**: single trajectory from θ₀=0.1 rad; ‖z_t‖ GT (blue) vs recursive prediction ‖ẑ_t‖ (orange dashed).
- **Panel 5 — Cosine alignment**: multiple perturbations along unstable eigenvector; cos(Δz_GT, Δz_pred) over time — high = model predicts the right divergence direction.

```bash
# 1-step AR
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_ar/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio.yaml \
    --title "One-step Prediction AR" --out results/summary_1sp_AR.pdf

# 1-step SIGReg
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_sigreg/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg.yaml \
    --title "One-step Prediction SIGreg" --out results/summary_1sp_SIGreg.pdf

# SIGReg + MS pred
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml \
    --title "Multi-step Prediction SIGreg" --out results/summary_ms_SIGreg.pdf

# AR-1step+MS
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_ar_1step_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_ar_1step_ms.yaml \
    --title "Multi-step AR" --out results/summary_ms_AR.pdf

# DINOv2 + MS pred
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/dinov2_rollout/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_dinov2_rollout.yaml \
    --title "Multi-step Prediction DINOv2" --out results/summary_ms_DINOv2.pdf

# iBOT + MS pred
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/ibot_rollout/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_ibot_rollout.yaml \
    --title "Multi-step Prediction iBOT" --out results/summary_ms_IBOT.pdf

# MS pred + MS AR
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_rollout_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_ar_rollout.yaml \
    --title "Multi-step Prediction + MS AR" --out results/summary_ms_pred_ms_AR.pdf

# MS pred + MS AR + SIGReg
python experiments/plot_checkpoint_summary_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout_ms.yaml \
    --title "Multi-step Prediction + MS AR + SIGReg" --out results/summary_ms_pred_ms_AR_sigreg.pdf
```

#### Physical-space summary (state decoder panels)

`plot_checkpoint_summary_physical_smwm.py` repeats the same landscapes but decoded back to physical state space via a ridge regression probe.

```bash
python experiments/plot_checkpoint_summary_physical_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_ar/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio.yaml \
    --title "One-step Prediction AR (physical)" \
    --out  results/summary_physical_1sp_ar.pdf

# (repeat for each model variant — same --ckpt/--cfg pairs as above)
```

#### Local stability probe (3 panels)

`probe_local_stability_smwm.py` produces:

- **Panel 1 — Local vector field**: GT arrows vs learned arrows on (θ, θ̇) grid.
- **Panel 2 — Empirical ROA**: closed-loop episode (60 steps) using LQR gain from latent linearization; green = stable, with GT LQR contour overlaid.
- **Panel 3 — Lyapunov condition**: ΔV = V(x') − V(x) using Riccati matrix P from GT DARE; blue = ΔV < 0 (certificate holds), red = violated.

```bash
# SIGReg + MS pred
python experiments/probe_local_stability_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_sigreg_rollout/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml \
    --title "MS SIGreg" --out results/stability_MS_SIGreg.pdf

# AR-1step+MS
python experiments/probe_local_stability_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/diff_proprio_ar_1step_ms/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_diff_proprio_ar_1step_ms.yaml \
    --title "MS AR" --out results/stability_MS_AR.pdf

# iBOT + MS pred
python experiments/probe_local_stability_smwm.py \
    --ckpt $RESULTS_DIR/cartpole/ibot_rollout/model_final.pt \
    --cfg  configs/cartpole_sensorimotor_world_model_ibot_rollout.yaml \
    --title "MS iBOT" --out results/stability_MS_IBOT.pdf
```

#### LQR trajectory visualization

```bash
python experiments/plot_lqr_trajectories.py \
    --jsons results/lqr_1sp_AR.json results/lqr_1sp_sigreg.json \
            results/lqr_ms_sigreg.json results/lqr_ms_dinov2.json \
    --subtitles "1sp AR" "1sp SIGReg" "MS SIGReg" "MS DINOv2" \
    --trial 5 \
    --out results/lqr_trajectories.pdf
```

### PointMaze Probes

```bash
# Latent goal-distance landscape (trained model)
MUJOCO_GL=egl python experiments/probe_pointmaze_landscapes.py \
    --ckpt $RESULTS_DIR/pointmaze/dinov2_rollout_fs5/model_best.pth \
    --config configs/pointmaze_smwm_rollout_dinov2_fs5.yaml \
    --goal-xy -0.20 1.05 \
    --output results/probes/pointmaze_landscape_dinov2.pdf

# Latent goal-distance landscape (DINO-WM baseline, visual-only)
MUJOCO_GL=egl python experiments/probe_dinowm_distance_field.py \
    --dino-wm-dir ~/dino_wm \
    --ckpt /path/to/dinowm/model_latest.pth \
    --goal-xy -0.20 1.05 \
    --visual-only \
    --output results/probes/pointmaze_landscape_dinowm.pdf

# Landscape comparison panel (DINO-WM internal representation)
MUJOCO_GL=egl python experiments/probe_pointmaze_landscapes_dinowm.py \
    --dino-wm-dir ~/dino_wm \
    --ckpt /path/to/dinowm/model_latest.pth \
    --goal-xy -0.20 1.05 \
    --output results/probes/pointmaze_landscapes_dinowm_compare.pdf
```

### PointMaze Trajectory Visualization

```bash
MUJOCO_GL=egl python experiments/plot_lqr_trajectories_pointmaze.py \
    --ckpt $RESULTS_DIR/pointmaze/rollout_ms_fs5/model_best.pth \
    --config configs/pointmaze_smwm_rollout_ms_fs5.yaml \
    --output results/probes/pointmaze_lqr_trajectories.pdf
```

---

## Results

### CartPole: Balancing Success Rate (%)

*10 trials, 300 steps, success threshold = 0.7 rad. LQR: hold = 10 steps. CEM: H=10, K=1, pop=300.*

| Model | LQR (SR) | CEM (SR) | CEM (held) | GBP (SR) | GBP (held) |
|-------|-----|-----|-----|-----|-----|
| Ground-truth.             | **100%** | **100%** | 50%        | **100%** | **100%**
| Action recon. (1-step AR) | **100%** | **100%** | **100%**   | 30%      | 0%
| SIGReg (1-step)           |    0%    | **100%** | **100%**   | 90%      | 20%  
| AR-1step + MS AR          | **100%** | **100%** | **100%**   | **100%** | **100%**
| 1step pred + Endpoint AR. | **100%** | **100%** | **100%**   | **100%** | **100%**
| MS pred + MS AR           | 0%       | 0%       | 0%         |          |       
| SIGReg + MS pred          | 40%      | **100%** | **100%**   |          |
| MS pred + MS AR + SIGReg  | 0%       | **100%** | **100%**   |          |
| DINOv2 + MS pred          | 0%       | 0%       | 0%         |          |
| iBOT + MS pred            | 0%       | **100%** | **100%**   |          |
| iBOT + projector + AR     | **100%** | 80.0%    | 80%        |          |
| DINOv2 + projector + AR   | 0%       |  0%      | 0%         |          |   
| IBOT + projector + SIGReg | 40.0%    | 90.0%    | 90%        |          |
| DINOv2 + projector + SIGReg | 0%    | 60%       | 60%        |          |
| IBOT + projector + AR (no proprio) | 0% |    0% |  0%        |          |
| IBOT + projector + SIGReg (no proprio) | 0% | 0%|  0%        |          |
| iBOT + MS pred (no proprio)           | 0% | 0% |  0%        |          |

---

### PointMaze: Navigation Success Rate (%)

*Success = agent reaches within 0.5 m of goal. 10 trials, episode length 200 steps, frame skip 5.*

| Model | Encoder | CEM | LQR | GBP
|-------|---------|-----|-----|-----|
| Ground truth (oracle) | GT xy state | **100%** | **80%** | 90.0%#
| DINO-WM | Frozen DINOv2 | **100%*** | **80%** | 60%
| Action recon. (1-step AR)  | Learned (diff) | 90% | **80%** | 
| MS pred + MS AR  | Learned (diff) | **100%** | **80%** | 90%
| MS pred + Endpoint AR| Learned (diff) | 90% | **80%** | **100%**
| 1step pred + Endpoint AR| Learned (diff) | **100%** | **80%** | 
| SIGReg+rollout | Learned (diff) | 70% | **80%** | 60%
| AR-1step+MS AR | Learned (diff) | **100%** | 70% |
| DINOv2+rollout | Frozen DINOv2 | 60% | 10% |
| iBOT+rollout | Frozen iBOT | 80% | **80%** |

*\* 98% was reported by Zen et al. (2024) on D4RL U-maze with mujoco_py rendering for 50 trials, whereas here we consider only 10 trials. 
#\# GD through the ground truth MuJoCo dynamics isn't directly possible via backprop (the simulator isn't differentiable). The equivalent is Simultaneous perturbation stochastic approximation (SPSA). We use the same Adam optimizer loop, but gradient estimated via two simulator rollouts per step.


---

### Walker2D


---

## Repository Structure

```
JEPA_control/
├── configs/                          # YAML configuration files
│   ├── cartpole_sensorimotor_world_model_diff_proprio.yaml
│   ├── cartpole_sensorimotor_world_model_diff_proprio_sigreg.yaml
│   ├── cartpole_sensorimotor_world_model_diff_proprio_ar_1step_ms.yaml
│   ├── cartpole_sensorimotor_world_model_diff_proprio_ar_rollout.yaml
│   ├── cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml
│   ├── cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout_ms.yaml
│   ├── cartpole_sensorimotor_world_model_dinov2_rollout.yaml
│   ├── cartpole_sensorimotor_world_model_ibot_rollout.yaml
│   ├── pointmaze_smwm_rollout_ms_fs5.yaml
│   ├── pointmaze_smwm_sigreg_rollout_fs5.yaml
│   ├── pointmaze_smwm_rollout_dinov2_fs5.yaml
│   ├── pointmaze_smwm_rollout_ibot_fs5.yaml
│   ├── pointmaze_smwm_ar_1step_ms_fs5.yaml
│   └── pointmaze_smwm_ar_1step_fs5.yaml
├── control/                          # Planning algorithms
│   ├── cem.py                        # Cross-entropy method
│   ├── lqr.py                        # Linear quadratic regulator
│   ├── jacobian.py                   # Jacobian computation for linearization
│   ├── observer.py                   # State estimation utilities
│   └── rollout.py                    # Latent rollout utilities
├── data/                             # Dataset loading utilities
│   └── dataset.py
├── envs/                             # Environment wrappers
│   ├── cartpole_visual.py            # CartPole with RGB observations
│   └── pointmaze_visual.py           # PointMaze U-maze with RGB observations
├── experiments/                      # Runnable scripts
│   ├── generate_data.py              # CartPole flat HDF5 generation
│   ├── convert_flat_to_split_hdf5.py # Convert flat HDF5 to split episode format
│   ├── generate_cartpole_dataset.py  # CartPole data collection (alt entry point)
│   ├── generate_pointmaze_dataset.py # PointMaze data collection
│   ├── train_sensorimotor.py         # Training (CartPole + PointMaze)
│   ├── compare_paper_cem_smwm.py      # CartPole CEM evaluation (multi-model)
│   ├── compare_lqr_smwm.py           # CartPole LQR evaluation (multi-model)
│   ├── compare_cem_dinowm_pointmaze.py# PointMaze CEM evaluation (trained, multi-model)
│   ├── cem_dinowm_pointmaze.py       # PointMaze CEM (DINO-WM baseline, our env)
│   ├── gt_cem_gymnasium_pointmaze.py # PointMaze CEM (GT oracle)
│   ├── compare_lqr_pointmaze.py      # PointMaze LQR evaluation (trained)
│   ├── lqr_dinowm_pointmaze.py       # PointMaze LQR (DINO-WM baseline)
│   ├── gt_lqr_gymnasium_pointmaze.py # PointMaze LQR (GT oracle)
│   ├── plot_gt_planning_cost.py       # GT planning cost landscape
│   ├── plot_checkpoint_summary_smwm.py       # 5-panel latent landscape per model
│   ├── plot_checkpoint_summary_physical_smwm.py  # Same in physical state space
│   ├── probe_local_stability_smwm.py # ROA, vector field, Lyapunov probe
│   ├── plot_lqr_trajectories.py      # CartPole LQR trajectory visualization
│   ├── visualize_cartpole_paper_cem_trajectory.py
│   ├── probe_pointmaze_landscapes.py # PointMaze latent landscape (trained)
│   ├── probe_dinowm_distance_field.py# PointMaze latent landscape (DINO-WM)
│   ├── probe_pointmaze_landscapes_dinowm.py
│   └── plot_lqr_trajectories_pointmaze.py
├── losses/                           # Training objective functions
│   ├── prediction.py                 # L_fwd and L_rollout
│   └── sigreg.py                     # Singular value regularization
├── models/                           # Neural network architectures
│   ├── sensorimotor_world_model.py   # Main SMWM model
│   ├── encoder.py                    # Diff encoder (learned, patch-based)
│   ├── vit_encoder.py                # Frozen ViT encoder wrapper (DINOv2/iBOT)
│   ├── predictor.py                  # Latent dynamics predictor (Transformer)
│   └── action_encoder.py            # Action embedding module
├── probes/                           # Probe utility functions
│   ├── spectral.py                   # Spectral analysis of dynamics
│   └── suite.py                      # Probe suite runner
├── ground_truth/                     # Ground-truth dynamics (oracle)
│   └── cartpole_gt.py
├── training/                         # Training loop
│   └── trainer.py
└── analysis/                         # Metrics and plotting utilities
    ├── metrics.py
    └── plotting.py
```

---

## References

- Ivashkov et al., *Sensorimotor World Models* (2024) — [GitHub](https://github.com/petr-ivashkov/sensorimotor-world-model)
- Zen et al., *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning* (2024) — [GitHub](https://github.com/gaoyuezhou/dino_wm)
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023)
- Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer* (2021)
