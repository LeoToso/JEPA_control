# JEPA World Models for Visual Control

Code for the paper **"Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models"** (Toso et al., ICLR 2027).

The model is a JEPA world model with architecture similar to the [Sensorimotor World Model (SMWM)](https://github.com/petr-ivashkov/sensorimotor-world-model) of Ivashkov et al. (2026). We compare four training objectives across three control benchmarks: CartPole, PointMaze, and Walker2D.

**Model variants** (paper names):

| Name | Prediction | Inverse Dynamics |
|------|-----------|-----------------|
| **1SP+SIG** | One-step | SIGReg |
| **MSP+SIG** | Multi-step | SIGReg |
| **1SP+EP-IDM** | One-step | Endpoint IDM |
| **MSP+EP-IDM+SIG** | Multi-step | Endpoint IDM + SIGReg |

We also compare against **DINO-WM** (Zen et al., 2024), which uses a frozen DINOv2 encoder with a separately learned predictor.

---

## Tasks

### CartPole

Visual pole-balancing. Observations: **128×128 RGB** + 4D proprioception (cart position/velocity, pole angle/angular velocity). Actions: scalar force. Frame skip = 5.

Dataset: **985 trajectories** (75% open-loop exploration, 25% LQR-stabilized).

**Control methods:**

| Planner | Description |
|---------|-------------|
| **LQR** | Linearize latent dynamics at the operating point; solve discrete-time DARE. |
| **CEM** | Cross-entropy method over action sequences in latent space. |
| **GBP** | Optimize action sequence by backpropagating through the differentiable latent rollout. |

---

### PointMaze

Goal-conditioned navigation in a **U-shaped maze**. Observations: **64×64 RGB** (196×196 for DINOv2/iBOT) + 4D proprioception (x, y, vx, vy). Actions: 5 sub-actions × 2D = 10D. Frame skip = 5.

Dataset: **2000 episodes** from a scripted expert.

**External baseline:** DINO-WM (Zen et al., 2024) — reports **98% success** on D4RL U-maze with mujoco_py rendering.

---

### Walker2D

Locomotion of a **bipedal walker**. Observations: **64×64 RGB** + joint proprioception. Control via iCEM in latent space (H=3).

---

## Architecture

The model encodes each observation into a latent vector `z_t` and learns a predictor `f(z_t, a_t) → z_{t+1}`. Encoders:

| Encoder | Type | Description |
|---------|------|-------------|
| **Diff** | Learned | Differentiable patch-difference encoder, trained end-to-end |
| **DINOv2** | Frozen | ViT-S/14 self-supervised features (384D per patch token) |
| **iBOT** | Frozen | ViT-S/16 masked-image-modeling features (384D per patch token) |

Training objectives:

| Loss | Description |
|------|-------------|
| L_fwd | One-step MSE in latent space |
| L_rollout | Multi-step prediction over H-step horizon |
| L_IDM | Action reconstruction from (z_t, z_{t+1}) |
| L_EP-IDM | Endpoint IDM: action sequence from (z_t, z_{t+H}) |
| L_SIG | SIGReg — [LeJEPA](https://github.com/galilai-group/lejepa) |

---

## Installation

### Prerequisites

- Python 3.9+
- MuJoCo
- GPU (recommended)

### Setup

```bash
git clone https://github.com/LeoToso/JEPA_control.git
cd JEPA_control
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt

# For PointMaze
pip install gymnasium-robotics

# For frozen encoders (DINOv2 / iBOT)
pip install timm
```

### Environment variables

```bash
export DATASET_DIR=/path/to/datasets
export RESULTS_DIR=/path/to/results
```

### DINO-WM baseline *(optional)*

```bash
git clone https://github.com/gaoyuezhou/dino_wm ~/dino_wm
```

> DINO-WM requires a separate conda environment (`py38`) with `mujoco_py`, `d4rl`, and `gym==0.23.1`.

---

## Data Generation

### CartPole

```bash
# Step 1 — collect episodes
python experiments/generate_data.py \
    --config configs/cartpole_jepa_recovery_vit_128_diff_latent64_excitation_depth4.yaml \
    --output data/cartpole_excitation_depth4_fs5_128_flat.h5 \
    --seed 42

# Step 2 — convert to split format
python experiments/convert_flat_to_split_hdf5.py \
    --src data/cartpole_excitation_depth4_fs5_128_flat.h5 \
    --out data/cartpole_excitation_depth4_fs5_128
```

### PointMaze

```bash
MUJOCO_GL=osmesa python experiments/generate_pointmaze_dataset.py \
    --output-dir data/pointmaze_u_fs5_ma_64 \
    --n-episodes 2000 --episode-steps 100 \
    --image-size 64 --maze-map U --seed 0 \
    --frame-skip 5 --multi-action
```

### Walker2D

```bash
python experiments/generate_walker2d_dataset.py \
    --output-dir data/walker2d_mixed_sac_fs5_64 \
    --n-episodes 2000 --frame-skip 5 --seed 42
```

---

## Training

All models are trained with `train.py` and a YAML config.

### CartPole

```bash
# 1SP+SIG
python experiments/train.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_jepa_diff_proprio_sigreg.yaml \
    --save-dir $RESULTS_DIR/cartpole/1sp_sig \
    --seed 42 --device cuda

# 1SP+EP-IDM
python experiments/train.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_jepa_diff_proprio.yaml \
    --save-dir $RESULTS_DIR/cartpole/1sp_ep_idm \
    --seed 42 --device cuda

# MSP+SIG
python experiments/train.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_jepa_diff_proprio_sigreg_rollout.yaml \
    --save-dir $RESULTS_DIR/cartpole/msp_sig \
    --seed 42 --device cuda

# MSP+EP-IDM+SIG
python experiments/train.py \
    --data data/cartpole_excitation_depth4_fs5_128 \
    --config configs/cartpole_jepa_diff_proprio_sigreg_rollout_ms.yaml \
    --save-dir $RESULTS_DIR/cartpole/msp_ep_idm_sig \
    --seed 42 --device cuda
```

### PointMaze

```bash
# 1SP+EP-IDM
MUJOCO_GL=egl python experiments/train.py \
    --config configs/pointmaze_jepa_ar_1step_fs5.yaml \
    --data $DATASET_DIR/pointmaze_u_fs5_ma_64 \
    --save-dir $RESULTS_DIR/pointmaze/1sp_ep_idm

# MSP+SIG
MUJOCO_GL=egl python experiments/train.py \
    --config configs/pointmaze_jepa_sigreg_rollout_fs5.yaml \
    --data $DATASET_DIR/pointmaze_u_fs5_ma_64 \
    --save-dir $RESULTS_DIR/pointmaze/msp_sig

# MSP+EP-IDM (rollout+MS)
MUJOCO_GL=egl python experiments/train.py \
    --config configs/pointmaze_jepa_ar_1step_ms_fs5.yaml \
    --data $DATASET_DIR/pointmaze_u_fs5_ma_64 \
    --save-dir $RESULTS_DIR/pointmaze/msp_ep_idm
```

### Walker2D

```bash
# 1SP+EP-IDM
python experiments/train.py \
    --data data/walker2d_mixed_sac_fs5_64 \
    --config configs/walker2d_fwd_ep_ar.yaml \
    --save-dir $RESULTS_DIR/walker2d/1sp_ep_idm \
    --seed 42 --device cuda

# 1SP+SIG
python experiments/train.py \
    --data data/walker2d_mixed_sac_fs5_64 \
    --config configs/walker2d_fwd_sr.yaml \
    --save-dir $RESULTS_DIR/walker2d/1sp_sig \
    --seed 42 --device cuda

# MSP+SIG
python experiments/train.py \
    --data data/walker2d_mixed_sac_fs5_64 \
    --config configs/walker2d_ms_sr.yaml \
    --save-dir $RESULTS_DIR/walker2d/msp_sig \
    --seed 42 --device cuda
```

---

## Evaluation

### CartPole — LQR

*10 trials, 300 steps, success threshold = 0.7 rad, hold = 10 steps.*

```bash
python experiments/eval_lqr_cartpole.py \
    --ckpts $RESULTS_DIR/cartpole/<variant>/model_final.pt \
    --cfgs  configs/cartpole_jepa_<variant>.yaml \
    --trials 10 --n-steps 300 \
    --success-threshold 0.7 --success-hold-steps 10 \
    --q-scale 1.0 --r-scale 1.0 \
    --device cuda --output results/lqr_<variant>.json
```

### CartPole — CEM

*H=10, K=1, pop=300, elites=30, iters=30.*

```bash
python experiments/eval_cem_cartpole.py \
    --ckpts $RESULTS_DIR/cartpole/<variant>/model_final.pt \
    --cfgs  configs/cartpole_jepa_<variant>.yaml \
    --trials 10 --primitive-budget 300 \
    --planning-horizon 10 --executed-steps 1 \
    --cem-population 300 --cem-elites 30 --cem-iters 30 \
    --cem-initial-variance 1.0 --success-threshold 0.7 \
    --seed 123 --device cuda --output results/cem_<variant>.json
```

### PointMaze — CEM

```bash
# Ground-truth oracle
MUJOCO_GL=osmesa python experiments/gt_cem_gymnasium_pointmaze.py \
    --trials 10 --n-steps 200 \
    --planning-horizon 25 --executed-steps 25 \
    --cem-population 300 --cem-elites 30 --cem-iters 10 \
    --frame-skip 5 --output results/gt_oracle_cem_H25_fs5.json

# Trained models
MUJOCO_GL=egl python experiments/compare_cem_dinowm_pointmaze.py \
    --ckpts \
        $RESULTS_DIR/pointmaze/msp_sig/model_final.pt \
        $RESULTS_DIR/pointmaze/msp_ep_idm/model_final.pt \
        $RESULTS_DIR/pointmaze/1sp_ep_idm/model_final.pt \
    --cfgs \
        configs/pointmaze_jepa_sigreg_rollout_fs5.yaml \
        configs/pointmaze_jepa_ar_1step_ms_fs5.yaml \
        configs/pointmaze_jepa_ar_1step_fs5.yaml \
    --trials 10 --n-steps 200 \
    --planning-horizon 25 --executed-steps 25 \
    --cem-population 300 --cem-elites 30 --cem-iters 10 \
    --frame-skip 5 --output results/cem_pointmaze.json
```

### PointMaze — LQR

```bash
MUJOCO_GL=egl python experiments/compare_lqr_pointmaze.py \
    --ckpts \
        $RESULTS_DIR/pointmaze/msp_sig/model_final.pt \
        $RESULTS_DIR/pointmaze/msp_ep_idm/model_final.pt \
    --cfgs \
        configs/pointmaze_jepa_sigreg_rollout_fs5.yaml \
        configs/pointmaze_jepa_ar_1step_ms_fs5.yaml \
    --trials 10 --n-steps 200 \
    --success-threshold 0.5 --success-hold-steps 1 \
    --frame-skip 5 --seed 0 --output results/lqr_pointmaze.json
```

### Walker2D — iCEM

*H=3, pop=500, elites=50, iters=5, 10 trials, 500 steps.*

```bash
# Ground-truth oracle (with SAC warm-start)
MUJOCO_GL=egl python experiments/gt_icem_walker.py \
    --trials 10 --primitive-budget 500 \
    --planning-horizon 3 --executed-steps 3 \
    --cem-population 500 --cem-elites 50 --cem-iters 5 \
    --sac-policy-warmstart \
    --sac-ckpt /path/to/sac_policy.pt \
    --output results/gt_icem_H3.json

# Latent iCEM (trained model)
MUJOCO_GL=egl python experiments/latent_icem_walker.py \
    --ckpt $RESULTS_DIR/walker2d/1sp_ep_idm/model_final.pt \
    --config configs/walker2d_fwd_ep_ar.yaml \
    --trials 10 --primitive-budget 500 \
    --planning-horizon 3 --executed-steps 3 \
    --cem-population 500 --cem-elites 50 --cem-iters 5 \
    --output results/latent_icem_1sp_ep_idm.json
```

---

## Results

### CartPole

*10 trials, 300 steps, success threshold = 0.7 rad. LQR: hold = 10 steps. CEM: H=10, K=1, pop=300.*

| Model | LQR (SR) | CEM (SR) | GBP (SR) |
|-------|----------|----------|----------|
| GT | **100%** | **100%** | **100%** |
| 1SP+SIG | 0% | **100%** | 90% |
| 1SP+EP-IDM | **100%** | **100%** | **100%** |
| MSP+SIG | 0% | **100%** | **100%** |
| MSP+EP-IDM+SIG | **100%** | **100%** | **100%** |
| DINO-WM | 30% | 40% | 10% |

---

### PointMaze

*Success = within 0.5 m of goal. 10 trials, 200 steps, frame skip = 5.*

| Model | CEM (SR) | LQR (SR) | GBP (SR) |
|-------|----------|----------|----------|
| GT | **100%** | **80%** | 90% |
| 1SP+EP-IDM | **100%** | **80%** | 80% |
| MSP+SIG | 90% | 60% | **100%** |
| MSP+EP-IDM | **100%** | **80%** | 90% |
| DINO-WM | **80%**\* | **80%** | 60% |

\* Zen et al. report 98% for 50 trials with mujoco_py rendering.

---

### Walker2D

*10 trials, 500 steps, H=3 iCEM.*

| Model | Avg Velocity (m/s) | Displacement (m) | Final Height (m) |
|-------|--------------------|-----------------|-----------------|
| Ground truth (GT) | **3.61** | **14.43** | **1.21** |
| 1SP + SIG | 0.08 | 0.06 | 0.98 |
| MSP + SIG | 0.46 | 0.66 | 0.85 |
| 1SP + EP-IDM | 2.91 | 9.15 | 0.92 |

---

## Probe Scripts

Probe scripts analyze latent space geometry without full rollouts.

### CartPole — latent landscape summary

`plot_summary.py` produces a 5-panel figure per model: phase portrait, H=3 prediction error, planning cost, latent norm divergence, and cosine alignment.

```bash
python experiments/plot_summary.py \
    --ckpt $RESULTS_DIR/cartpole/<variant>/model_final.pt \
    --cfg  configs/cartpole_jepa_<variant>.yaml \
    --title "<variant>" --out results/summary_<variant>.pdf
```

### CartPole — local stability probe

`probe_stability.py` produces: local vector field, empirical ROA, and Lyapunov condition.

```bash
python experiments/probe_stability.py \
    --ckpt $RESULTS_DIR/cartpole/<variant>/model_final.pt \
    --cfg  configs/cartpole_jepa_<variant>.yaml \
    --out results/stability_<variant>.pdf
```

### PointMaze probes

```bash
MUJOCO_GL=egl python experiments/probe_pointmaze_landscapes.py \
    --ckpt $RESULTS_DIR/pointmaze/msp_ep_idm/model_best.pth \
    --config configs/pointmaze_jepa_rollout_ms_fs5.yaml \
    --goal-xy -0.20 1.05 --output results/pointmaze_landscape.pdf
```

---

## Repository Structure

```
JEPA_control/
├── configs/                          # YAML configuration files
│   ├── cartpole_jepa_*.yaml          # CartPole model configs
│   ├── pointmaze_jepa_*.yaml         # PointMaze model configs
│   └── walker2d_jepa_*.yaml          # Walker2D model configs
├── control/                          # Planning algorithms
│   ├── cem.py                        # Cross-entropy method (iCEM)
│   ├── lqr.py                        # Linear quadratic regulator
│   ├── jacobian.py                   # Jacobian computation for linearization
│   ├── observer.py                   # State estimation utilities
│   └── rollout.py                    # Latent rollout utilities
├── data/                             # Dataset loading utilities
│   └── dataset.py
├── envs/                             # Environment wrappers
│   ├── cartpole_visual.py            # CartPole with RGB observations
│   ├── pointmaze_visual.py           # PointMaze U-maze with RGB observations
│   └── walker2d_visual.py            # Walker2D with RGB observations
├── experiments/                      # Runnable scripts
│   ├── generate_data.py              # CartPole data generation
│   ├── generate_pointmaze_dataset.py # PointMaze data generation
│   ├── generate_walker2d_dataset.py  # Walker2D data generation
│   ├── train.py                      # Training (all tasks)
│   ├── eval_cem_cartpole.py          # CartPole CEM evaluation
│   ├── eval_lqr_cartpole.py          # CartPole LQR evaluation
│   ├── compare_cem_dinowm_pointmaze.py  # PointMaze CEM evaluation
│   ├── compare_lqr_pointmaze.py      # PointMaze LQR evaluation
│   ├── gt_icem_walker.py             # Walker2D GT iCEM (oracle)
│   ├── latent_icem_walker.py         # Walker2D latent iCEM
│   ├── eval_latent_sac_walker.py     # Walker2D latent-SAC bridge
│   ├── make_trajectory_figure.py     # Multi-row trajectory figure from GIFs
│   ├── plot_summary.py               # 5-panel latent landscape per model
│   ├── probe_stability.py            # ROA, vector field, Lyapunov probe
│   ├── probe_utils.py                # Shared probe utilities
│   └── walker2d_utils.py             # Walker2D model loading utilities
├── losses/                           # Training objectives
│   ├── prediction.py                 # L_fwd, L_rollout
│   └── sigreg.py                     # Singular value regularization (SIGReg)
├── models/                           # Neural network architectures
│   ├── jepa_world_model.py           # Main JEPA world model (JEPAWorldModel)
│   ├── encoder.py                    # Diff encoder (learned, patch-based)
│   ├── vit_encoder.py                # Frozen ViT encoder (DINOv2/iBOT)
│   ├── predictor.py                  # Latent dynamics predictor (Transformer)
│   └── action_encoder.py             # Action embedding module
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

- Toso et al., *Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models* (ICLR 2027)
- Ivashkov et al., *Sensorimotor World Models* (2026) — [GitHub](https://github.com/petr-ivashkov/sensorimotor-world-model)
- Zen et al., *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning* (2024) — [GitHub](https://github.com/gaoyuezhou/dino_wm)
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023)
- Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer* (2021)
