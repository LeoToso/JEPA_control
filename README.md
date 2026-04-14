# JEPA Control

A research codebase for **Joint Embedding Predictive Architecture (JEPA) world models** applied to control of visual dynamical systems. The project studies whether structure-preserving representation losses (spectral regularisation, PBH stabilisability, NMP zero preservation) improve the control-theoretic properties of learned latent dynamics.

## Overview

The cartpole system is rendered as 64×64 pixel observations. A JEPA encoder maps images to a latent space; a predictor models latent transitions; post-training DMDc fits a linear system `(A_hat, B_hat)` to the latent dynamics; an LQR controller is designed on the latent system and deployed on the true system.

Six encoder variants are compared:

| Variant | Action encoder | Spectral loss | PBH loss |
|---------|---------------|--------------|---------|
| E-noact | Identity | — | — |
| E-spec | Identity | ✓ | — |
| E-PBH | Identity | — | ✓ |
| E-both-r | Identity | ✓ | ✓ |
| E-lift | Linear (d=32) | — | — |
| E-full | Linear (d=32) | ✓ | ✓ |

## Repository structure

```
JEPA_control/
├── configs/                  # YAML experiment configs
│   ├── cartpole.yaml
│   └── experiments.yaml
├── ground_truth/             # Exact cartpole linearisation (A*, B*, C*)
│   └── cartpole_gt.py
├── envs/                     # Pixel cartpole environment
│   └── cartpole_visual.py
├── identification/           # DMDc system identification
│   └── dmdc.py
├── models/                   # JEPA model components
│   ├── encoder.py            # ResNet visual encoder
│   ├── predictor.py          # MLP predictor
│   ├── action_encoder.py     # Linear / MLP / Identity action encoders
│   └── jepa.py               # Full JEPA model + factory
├── losses/                   # Training losses
│   ├── prediction.py         # JEPA prediction loss + VICReg
│   ├── pbh.py                # PBH stabilisability loss
│   ├── spectral.py           # Spectral matching loss
│   ├── straightening.py      # Temporal curvature loss
│   └── nmp.py                # NMP zero preservation loss
├── probes/                   # Evaluation probes (P1–P4, D1)
│   ├── spectral.py           # P1.1–P1.3: eigenvalue, Jordan block, frequency
│   ├── pbh.py                # P2.1–P2.3: stabilisability, detectability, separation
│   ├── kalman.py             # P3.1–P3.4: Kalman decomposition, phantom instability
│   ├── zeros.py              # P4.1–P4.4: transmission zeros, undershoot, Markov
│   ├── decoder.py            # D1: action decoder fidelity
│   └── suite.py              # Master probe runner
├── control/                  # Control design
│   ├── lqr.py                # Discrete LQR via DARE
│   ├── observer.py           # Luenberger observer design
│   └── rollout.py            # Closed-loop evaluation in true system
├── data/                     # Dataset generation and loading
│   └── dataset.py
├── training/                 # Training loop
│   └── trainer.py
├── experiments/              # Experiment runners
│   ├── run_experiment.py     # Single experiment
│   └── run_all.py            # Full grid (162 experiments)
├── analysis/                 # Results analysis and plotting
│   ├── metrics.py            # Correlation analysis, pivot tables
│   └── plotting.py           # All figures
└── tests/
    └── test_probes.py        # 32 unit tests
```

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, gymnasium, numpy, scipy, h5py, matplotlib, pandas, tqdm, pyyaml.

## Quick start

### 1. Run a single experiment

```bash
python -m experiments.run_experiment \
    --variant E-noact \
    --dataset random \
    --frame_skip 1 \
    --seed 42
```

### 2. Run the full experiment grid

```bash
python -m experiments.run_all --n_workers 4 --resume
```

The grid covers 6 variants × 3 datasets (`random`, `lqr`, `mixed`) × 3 frame-skips (1, 5, 10) × 3 seeds = **162 experiments**.

### 3. Run tests

```bash
pytest tests/test_probes.py -v
```

All 32 tests should pass.

### 4. Analyse results

```python
from analysis.metrics import load_all_results, compute_correlation_matrix
from analysis.plotting import generate_all_plots

df = load_all_results("results/")
corr = compute_correlation_matrix(df)
print(corr.head(10))
generate_all_plots(df, output_dir="figures/")
```

## Ground truth system

The cartpole linearisation (`ground_truth/cartpole_gt.py`) computes exact `A*`, `B*` via Jacobian linearisation around the upright equilibrium:

```
denom = 4M + m
A_c[1,2] = -3mg / denom
A_c[3,2] = +3(M+m)g / (l·denom)
B_c[1]   = +4 / denom
B_c[3]   = -3 / (l·denom)
```

The cartpole with standard parameters (`M=1.0 kg`, `m=0.1 kg`, `l=0.5 m`) is:
- **Unstable** (one eigenvalue > 1)
- **Non-minimum phase** (NMP zero outside the unit circle)

## Probes

Probes measure how well learned latent dynamics `(A_hat, B_hat, C_hat)` capture the control-relevant structure of the true system.

| Probe | What it measures |
|-------|----------------|
| P1.1 | Eigenvalue matching distance and unstable mode recall (UMR) |
| P1.2 | Jordan block structure detection |
| P1.3 | Marginal mode frequency alignment |
| P2.1 | PBH stabilisability index μ_S |
| P2.2 | PBH detectability index μ_D and mode sensitivity ratio (MSR) |
| P2.3 | Separation principle: LQR + observer in true environment |
| P3.1 | Kalman decomposition: controllable/observable subspace dimensions |
| P3.2 | Controllable subspace alignment with true system |
| P3.3 | Stable/unstable manifold geometry (growth rate fitting) |
| P3.4 | Phantom instability detection |
| P4.1 | Transmission zero count and matching distance |
| P4.2 | Step response undershoot (NMP detection) |
| P4.3 | Markov parameter sequence error and relative degree |
| P4.4 | Zero direction alignment |
| D1   | Action decoder fidelity and Lipschitz constant |

## Citation

If you use this codebase, please cite the associated paper (forthcoming).
