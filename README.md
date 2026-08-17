# JEPA control: does the latent representation preserve controllability of unstable modes?

This is a small, fully-linear codebase built to study a question in
isolation, where it can be analyzed exactly instead of only observed
empirically on pixels: when you jointly train an encoder and a predictor
with a **multistep prediction loss + SIGReg** (sketched isotropic-Gaussian
regularization, as in LeJEPA) on an **open-loop-unstable** system, does the
resulting latent representation retain enough information about the
unstable mode to design a controller that stabilizes the *original*
dynamics? And does swapping SIGReg for **multistep action reconstruction**
fix it?

Everything -- ground-truth dynamics, "sensing", encoder, predictor, control
synthesis, and evaluation -- is a single linear layer and closed-form/SGD
training, so the whole pipeline is exact and inspectable at every step, with
no confounds from a CNN/ViT encoder or a nonlinear plant.

## TL;DR result

Three examples, each trained with two configs (`L_pred + L_SIGReg` vs.
`L_pred + L_act`) via naive joint SGD (encoder and predictor optimized
together end-to-end, no closed-form solves):

| Example | SIGReg: unstable-mode R² | SIGReg: closed-loop success | act-recon: unstable-mode R² | act-recon: closed-loop success |
|---|---|---|---|---|
| 1 -- Synthetic, Unstable | ~0.01-0.10 (collapsed) | **0%** | ~1.00 (retained) | **100%** |
| 2 -- Synthetic, Stable | ~0.01 (collapsed) | ~100%* | ~0.01 (collapsed)* | ~100%* |
| 3 -- Cartpole | 0.001-0.012 (collapsed) | **0%** | 0.994-0.998 (retained) | 30-100%† |

\* Example 2 uses a system where BOTH modes are open-loop stable -- SIGReg
still collapses the low-variance mode (same mechanism as Example 1), but
since neither mode is actually unstable, losing one no longer breaks
control. Action-reconstruction *also* collapses here, because its
Example-1 advantage relies on the collapsed mode's response to actions
*amplifying* over time, which requires genuine instability. This isolates
that it's specifically losing the **unstable** mode that breaks control,
not "losing a mode" per se.

† Seed-dependent (see `experiments/example3_cartpole/run_actrecon_naive.py`'s
docstring for why) -- still a clear, reproducible separation from SIGReg's
reliable 0%.

## The three examples

All are discrete-time LTI systems `x_{t+1} = A x_t + B u_t`, stabilizable
(a linear controller that stabilizes the true dynamics *does* exist -- the
question is only whether the learned representation lets you find it), and
use `latent_dim < system.n` -- a latent strictly smaller than the true
state dimension -- so that the multistep prediction loss `L_pred` ties
*exactly* at zero between multiple candidate subspaces the encoder could
retain (only eigenspaces / A-invariant subspaces are exactly forward
-invariant under noiseless linear dynamics). This turns "did the
regularizer preserve the unstable mode" into a clean either/or outcome
instead of a matter of degree.

- **Example 1 -- Synthetic, Unstable**
  (`experiments/example1_synthetic_unstable/`): a minimal 2-state system
  (`make_double_mode_system`, eigenvalues `1.25` unstable / `0.85` stable)
  with `latent_dim=1`, forcing the encoder to keep *either* the unstable
  eigenspace *or* the stable one. Combined with the asymmetric
  initial-condition distribution below, SIGReg reliably collapses onto the
  stable (wrong) one; action reconstruction reliably keeps the unstable
  one.

- **Example 2 -- Synthetic, Stable**
  (`experiments/example2_synthetic_stable/`): identical construction to
  Example 1, except the "unstable" eigenvalue is lowered to `0.25` so
  **both** modes are open-loop stable. Isolates whether it's collapse
  *per se* or specifically losing the *unstable* mode that breaks control
  (it's the latter).

- **Example 3 -- Cartpole**
  (`experiments/example3_cartpole/`): the linearized cart-pole about its
  upright equilibrium (`make_linearized_cartpole_system`; state = `[cart
  pos, cart vel, pole angle, pole angular vel]`). Eigenvalues `{1, 1,
  ~1.10 (unstable pole), ~0.91 (stable pole)}` -- the repeated `1` is a
  genuine (non-diagonalizable) Jordan block spanning cart position/velocity.
  `latent_dim=3` is the n-state generalization of the same construction;
  see `experiments/example3_cartpole/run_sigreg_naive.py`'s docstring for
  the extra subtlety this system's Jordan block introduces.

## Why the initial-condition distribution is asymmetric

Each example draws its initial state with **larger variance along the
stable mode and smaller variance along the unstable mode**
(`make_modal_contaminated_x0_sampler` / `make_cartpole_modal_contaminated_x0_sampler`
in `src/jepa_lds/data.py`). This isn't an arbitrary adversarial trick: it
reflects realistic data collection near an unstable equilibrium. A policy
(human or automated) that avoids catastrophic failure will naturally spend
most of its time exploring near-equilibrium, stable-mode excursions and
only rarely encounter the large deviations that reveal the unstable
direction. Under a symmetric/isotropic initial-condition distribution,
`L_pred`'s own gradient already has a built-in incentive to keep the
unstable (growing) mode over the stable (shrinking) one, since omitting a
growing quantity costs more prediction error over a rollout -- which is
why SIGReg can *look* safe under "nice" data. The asymmetric distribution
is a minimal stress test that removes that accidental protection, so what
SIGReg's own objective actually does (nothing dynamics-aware) becomes
visible.

## The losses (`src/jepa_lds/losses.py`)

- **`L_pred`**: recursive multistep latent rollout error,
  `sum_h ||z_{t+h} - f_pred(z_{t+h-1}, a_{t+h-1})||^2`.
- **SIGReg**: embeddings are projected onto random 1D directions, and each
  projection is pushed toward the characteristic function of a standard
  normal (an Epps-Pulley-type statistic). By the Cramér-Wold theorem,
  matching all 1D projections implies the joint distribution is isotropic
  Gaussian. Crucially this only constrains the *marginal distribution* of
  `z` -- it has no notion of actions or controllability, and (since the
  encoder has a free rescaling) no preference between any two candidate
  subspaces that are both marginally Gaussian.
- **Multistep action reconstruction**: a linear decoder recovers the
  applied action sequence from a window of consecutive latents. Because
  `B` couples the action into the unstable mode, this loss directly forces
  the encoder to retain the action-relevant subspace -- which is only
  reliably the *unstable* one when that mode's response to actions
  genuinely amplifies over time (see Example 2's caveat above).

## Designing and evaluating a controller

For each trained `(encoder, predictor)` pair (`src/jepa_lds/control.py`):

1. **Extract** the learned latent linear system `(A_z, B_z)` -- exact,
   since the predictor literally *is* `z_{t+1} = A_z z_t + B_z a_t`.
2. **Check stabilizability** via the Popov-Belevitch-Hautus (PBH) test. If
   an unstable latent eigenvalue is uncontrollable, no linear feedback can
   stabilize it -- reported explicitly rather than silently producing a
   useless gain.
3. **Design** an LQR gain `K_z` via the discrete algebraic Riccati
   equation.
4. **Evaluate in closed loop on the TRUE system**: simulate
   `u_t = -K_z * encoder(y_t)` against the ground-truth
   `x_{t+1} = A x_t + B u_t` from many initial conditions and report
   success rate.

## Diagnostics (`src/jepa_lds/diagnostics.py`)

- **`unstable_mode_retention`**: a ridge probe fit on training data,
  evaluated on held-out data, regressing the true unstable modal
  coordinate from `z`. The quantitative version of "did the unstable mode
  collapse."
- **`unstable_eigenvector_alignment`**: compares the true system's
  dominant eigenvector against the learned predictor's own dominant
  eigenvector, mapped back to state space via the pseudoinverse of the
  encoder's exact analytic linear map (no data-fit probe involved) -- a
  direct, grid-free alignment check.
- **`closed_loop_trajectory_panel`** / **`lyapunov_decrease_panel`**: the
  two grid-based panels behind `probe_local_stability.py` (below).

## Repository layout

```
src/jepa_lds/
  systems.py         ground-truth LTI systems + LQR/PBH/controllability utilities
  data.py            observation model + episode generation + the asymmetric x0 samplers
  models.py          linear encoder / linear latent predictor / action decoders
  losses.py          multistep prediction, SIGReg, action reconstruction
  train.py           training loops: closed-form-predictor alternating scheme (train_jepa)
                      and naive end-to-end joint SGD (train_jepa_naive, used by all 3 examples)
  control.py         latent controller synthesis + closed-loop evaluation on the true system
  diagnostics.py     unstable-mode retention probe, eigenvector alignment, trajectory/Lyapunov panels
  plotting.py        figures for probe_local_stability.py / probe_eigenvector_alignment_grid.py
  checkpoint.py      checkpoint save/load
experiments/
  example1_synthetic_unstable/   run_sigreg_naive.py, run_actrecon_naive.py
  example2_synthetic_stable/     run_sigreg_naive.py, run_actrecon_naive.py
  example3_cartpole/             run_sigreg_naive.py, run_actrecon_naive.py
  checkpoint_io.py                  shared checkpoint -> (system, obs_model) reconstruction
  probe_local_stability.py         3-panel per-checkpoint figure (all 3 examples)
  probe_eigenvector_alignment_grid.py   all-pairs-of-dims alignment figure (Example 3)
tests/                                  unit tests (pytest)
```

## Installation

```bash
pip install -e .
```

## Running Example 1 (Synthetic, Unstable)

```bash
python experiments/example1_synthetic_unstable/run_sigreg_naive.py
python experiments/example1_synthetic_unstable/run_actrecon_naive.py

python experiments/probe_local_stability.py \
    --checkpoint results/example1_sigreg_naive/checkpoint_sigreg_naive_H4_seed0.pt \
    --traj-x0 0 80 --traj-x0 25 25 --traj-x0 0 -70 --traj-x0 -20 -50

python experiments/probe_local_stability.py \
    --checkpoint results/example1_actrecon_naive/checkpoint_actrecon_naive_H4_seed0.pt \
    --traj-x0 0 80 --traj-x0 25 25 --traj-x0 0 -70 --traj-x0 -20 -50
```

## Running Example 2 (Synthetic, Stable)

```bash
python experiments/example2_synthetic_stable/run_sigreg_naive.py
python experiments/example2_synthetic_stable/run_actrecon_naive.py

python experiments/probe_local_stability.py \
    --checkpoint results/example2_sigreg_naive/checkpoint_sigreg_naive_H4_seed0.pt \
    --skip-lyapunov --traj-x0 0 80 --traj-x0 25 25 --traj-x0 0 -70 --traj-x0 -20 -50

python experiments/probe_local_stability.py \
    --checkpoint results/example2_actrecon_naive/checkpoint_actrecon_naive_H4_seed0.pt \
    --skip-lyapunov --traj-x0 0 80 --traj-x0 25 25 --traj-x0 0 -70 --traj-x0 -20 -50
```

## Running Example 3 (Cartpole)

```bash
python experiments/example3_cartpole/run_sigreg_naive.py
python experiments/example3_cartpole/run_actrecon_naive.py

python experiments/probe_local_stability.py \
    --checkpoint results/example3_sigreg_naive/checkpoint_sigreg_naive_H8_seed0.pt \
    --hide-axis-units --traj-x0 0 80 --traj-x0 25 25 --traj-x0 0 -70 --traj-x0 -20 -50

python experiments/probe_local_stability.py \
    --checkpoint results/example3_actrecon_naive/checkpoint_actrecon_naive_H8_seed0.pt \
    --hide-axis-units --traj-x0 0 80 --traj-x0 25 25 --traj-x0 0 -70 --traj-x0 -20 -50
```

All combinations of state dimensions for eigenvector alignment (most
useful for Example 3's 4-state system, where a single 2D slice can't show
the whole picture):

```bash
python experiments/probe_eigenvector_alignment_grid.py \
    --checkpoint results/example3_sigreg_naive/checkpoint_sigreg_naive_H8_seed0.pt

python experiments/probe_eigenvector_alignment_grid.py \
    --checkpoint results/example3_actrecon_naive/checkpoint_actrecon_naive_H8_seed0.pt
```

Every training script accepts `--seed`, `--outer-rounds`, and many other
flags -- run any of them with `--help` for the full list.

## Testing

```bash
pytest
```
