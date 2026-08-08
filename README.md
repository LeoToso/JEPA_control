# JEPA control: does the latent representation preserve controllability of unstable modes?

This is a small, fully-linear codebase built to answer a question from the
cartpole-JEPA notes in isolation, where it can be analyzed exactly instead of
only observed empirically on pixels: when you jointly train an encoder and a
predictor with a **multistep prediction loss + SIGReg** (sketched isotropic
Gaussian regularization, as in LeJEPA) on an **open-loop-unstable** system,
does the resulting latent representation retain enough information about the
unstable mode to design a controller that stabilizes the *original* dynamics?
And does swapping SIGReg for **multistep action reconstruction** fix it?

Two synthetic discrete-time LTI (linear time-invariant) examples are used so
the whole pipeline -- ground-truth dynamics, "sensing", encoder, predictor,
control synthesis, and evaluation -- is closed-form and inspectable at every
step, with no confounds from a CNN/ViT encoder or a nonlinear plant.

## TL;DR result

Example 1 (`double_mode`, 2-state):

| config | unstable-mode R² | closed-loop success rate |
|---|---|---|
| `pred_only` (prediction, no regularizer) | 0.53 | 0% |
| `pred_sigreg` (prediction + SIGReg) | **0.23** | **0%** |
| `pred_actrecon` (prediction + action reconstruction) | **1.00** | **100%** |
| `oracle` (fixed, faithful encoder -- reference upper bound) | 1.00 | 100% |

Example 2 (`cartpole_linear`, 4-state):

| config | unstable-mode R² | closed-loop success rate |
|---|---|---|
| `pred_only` | -0.00 | 0% |
| `pred_sigreg` | **0.05** | **0%** |
| `pred_actrecon` | **1.00** | **100%** |
| `oracle` | 1.00 | 100% |

Both hold up across repeated seeds (see "Robustness" below), not just this
one run. Run `experiments/run_example1_double_mode.py` and
`experiments/run_example2_cartpole_linear.py` to reproduce (see below).

## The two examples

Both are discrete-time LTI systems `x_{t+1} = A x_t + B u_t` with spectral
radius `> 1` (open-loop unstable) and stabilizable (the unstable mode is
controllable, so a stabilizing linear controller *does* exist for the true
dynamics -- the question is only whether the learned representation lets you
find it).

- **Example 1 -- `double_mode`** (`src/jepa_lds/systems.py::make_double_mode_system`):
  a minimal 2-state system built directly in modal coordinates
  (`diag(1.25, 0.85)`) then rotated by a similarity transform so the unstable
  direction isn't trivially axis-aligned with the observation. One unstable
  real mode, one stable real mode, single input, fully controllable. Small
  enough to plot and to eigendecompose by hand.

- **Example 2 -- `cartpole_linear`** (`make_linearized_cartpole_system`): the
  classic cart-pole linearized about the *upright* (unstable) equilibrium,
  zero-order-hold discretized. State = `[cart pos, cart vel, pole angle, pole
  angular vel]`, input = force on the cart. Continuous-time spectrum is a
  genuine saddle (`0, 0, +omega, -omega`); after discretization this becomes
  a marginal repeated eigenvalue at `1` (cart translation) plus one unstable
  and one stable real eigenvalue from the pole. This is the linear analogue
  of the pixel-based cartpole world model the notes describe.

## Why a purely linear setting, and why it's non-trivial anyway

Everything -- encoder, predictor, action decoder -- is a single linear layer,
matching the "linear setting" framing of the request. This makes the
representation-learning problem exactly analyzable (eigenvalues of the
learned latent transition matrix `A_z` can be read off and compared directly
to the true system's eigenvalues), but it is *not* trivial: naively training
a linear encoder + linear recursive predictor jointly by gradient descent
from a random initialization reliably converges to a spuriously
*contractive* `A_z` (spectral radius `< 1`) that fits the training
trajectories almost perfectly yet completely misses the true instability --
independent of SIGReg or action reconstruction. This is a real, known
pathology of fitting unstable linear dynamics via a recursive-rollout loss
under gradient descent (see `src/jepa_lds/train.py`'s docstring). To isolate
what SIGReg vs. action reconstruction actually *do* to a representation
(rather than conflating that with "can SGD solve linear system ID from
scratch"), training alternates between a closed-form least-squares fit of the
predictor given the current encoder, and a few gradient steps on the encoder
(+ regularizer) given the current predictor -- see "Training procedure"
below.

## The observation model

The encoder never sees the true state directly. `ObservationModel`
(`src/jepa_lds/data.py`) maps `x_t` through a fixed random "rendering" matrix
into a higher-dimensional, redundant observation, plus extra i.i.d. Gaussian
nuisance channels -- a linear stand-in for the pixel encoder in the real
project: the JEPA encoder has to learn which directions of a
higher-dimensional signal are dynamically relevant.

## The losses (`src/jepa_lds/losses.py`)

Following the notes, the prediction backbone always combines a one-step
teacher-forced loss (`L_fwd`, H=1) with a recursive multistep rollout loss
(`L_rollout`, H = full episode). On top of that:

- **SIGReg**: LeJEPA-style sketched isotropic Gaussian regularization.
  Embeddings are projected onto random 1D directions, and each projection is
  pushed toward the characteristic function of a standard normal (an
  Epps-Pulley-type statistic). By the Cramér-Wold theorem, matching all 1D
  projections implies the joint distribution is isotropic Gaussian. Crucially
  this only constrains the *marginal distribution* of `z` -- it has no notion
  of actions or controllability.
- **Multistep action reconstruction**: a linear decoder recovers the applied
  action sequence from a window of consecutive latents, `(z_t, ..., z_{t+H})
  -> (a_t, ..., a_{t+H-1})`. Because the input matrix `B` couples the action
  into the unstable mode (that's what makes the system stabilizable in the
  first place), this loss directly forces the encoder to retain the
  action-relevant -- hence controllable, hence here the *unstable* --
  subspace, regardless of whether that subspace's marginal distribution
  looks Gaussian.

## Why SIGReg specifically fights the unstable mode

For an open-loop-unstable system driven by a mix of passive / PRBS / random /
LQR-guided episodes of fixed length, the *pooled* (over time and over
episodes) marginal distribution of the true unstable modal coordinate is a
scale-mixture across time: it's small early in an episode and grows
geometrically by the end, so pooling across all timesteps gives a
heavy-tailed, non-Gaussian distribution (measurably higher excess kurtosis --
the experiment scripts print this directly). The stable mode's marginal, by
contrast, is comparatively homoscedastic. Since the encoder here is
constrained to be *linear*, no rescaling can Gaussianize a genuinely
heavy-tailed coordinate. SIGReg's pressure toward isotropic-Gaussian `z`
therefore has a real incentive to attenuate the unstable direction relative
to the stable one -- exactly the "collapse of the unstable mode" the request
asked to illustrate. Action reconstruction has no competing Gaussianity
objective, so it doesn't fight this.

## What "designing a controller that stabilizes the original dynamics" means here

For each trained (encoder, predictor) pair:

1. **Extract** the learned latent linear system `(A_z, B_z)` -- exact, since
   the predictor literally *is* `z_{t+1} = A_z z_t + B_z a_t`
   (`src/jepa_lds/control.py::extract_latent_system`).
2. **Check stabilizability** of `(A_z, B_z)` via the Popov-Belevitch-Hautus
   (PBH) test. If a latent eigenvalue with `|lambda| >= 1` is uncontrollable,
   *no* linear state feedback can stabilize it -- controller design fails
   outright, and this is reported explicitly rather than silently producing
   a useless gain.
3. **Design** an LQR gain `K_z` on `(A_z, B_z)` via the discrete algebraic
   Riccati equation, if stabilizable.
4. **Evaluate in closed loop on the TRUE system**: simulate
   `u_t = -K_z * encoder(y_t)` against the ground-truth `x_{t+1} = A x_t + B
   u_t` from many random initial conditions, and report success rate, mean
   fraction of steps spent "stable", and final-state distance -- the same
   metrics format as the cartpole notes.

This is the crux of the demonstration: a representation can achieve near-zero
*prediction* loss and even a formally "stabilizable" latent system, and
still fail step 4, because nothing in prediction-only (or
prediction+SIGReg) training requires the encoder to be a *faithful*
(information-preserving) embedding of the true state -- only that it be
locally self-consistent under the predictor on the training distribution.

## Diagnostics (`src/jepa_lds/diagnostics.py`)

- **Unstable-mode retention R²**: a ridge probe fit on training data,
  evaluated on held-out data, regressing the true unstable modal coordinate
  from `z`. This is the quantitative version of "did the unstable mode
  collapse in the representation."
- **Eigenvalue comparison**: eigenvalues of `A_z` plotted against the true
  system's eigenvalues in the complex plane, against the unit circle.
- **Latent norm divergence** and **cosine alignment along the unstable
  eigenvector** (the notes' Panels 4 and 5): does a passive recursive latent
  rollout track the re-encoded ground truth, and does it diverge in the
  right *direction*.

## Robustness across seeds

Single-seed numbers can be lucky, so both regularizer strengths were tuned
against a 6-seed sweep (varying data + initialization, with the observation
matrix held fixed -- see "Running it" below) before being locked into the
experiment scripts:

- `pred_actrecon`: unstable-mode R² = 1.00 and closed-loop success = 100% on
  **6/6 seeds**, both examples.
- `pred_sigreg`: closed-loop success = 0% on **6/6 seeds** for `double_mode`,
  and 0% on 5/6 seeds for `cartpole_linear` (occasionally partial, never
  matching action-reconstruction's R²/success).
- `pred_only` (no regularizer at all): unreliable, sometimes collapses,
  sometimes not -- consistent with representation collapse being a real risk
  that *some* anti-collapse mechanism is needed for, and SIGReg's specific
  mechanism not being a safe choice for it here.

A naive, non-alternating joint fit of encoder+predictor (see `train.py`'s
docstring) is much noisier and can spuriously fail even the unregularized
case -- that pathology is orthogonal to the SIGReg-vs-action-reconstruction
question and is specifically what the alternating training scheme avoids.

## Repository layout

```
src/jepa_lds/
  systems.py        ground-truth LTI systems + LQR/PBH/controllability utilities
  data.py             observation model + open-loop episode generation (passive/PRBS/random/LQR mixture)
  models.py            linear encoder / linear latent predictor / action decoders
  losses.py             one-step & multistep prediction, SIGReg, action reconstruction
  train.py               alternating (closed-form predictor / gradient-descent encoder) training loop
  control.py               latent controller synthesis + closed-loop evaluation on the true system
  diagnostics.py           unstable-mode retention probe, eigenvalue comparison, norm/cosine panels
  plotting.py                all figures
experiments/
  common.py                shared end-to-end pipeline (data -> train 4 configs -> control -> plots -> summary)
  run_example1_double_mode.py
  run_example2_cartpole_linear.py
tests/                       unit + integration tests (pytest)
```

## Running it

```bash
pip install -e .
python experiments/run_example1_double_mode.py
python experiments/run_example2_cartpole_linear.py
```

Each writes plots, a `summary.json`, and a `summary.md` table to
`results/<example>/`. Run with `--seed N` to check a different random seed
(data + initialization; the observation/"sensor" matrix is intentionally
kept fixed across seeds via `--obs-seed`, since it represents a fixed
physical sensing apparatus, not something that should be re-randomized on
every run).

```bash
pytest
```
