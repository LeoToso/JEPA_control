# JEPA Control — Codebase Overview

*Can JEPA (Joint Embedding Predictive Architecture) world models, augmented with structure-preserving regularization losses, learn latent representations whose dynamics are both **predictive** and **controllable**?* The testbed is the classic **visual cartpole** balancing task — learning to balance a pole from pixel observations.

The core thesis: a good world model isn't just one that predicts well — its learned latent dynamics should have good *control-theoretic* properties (stability, controllability, low linearization error), and you can encourage that with the right training losses.

## General pipeline

```
Consecutive pixel frames (frame_stack=2) → [CNN Encoder] → latent z → [Predictor] → z_{t+1}
                              ↑ (EMA target)            ↑
                         action u → [Action Encoder] → c
```

### 1. Representation learning (`models/`, `training/`)

**The JEPA world model — what this repo does:**
- `models/jepa.py` — encoder + MLP predictor + EMA target encoder (BYOL-style EMA; `target_encoder_momentum` = 0.99 in fullspec, model default 0.996). The target encoder supplies the prediction targets $z_{1..H}$ only when `use_target_encoder: true` (set in the JEPA configs, off in the AE configs and unused by the minimal trainer); otherwise targets are stop-gradient through the online encoder. Predicts in latent space, no pixel reconstruction.
- `models/predictor.py` — windowed MLP predictor z_{t+1} = f([z_{t-W+1..t}], [c_{t-W+1..t}]), window W=3. Also `ResidualMLPPredictor` (Markovian, W=1): $f(z,c) = z + g(z,c) - g(z^*,0)$, which makes $f(z^*,0)=z^*$ an **exact fixed point by construction** (enabled via `predictor_residual: true`; used by the minimal trainer).
- `models/action_encoder.py` — lifts scalar action u ∈ ℝ¹ → latent action c ∈ ℝ^da (Linear/MLP); da=1 in fullspec.
- `training/trainer.py` — combines many losses (prediction, spectral matching, PBH, VICReg, SIGreg, dynamics-aware SIGreg), EMA target updates, Jacobian regularization, and online sliding-window DMDc identification.
- `training/trainer_minimal.py` — **minimal trainer** (selected with `trainer: minimal` in the config), a stripped-down variant of the method: only four loss terms,
  $\mathcal{L} = \mathcal{L}_\text{1step} + \beta(e)\,\mathcal{L}_\text{multi} + \lambda_\text{inv}\mathcal{L}_\text{inv} + \lambda_\text{fp}\mathcal{L}_\text{fp} + \lambda_\text{dynSIG}\mathcal{L}_\text{dynSIG}$, where $\mathcal{L}_\text{multi}$ is an open-loop rollout loss (no re-encoding) warmed up by $\beta(e)=\min(1, e/50)$. Requires a Markovian (W=1) predictor; everything else (spec, PBH, local, unstable, temporal, mirror, anchors, state supervision, EMA targets) is deliberately omitted. The dynSIG target starts isotropic ($\Sigma_\text{tgt}=I$) and is EMA-shifted toward the Gramian target once the Jacobian is available.
- `training/diagnostics.py` — standalone diagnostics for the minimal trainer (one-step/multi-step prediction error, fixed-point drift, action sensitivity, $\rho(A)$, $\lVert B\rVert$, PBH $\sigma_{\min}$, local-LQR closed-loop eval) with a documented failure→fix mapping (e.g. $B\approx0$ → strengthen inverse dynamics/PBH; $\rho(A)<1$ despite unstable env → add instability-margin loss).

**Encoder options** (selected via `encoder_type: cnn | vit`):
- `models/cnn_encoder.py` — Bounou et al. (2021) 6-block CNN (Conv3×3→MaxPool→BN→ReLU; spatial 64→32→…→1). **Default for `cartpole_v2_fullspec`** (latent_dim=8, `frame_stack=2` → 6 input channels); ~1000× better near-equilibrium sensitivity than the ViT.
- `models/vit_encoder.py` — Vision Transformer alternative: patch embedding → transformer blocks → mean-pooled latent projection (CLS token removed).

**Observability note:** with `frame_stack=2` the encoder input is two consecutive frames, so the latent `z` captures velocity, not just configuration (`x, θ`); the predictor additionally sees a window of W=3 latents.

**Baselines (AE / Bounou et al.):**
- `models/autoencoder.py` — AE baseline that adds a pixel decoder + reconstruction loss for comparison; trained by `experiments/train_ae.py` (the `cartpole_ae_*` configs: pixel-DMD + reconstruction, with `cartpole_ae_bounou` being the faithful Bounou et al. setup planned via global DMD-LQR). Its pixel decoders are the components in `models/cnn_decoder.py` / `models/decoder.py`.

### 2. Losses (`losses/`)

**Control-aware losses — the research contribution:**
- `pbh.py` — PBH stabilisability. The **active** fullspec loss is a **cosine-alignment** variant computed in `trainer.py` (align the input direction with the unstable eigenvectors of the companion-form Jacobian — see the objective table). The file's `pbh_stabilizability_loss` is the older log-barrier on σ_min([λI − Â, B̂]) (DMDc-estimated Â, B̂).
- `spectral.py` — spectral matching: pulls learned A eigenvalues toward the cartpole's true unstable eigenvalues.
- `dyn_sigreg.py` — dynamics-aware SIGreg: SIGreg with the target covariance aligned to the controllability Gramian W_T (the control-aware twist on plain SIGreg).

**Generic self-supervised losses:**
- `prediction.py` — the JEPA prediction loss (`jepa_prediction_loss`, stop-gradient targets) plus optional VICReg (variance-invariance-covariance) collapse prevention (`vicreg_loss`, `vicreg_collapse_loss`).
- `sigreg.py` — SIGreg (LeJEPA): collapse prevention via an Epps–Pulley goodness-of-fit test on random 1D projections of z, enforcing z ≈ N(0, I).
- `nmp.py`, `straightening.py` — additional manifold/straightening losses.

#### Training objective

The trainer (`training/trainer.py`) assembles a weighted sum of loss terms; only those with non-zero weight in a given config are active:

$$
\mathcal{L}_\text{total} = \sum_i \lambda_i\,\mathcal{L}_i
$$

**Notation:** $z^*$ is the latent equilibrium; $f$ the predictor; $\mathrm{enc}$ the online encoder; $A, B$ the latent Jacobian of $f$ at $z^*$ (detached); $c = W_\text{enc}u$ the lifted action; $\mathrm{sg}(\cdot)$ stop-gradient; $\mathrm{sh}$ the state head; $H$ the prediction horizon; $\rho = \max_i|\lambda_i(A)|$ the spectral radius. For the SIGreg-family terms, $w_k$ are $K$ random unit projection directions, $\omega_p$ are $P$ frequency points in $[0.5, 3]$, and $\phi^{re}_{kp}, \phi^{im}_{kp}$ are the real/imaginary parts of the empirical characteristic function of the projected batch $w_k^\top z$.

This table is the **full menu** of available terms; only those with non-zero weight in a given config are active. The **first six rows are the active `cartpole_v2_fullspec` terms**, in the order they appear in the objective below the table; the remaining rows have zero weight in the current configs.

| Term | Symbol | λ flag | Equation | Purpose |
|------|--------|--------|----------|---------|
| Prediction | $\mathcal{L}_\text{pred}$ | `lambda_pred` | $\dfrac{1}{H}\sum_{k=1}^{H}\lVert \hat z_k - \mathrm{sg}(z_k)\rVert^2,\ \ \hat z_k=f(z_{k-1},c_{k-1})$ | Latent multi-step prediction (main JEPA signal) |
| Inverse dyn. | $\mathcal{L}_\text{inv}$ | `lambda_inv` | $\big\lVert \psi(z_0, z_H) - \bar u/s\big\rVert^2,\ \ \bar u=\tfrac1H\sum_k u_k$ | Recover mean action from $(z_0, z_H)$ (anti-collapse) |
| Fixed point | $\mathcal{L}_\text{fp}$ | `lambda_fp` | $\lVert f(z^*, 0) - z^*\rVert^2$ | $z^*$ is an equilibrium under $u=0$ |
| Spectral | $\mathcal{L}_\text{spec}$ | `lambda_spec` | $\sum_{\lambda^*}\min_i\lvert\lambda_i-\lambda^*\rvert^2 + 2(\rho-\rho^*)^2$ | Match learned eigenvalues to GT unstable set |
| PBH | $\mathcal{L}_\text{PBH}$ | `lambda_PBH` | $1 - \max_{u}\cos^2\!\big(B_\text{aug},\, v_u\big)$, with $v_u$ = unstable eigenvectors of $A_\text{aug}$ | Input must excite the unstable modes: align $B_\text{aug}$ with the unstable eigenvectors of the companion-form Jacobian |
| dynSIG | $\mathcal{L}_\text{dynSIG}$ | `lambda_dynSIG` | $\dfrac{1}{KP}\sum_{k,p}\big[(\phi^{re}_{kp} - e^{-\frac12\omega_p^2 v_k})^2 + (\phi^{im}_{kp})^2\big],\ \ v_k = w_k^\top\Sigma_\text{tgt}w_k$ | Shape $\mathrm{cov}(z)$ toward the Gramian target $\Sigma_\text{tgt}$ |
| Mirror | $\mathcal{L}_\text{mirror}$ | `lambda_mirror` | $\lVert \mathrm{enc}(o) + \mathrm{enc}(\mathrm{flip}(o)) - 2z^*\rVert^2$ | Break the $\lvert\theta\rvert$ vs $\theta$ sign degeneracy (no labels). See [mirror_loss.md](mirror_loss.md)|
| SIGreg | $\mathcal{L}_\text{sigreg}$ | `lambda_sigreg` | $\dfrac{1}{KP}\sum_{k,p}\big[(\phi^{re}_{kp} - e^{-\omega_p^2/2})^2 + (\phi^{im}_{kp})^2\big]$ | Isotropic collapse prevention ($z\sim\mathcal N(0,I)$). See [sigreg.md](sigreg.md) |
| Variance floor | $\mathcal{L}_\text{varfloor}$ | `lambda_varfloor` | $\dfrac{1}{d}\sum_j \mathrm{relu}\!\big(\sqrt{\Sigma_{\text{tgt},jj}} - \sqrt{\mathrm{var}(z_j)+\epsilon}\big)^2$ | Anti-collapse gradient that stays finite at full collapse |
| Spectral eigvec | $\mathcal{L}_\text{spec\_eig}$ | `lambda_spec_eig` | $\lVert A\,\hat v_u - \lambda^*\,\hat v_u\rVert^2$ | Align dominant unstable mode of $A$ with GT ($\hat v_u$ = latent GT eigvec, $\lambda^*$ its eigenvalue) |
| State decode | $\mathcal{L}_\text{state}$ | `lambda_state` | $\dfrac{1}{H+1}\sum_{k=0}^{H}\mathrm{mean}\big(w\odot(\mathrm{sh}(z_k)-s_k)\big)^2$ | Decode physical state $s=(x,\dot x,\theta,\dot\theta)$, weights $w=[50,0.1,100,1]$ |
| Eq. anchor | $\mathcal{L}_\text{anchor}$ | `lambda_anchor` | $\mathrm{mean}\big(w\odot \mathrm{sh}(z^*)^2\big)$ | $\mathrm{sh}(z^*)$ decodes to the zero state |
| Encoder anchor | $\mathcal{L}_\text{enc\_anchor}$ | `lambda_enc_anchor` | $\mathrm{mean}\big(\mathrm{enc}(o_\text{eq})^2\big)$ | Pull $\mathrm{enc}(o_\text{eq})$ toward the origin |
| Spectral marginal | $\mathcal{L}_\text{spec\_marginal}$ | `lambda_spec_marginal` | $\sum_{i\in\mathcal{K}}\lvert\lambda_i - 1\rvert^2$ | Pull the $k$ near-marginal eigenvalues toward 1 |
| Local linear. | $\mathcal{L}_\text{local}$ | `lambda_local` | $\big\lVert f(z,u) - \big(z^* + A(z-z^*) + Bu\big)\big\rVert^2$ | Enforce local linear dynamics near eq |
| Unstable margin | $\mathcal{L}_\text{unstable}$ | `lambda_unstable` | $\mathrm{relu}(1+\eta-\rho)^2 + \mathrm{relu}(\rho-\rho_{\max})^2$ | Keep $\rho$ in $[1+\eta,\ \rho_{\max}]$ (data-driven, no GT) |
| VICReg | $\mathcal{L}_\text{vicreg}$ | `use_vicreg` | $\lambda_\text{var}\,\dfrac1d\sum_j\mathrm{relu}\!\big(1-\sqrt{\mathrm{var}(z_j)+\epsilon}\big) + \nu\,\dfrac1d\big(\lVert\mathrm{Cov}(z)\rVert_F^2 - \textstyle\sum_j\mathrm{Cov}(z)_{jj}^2\big)$ | Variance + covariance collapse prevention |
| Temporal cov. | $\mathcal{L}_\text{temp}$ | `lambda_temp` | $\dfrac{\lVert \Sigma_1^\text{res} - A\Sigma_0\rVert_F^2}{\lVert\Sigma_1^\text{res}\rVert_F^2 + \epsilon},\ \ \Sigma_0=\tfrac1N dz_t^\top dz_t,\ \Sigma_1^\text{res}=\tfrac1N (dz_{t+1}-Bc_t)^\top dz_t$ | Latent covariance must propagate by $A$ |

**Jacobian convention:** for the windowed predictor (W=3) the spec / PBH / dynSIG terms use the **companion-form augmented** Jacobian $A_\text{aug}$ (Wd×Wd) and $B_\text{aug}$ (Wd×m); eigenvalues, $\rho$, and the eigenvectors $v_u$ are taken on $A_\text{aug}$. See [jacobian.md](jacobian.md).

The Gramian target used by dynSIG / varfloor is $\Sigma_\text{tgt} = (1-\alpha)\bar W + \alpha I$, where $\bar W$ is the finite-horizon controllability Gramian $W_T = \sum_{k=0}^{T_g-1} A^k B B^\top (A^\top)^k$ normalised to unit mean-diagonal. It is computed on the augmented system and then **projected to the current-state $(d\times d)$ block** (last $d$ rows/cols) so it matches `z`. $\Sigma_\text{tgt}$ is `detach()`ed, so no gradient flows through $A, B$.


**Gating** (see `trainer.py`): `L_spec` / `L_PBH` require the GT unstable eigenvalues and the latent Jacobian, recomputed every `jacobian_every` steps (`jacobian_every: 1` in the JEPA configs — cost is negligible at d=8); `L_dynSIG` / `L_varfloor` require $\Sigma_\text{tgt}$, built from that Jacobian and inert before the first update; `L_spec_eig` (when on) fires every step via a forward-mode JVP after `epoch ≥ spec_eig_warmup_epochs`. An optional warm-up phase (`warmup_epochs`, = 0 in fullspec) disables pred/spec/PBH/fp and trains encoder + inverse-dynamics only.

**Active objective for `cartpole_v2_fullspec.yaml`** (all other λ = 0, `use_vicreg: false`; CNN encoder, latent_dim 8, frame_stack 2):

$$
\mathcal{L}_\text{total} =
1.0\,\mathcal{L}_\text{pred}
+ 1.0\,\mathcal{L}_\text{inv}
+ 30.0\,\mathcal{L}_\text{fp}
+ 5.0\,\mathcal{L}_\text{spec}
+ 1.0\,\mathcal{L}_\text{PBH}
+ 2.0\,\mathcal{L}_\text{dynSIG}
$$

#### Training phase

How `Trainer.fit()` (`training/trainer.py:765`) actually runs the optimization:

**Optimization setup.** Adam (`lr: 1e-4`, `weight_decay: 1e-4`; optional `predictor_lr_mult` puts the predictor in its own param group) with cosine annealing down to `0.1·lr`. Gradients are clipped to global norm 1.0 every step.

**Per step** (`train_epoch` → `_compute_loss`):
1. Encode the first frame **online**: $z_0 = \mathrm{enc}(o_0)$ — the only encoder output in the prediction path that carries gradient.
2. Encode targets $z_{1..H}$ under `no_grad`, via the **EMA target encoder** when `use_target_encoder: true` (fullspec), else the online encoder (plain stop-gradient).
3. Optionally encode a pixel-noise-augmented copy $z_0^\text{aug}$ (`aug_noise_std`) used **only** by the collapse-prevention terms (dynSIG / varfloor), keeping noise out of the predictor's training distribution.
4. $z^*$ comes from encoding the exact equilibrium image (`set_obs_eq`, supplied by `run_experiment.py`) rather than an EMA over near-eq samples — eliminates the train/eval $z^*$ mismatch.
5. Roll the predictor out $H$ steps for $\mathcal{L}_\text{pred}$; assemble all active loss terms. The Jacobian-based terms (spec / PBH / dynSIG target) recompute $A_\text{aug}, B_\text{aug}$ every `jacobian_every` steps (= 1 in the JEPA configs); `spec_eig` (when on) fires every step via a forward-mode JVP.
6. `backward()` → clip → `step()` → EMA update of the target encoder (`target ← 0.99·target + 0.01·online` in fullspec).

**Per epoch:** train epoch → val epoch (`no_grad`, same losses) → scheduler step → CSV log (`training_log.csv`) → checkpoint every `checkpoint_every` epochs. If `lambda_unstable > 0`, $\rho_{\max}$ is re-estimated each epoch from the growth rates observed in the previous one (mean + 2σ, clipped to [1+η+0.01, 1.6]).

**Phase schedule** (all optional; fullspec uses none of them — it trains the full objective from epoch 0):
1. **Warm-up** (`warmup_epochs` > 0): pred / spec / PBH / fp are zeroed; only encoder-shaping terms (state supervision via `warmup_lambda_state`, inverse dynamics, anchor) train — seeds the encoder before the prediction loss can lock it into a dynamics-blind latent space.
2. **Encoder freeze** (`freeze_encoder_epochs` > 0): right after warm-up the encoder is frozen so the predictor learns dynamics on a fixed latent space; on unfreeze the encoder LR can be scaled down (`encoder_lr_unfreeze_mult`) to avoid prediction-loss oscillation.
3. **Full objective** for the remaining epochs.

**Model selection.** `fit()` tracks the best-val-loss state *and* the first epoch where `spec_loss` < `spec_converge_thresh` (0.01). At the end it keeps the **best-val model only if the spectral regularizer had already converged by that epoch** — otherwise low val loss may coexist with a wrong spectrum, so the **final-epoch** weights are used for Jacobian identification and control instead.

**MinimalTrainer** (`trainer: minimal`) uses the same epoch skeleton (Adam + cosine, clip 1.0, CSV, checkpoints) but no phase schedule and no EMA encoder; instead: the multi-step rollout loss ramps in via $\beta(e)=\min(1, e/\text{multistep\_warmup\_epochs})$, $z^*$ is refreshed from $\mathrm{enc}(o_\text{eq})$ and pushed into the residual predictor (`set_predictor_z_star`), the dynSIG covariance target is EMA-updated from the Jacobian Gramian every `jacobian_every` steps, and the best-val model is always restored at the end (no spectral-convergence gate — there is no spectral loss).

### 3. Control (`control/`) — runs on the learned latent model
- `lqr.py` — discrete LQR via DARE (with phantom eigenvalue deflation).
- `cem.py` — Cross-Entropy Method planner (batched action sampling in latent space).
- `mpc.py`, `grad_mpc.py` — gradient-based MPC (optimizes action sequences via backprop).
- `jacobian.py` — linearizes the windowed predictor around equilibrium via autograd → **companion-form** $A_\text{aug}$ (Wd×Wd), $B_\text{aug}$ (Wd×m). See [jacobian.md](jacobian.md) for the construction.
- `observer.py`, `rollout.py`, `visualize.py` — state observer, trajectory rollout/execution, visualization.

### 4. System identification (`identification/`)
- `dmdc.py` — Dynamic Mode Decomposition with control: least-squares fit of z_{t+1} = Az_t + Bu_t from trajectories (also a proximal ADMM variant).

### 5. Evaluation (`probes/`, `ground_truth/`)
- `probes/suite.py` — control-theoretic probes:
  - **P1 (Spectral):** eigenvalue match of A_jac vs ground truth.
  - **P2 (PBH):** stabilisability rank test via singular values.
  - **P3 (Residual):** linearization error near the equilibrium z*.
  - **P4 (Decoder/Kalman):** reconstruction/filtering residuals.
- `probes/pbh.py`, `spectral.py`, `kalman.py`, `zeros.py`, `decoder.py` — individual probe implementations.
- `ground_truth/cartpole_gt.py` — analytical linearization of the cartpole around the upright equilibrium (true A*, B*, eigenvalues, controllability), used to validate the learned matrices.

### 6. Data (`data/`)
- `dataset.py` — cartpole trajectory dataset mixing: random exploration, LQR expert rollouts, equilibrium/self-loop (stability), passive divergence (u=0), and persistent excitation (linearization validation).
- **Rendering fix** (`envs/cartpole_visual.py`): observations are downsampled with **area averaging** (`cv2.INTER_AREA`, PIL `BOX` fallback) instead of nearest-neighbour subsampling, which aliased small pole angles (< ~0.03 rad) into bit-identical frames near equilibrium. `experiments/check_render_aliasing.py` verifies a dataset is clean.
- Cached h5 datasets and results dirs are **keyed by config stem** (e.g. `cartpole_ae_noLQR_ep_seed43.h5`), so configs with different data mixes don't collide on the same cache file.

### 7. Analysis (`analysis/`)
- `metrics.py` — loads/correlates results, flattens experiment JSON.
- `plotting.py` — visualization utilities.

## Techniques used

| Technique | Purpose |
|-----------|---------|
| JEPA | Self-supervised representation learning via latent prediction (no reconstruction) |
| CNN encoder (Bounou 2021) / ViT | Visual encoder (pixels → latent; CNN is the fullspec default, d=8, frame_stack=2) |
| LQR | Optimal linear feedback control on learned A_jac, B_jac |
| CEM | Model-predictive planning in latent space |
| DMD / DMDc | System identification of linear latent dynamics from data |
| PBH stabilisability | Ensure unstable modes are controllable |
| Spectral regularisation | Align learned eigenvalues with ground truth |
| SIGreg / dynSIGreg | Collapse prevention (isotropic / Gramian-aligned target covariance) |
| VICReg | Alternative collapse prevention |
| Jacobian linearisation | Local linear model at equilibrium via autograd |
| EMA target encoder | Stable training targets (BYOL-style) |

## How it's run

```bash
python -m experiments.run_experiment \
  --config configs/cartpole_v2_fullspec.yaml \
  --results_dir results --seed 42
```

#### What this run does, step by step

1. **Skip check.** The experiment name is built as `v2_<variant>_<dataset>_fs<frame_skip>[_fstack<k>]_seed<seed>` (here `v2_E-full_mixed_fs1_fstack2_seed42`). If `results/<exp_name>/model_final.pt` already exists, the run **exits immediately and returns the latest cached `results.json`** — pass `--force` to retrain.
2. **Ground truth.** Builds the analytical cartpole linearization (`ground_truth/cartpole_gt.py`) → true unstable eigenvalues for the spectral loss and probes.
3. **Equilibrium image.** Renders the upright/centered frame $o_\text{eq}$ (the deterministic render of state $0$) — used for self-loop injection and as the fp-loss anchor ($z^* = \mathrm{enc}(o_\text{eq})$). It is **not saved anywhere**: it's re-rendered each run, and the `n_eq_selfloop` self-loop transitions built from it are injected into the train split in memory at load time (`data/dataset.py:_inject_eq_selfloops`) — they are not part of the h5 file.
4. **Dataset — generate or load.** Looks for `data/<config_stem>_ep_fs<frame_skip>_seed<seed>.h5` (here `data/cartpole_v2_fullspec_ep_fs1_seed42.h5`). If it exists it is **loaded as-is** (no simulation). Otherwise `data/dataset.py:generate_dataset()` rolls out the real pygame env for every mix in the config's `data:` section (random / LQR expert / equilibrium / PRBS / passive), renders all frames to 64×64, and saves the h5 to that path. Note the cache key is only *(config stem, frame_skip, seed)* — editing the `data:` section of an existing config does **not** regenerate; delete the h5 first. (Exception: `n_eq_selfloop` is applied at load time, not baked into the h5, so editing it takes effect immediately.)
5. **Model.** Builds the JEPA model from the `model:` section (CNN encoder, d=8, W=3 for fullspec).
6. **Training.** `Trainer` (or `MinimalTrainer` if `trainer: minimal`) runs the [training phase](#training-phase); checkpoints go to `results/<exp_name>/checkpoints/`, final weights to `results/<exp_name>/model_final.pt`.
7. **Identification.** $z^* = \mathrm{enc}(o_\text{eq})$; companion-form Jacobian at $z^*$ → slice the current-state block $A_\text{jac}$ ($d\times d$), $B_\text{jac}$ ($d\times m$).
8. **Control evaluation.** LQR on $(A_\text{jac}, B_\text{jac})$ and CEM/MPC planning, executed closed-loop on the real env (skippable via `--no-control`, restrictable via `--cem-only`).
9. **Probes.** P1 spectral / P2 PBH / P3 linearization residual / P4 decoder vs. ground truth.
10. **Outputs.** Everything is written to a **timestamped** dir `results/<exp_name>/<YYYYMMDD_HHMMSS>/` — `results.json` (control success rate, settling time, cost, spectral-radius error, PBH μ_S, residuals, DMDc quality) plus eval artifacts (frames, videos, npy arrays). The timestamping means repeated runs never overwrite each other's results; only the model weights at `results/<exp_name>/model_final.pt` are a stable, overwritten path (so `--eval-only` always finds the latest model).

Other useful flags: `--eval-only` (+ optionally `--checkpoint <path>`) re-runs steps 7–10 on saved weights; `--resume <ckpt>` continues training; `--epochs N` overrides the config.

- **Configs** (`configs/*.yaml`) define the variants being compared:
  - `cartpole_v2_fullspec.yaml` — flagship: CNN encoder, frame_stack 2; pred + inv + fp + spec + PBH + dynSIG.
  - `cartpole_v2_fullspec_nodynamSIG.yaml` — fullspec ablation without dynSIG.
  - `cartpole_jepa_minimal.yaml` — **minimal trainer** (`trainer: minimal`): Markovian residual predictor + 4-loss objective (λ_fp = 1, not 30 — the residual predictor enforces the fixed point by construction).
  - `cartpole_jepa_sigreg_baseline.yaml` — SIGreg-only variant.
  - `cartpole_ae_baseline.yaml` — autoencoder baseline.
  - `cartpole_ae_noLQR.yaml` / `cartpole_ae_withLQR.yaml` — AE/DMD baselines (pixel-DMD + reconstruction + a secondary CEM predictor), without vs. with near-equilibrium LQR data (LQR data helps Jacobian-LQR but its z*→z* self-loops can hurt CEM).
  - `cartpole_ae_bounou.yaml` — **faithful Bounou et al. (2021)**: DMD-pixel + reconstruction only (no MLP predictor loss); planning via global DMD-LQR.
  - `cartpole_jepa_recon.yaml` — JEPA with reconstruction.
- **Entry points** (`setup.py`): `jepa-run` → `experiments.run_experiment:main`, `jepa-grid` → `experiments.run_all:main`.
- **Dependencies** (`requirements.txt`): `torch>=2.0`, `torchvision`, `numpy`, `scipy`, `gymnasium>=0.29`, `pygame`, `h5py`, `PyYAML`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `pytest`.

#### Oumayma's torch

Personal setup: datasets and results go to scratch instead of the repo defaults (`JEPA_SCRATCH` is exported in `~/.bashrc` as `/scratch/ob2184/JEPA_control_scratch`):

```bash
python -m experiments.run_experiment \
    --config configs/cartpole_v2_fullspec.yaml --seed 42 \
    --data_dir "$JEPA_SCRATCH/data" --results_dir "$JEPA_SCRATCH/results"
```

(`train_ae.py` / `generate_data.py` take the same dirs with dash spelling: `--data-dir`, `--results-dir`.)

## Experiments & outputs

- `experiments/run_experiment.py` — main pipeline: generate mixed dataset → train model (`Trainer` or `MinimalTrainer` per the config's `trainer:` key) → identify A_jac/B_jac (current-state $(d\times d)$ block sliced from the companion-form augmented Jacobian) → run control (LQR/CEM) → evaluate probes → save `results/<exp_name>/<timestamp>/results.json` (control success rate, settling time, cost, spectral radius error, PBH μ_S, linearization residual, DMDc quality).
- `experiments/run_all.py` — grid search runner.
- `experiments/train_ae.py` — autoencoder/DMD baseline training; evaluates both CEM (MLP predictor rollouts) and **DMD-LQR** (Bounou et al. planning: a single global $(A, B)$ ridge-fit from all training transitions, then LQR-MPC on the real env); data/results paths keyed by config stem.
- `experiments/visualize_ae_recon.py`, `visualize_ae_predictor.py` — AE reconstruction and predictor-quality figures (one-step prediction vs. open-loop rollout error accumulation).
- `experiments/check_render_aliasing.py` — scans an h5 dataset for bit-identical observations of distinct physical states (the signature of the old nearest-neighbour aliasing); distinguishes real aliasing from the sub-pixel resolution floor; `--dump-outliers` inspects offending pairs.
- `experiments/compare_latent_dynamics.py` — phase-portrait comparison (A_jac vs GT).
- `experiments/validate_world_model.py` — forward prediction MSE.
- `experiments/visualize_latent.py` — t-SNE/UMAP embedding analysis.
- `experiments/sensitivity_analysis.py`, `sweep_cem_qr.py` — hyperparameter sweeps.

The project uses YAML configs + CLI runners. A LaTeX writeup of the architecture, losses, and data generation lives at `jepa_writeup.tex` (repo root).

## Recent direction (from git history)

Most recent: **fixed observation aliasing** — area-average downsampling replaces a nearest-neighbour fallback that collapsed distinct near-equilibrium states into bit-identical frames (verified by `check_render_aliasing.py`); built out the **AE/DMD baseline family faithful to Bounou et al.** (pixel-DMD + reconstruction training, global DMD fit + LQR-MPC planning evaluated alongside CEM in `train_ae.py`, plus recon/predictor visualizations); and added a **minimal JEPA trainer baseline** (`trainer: minimal` + `cartpole_jepa_minimal.yaml`) with a Markovian residual predictor (exact fixed point by construction) and only four losses, paired with standalone diagnostics and a failure→fix mapping — the philosophy: first make the latents predictive, action-sensitive, and anchored at equilibrium; add control-theoretic regularizers only when a diagnostic shows a specific failure.

Earlier: **switched the encoder from ViT to a Bounou-style CNN** (far better near-equilibrium sensitivity) with `frame_stack=2` (so the encoder sees velocity, not just configuration); **corrected the latent Jacobian to companion form** for the windowed predictor ($A_\text{aug}$, $B_\text{aug}$); **replaced the PBH log-barrier with a cosine-alignment loss** on the companion-form unstable eigenvectors and re-enabled it; and turned on the inverse-dynamics loss while disabling mirror / spec_eig / varfloor in the fullspec objective. Earlier work added passive (u=0) divergence data so the identified spectral radius ρ correctly exceeds 1 (reflecting the genuinely unstable cartpole).
