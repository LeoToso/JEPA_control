# Latent Jacobian ($A_\text{aug}$, $B_\text{aug}$) — how it's computed

The control-theoretic probes (P1 spectral, P2 PBH), the LQR baseline, and the dynSIG Gramian all use a **local linear model** of the latent dynamics around the upright equilibrium. Because the predictor is **windowed** (it uses a short history), the correct linearization is the **companion-form augmented system** on the stacked state $Z_t = [z_{t-W+1}, \dots, z_{t-1}, z_t]$:

$$
Z_{t+1} - Z^* \;\approx\; A_\text{aug}\,(Z_t - Z^*) \;+\; B_\text{aug}\,u_t,
\qquad Z^* = [z^*, \dots, z^*].
$$

$A_\text{aug}$ and $B_\text{aug}$ are Jacobians of the predictor $f$, evaluated at the equilibrium by autograd. Code: `control/jacobian.py` (`compute_jacobian_np` for numpy/no-graph; `compute_jacobian_torch` for the differentiable version used by training losses). For $W=1$ both collapse to the standard $(d\times d)/(d\times m)$ pair.

## What `f` (the predictor) takes and outputs

Dimensions: $d=8$ (latent), $W=3$ (window), $d_a=1$ (encoded action). `f` is the windowed predictor MLP; its input is a window of $W$ latents and a window of $W$ actions, its output is the single next latent (`models/jepa.py:117–142`):

$$
z_{t+1} = f\big(\underbrace{[z_{t-2}, z_{t-1}, z_t]}_{W\text{ latents}},\ \underbrace{[u_{t-2}, u_{t-1}, u_t]}_{W\text{ actions}}\big).
$$

Internally it encodes each scalar action $u \mapsto c \in \mathbb{R}^{d_a}$, then flattens and concatenates both windows into one input vector of $W\cdot d + W\cdot d_a = 24 + 3 = 27$ numbers, and outputs $z_{t+1} \in \mathbb{R}^{8}$. So $f:\ \mathbb{R}^{27} \to \mathbb{R}^{8}$.

The Jacobian is the derivative of this map at the **equilibrium**: the whole latent window sits at $z^*$ (the encoding of the upright/centered image) and the action is zero. This is a fixed point — $f([z^*, z^*, z^*], 0) = z^*$ — which is exactly what $\mathcal{L}_\text{fp}$ trains.

## The companion form

Stacking the window into one state $Z_t \in \mathbb{R}^{Wd}$ turns the non-Markov windowed predictor into a first-order (Markov) system. The augmented matrices are block-structured ($W\times W$ blocks, each $d\times d$):

$$
A_\text{aug} =
\begin{bmatrix}
0 & I & 0 & \cdots & 0\\
0 & 0 & I & \cdots & 0\\
\vdots & & & \ddots & \vdots\\
0 & 0 & 0 & \cdots & I\\
J_0 & J_1 & J_2 & \cdots & J_{W-1}
\end{bmatrix}
\ (Wd\times Wd),
\qquad
B_\text{aug} =
\begin{bmatrix}
0\\ \vdots\\ 0\\ B_\text{jac}
\end{bmatrix}
\ (Wd\times m).
$$

- The **identity super-diagonal blocks** just shift the history forward: the new $Z_{t+1}$ drops $z_{t-2}$ and appends the freshly predicted $z_{t+1}$.
- The **last block-row** $[J_0, \dots, J_{W-1}]$ is the actual one-step prediction, where
  $$J_k = \left.\frac{\partial z_{t+1}}{\partial (\text{window slot } k)}\right|_{\text{all slots}=z^*,\ c=0}\ \in \mathbb{R}^{d\times d}.$$
- $B_\text{aug}$ is zero except the last $d$ rows, which hold $B_\text{jac} = \partial z_{t+1}/\partial c_t$.

For $d=8, W=3$: $A_\text{aug}$ is $24\times24$ and $B_\text{aug}$ is $24\times1$.

**Why this and not the old "freeze history" reduction:** an earlier version differentiated w.r.t. only the *last* window slot ($J_{W-1}$), holding history fixed at $z^*$, and used the resulting $(d\times d)$ matrix as $A$. That omits $J_0..J_{W-2}$ and gives the **wrong eigenvalues** whenever the predictor genuinely uses history (which it must, to represent a 2nd-order system). The companion form keeps all $W$ blocks, so the velocity degrees of freedom enter correctly through the shift structure.

## Computing the $J_k$ blocks → $A_\text{aug}$

All $W$ window slots are made leaf tensors (each equal to $z^*$), and we differentiate the output w.r.t. **every** slot (`jacobian.py:_build_companion` + `compute_jacobian_torch`):

```python
z_entries = [z_star.clone().requires_grad_(True) for _ in range(W)]   # W leaves, each (1, d)
z_win = torch.cat([e.unsqueeze(1) for e in z_entries], dim=1)          # (1, W, d) = [z*, z*, z*]
u_win = torch.zeros(1, W, 1)                                           # all actions zero

z_out = model.predict(z_win, u_win)                                   # f at equilibrium → (1, d)

J_blocks = []                                                          # one (d, d) block per slot
for k in range(W):
    rows = [torch.autograd.grad(z_out[0, i], z_entries[k], retain_graph=True)[0][0]
            for i in range(d)]
    J_blocks.append(torch.stack(rows))                                # J_k = ∂z_out / ∂(slot k)

A_aug = _build_companion(J_blocks, d, W)                               # (Wd, Wd)
```

`_build_companion` places identity blocks on the super-diagonal of the first $W-1$ block-rows and the $J_k$ blocks in the last block-row.

## Computing $B_\text{aug}$ (differentiate the action)

Hold all latents at $z^*$ and differentiate w.r.t. the **current** encoded action only; place the result in the last $d$ rows (`jacobian.py:46–66`):

```python
c_last  = torch.zeros(1, 1, m, requires_grad=True)                    # current action, the variable
c_win   = torch.cat([torch.zeros(1, W-1, m), c_last], dim=1)          # history actions = 0
z_win_b = z_star.expand(1, W, d)                                      # all latents frozen at z*

z_out_b = model.predict_from_encoded(z_win_b, c_win)                  # (1, d)
B_last  = torch.stack([torch.autograd.grad(z_out_b[0, i], c_last, retain_graph=True)[0][0, 0]
                       for i in range(d)])                            # (d, m) = B_jac
B_aug   = torch.cat([torch.zeros((W-1)*d, m), B_last], dim=0)         # (Wd, m)
```

This differentiates w.r.t. the **encoded/lifted** action $c \in \mathbb{R}^m$ ($m = $ `action_latent_dim` $= 1$ in fullspec), so $B_\text{jac}$ is $(d\times m) = (8\times1)$ and $B_\text{aug}$ is $(24\times1)$.

## From $B_\text{aug}$ to the scalar-action $B$

The real input is a scalar $u \in \mathbb{R}^1$ lifted by the linear action encoder $c = W_\text{enc}\,u$. The effective augmented input matrix follows by the chain rule on the last $d$ rows (`trainer.py:611–617`):

$$
B_\text{eff,aug} = \big[\,0;\ \dots;\ 0;\ B_\text{jac}\,W_\text{enc}\,\big] \in \mathbb{R}^{Wd \times 1}.
$$

$B_\text{eff,aug}$ is what the controllability Gramian / PBH alignment use for the scalar action.

## How the losses consume it (`trainer.py:597–648`)

- **Eigenvalues / $\rho$ / spectral loss** are taken on the **full** $A_\text{aug}$ (`eigvals = eigvals(A_aug)`).
- **PBH** aligns $B_\text{eff,aug}$ with the unstable eigenvectors of $A_\text{aug}$ (cosine-alignment loss).
- **dynSIG**: the controllability Gramian is computed on the augmented system, then **projected to the current-state $(d\times d)$ block** (last $d$ rows/cols) so $\Sigma_\text{tgt}$ matches `z`.
- **`L_local` / `L_temp`** cache the current-state block $A_\text{jac} = A_\text{aug}[(W-1)d:,\,(W-1)d:]$ and the last-$d$-rows slice of $B$.

## Notes

- **Velocity lives in the augmented state**, both via the companion shift structure and (now) via `frame_stack=2` at the encoder. See [the partial-observability discussion](CODEBASE_OVERVIEW.md).
- **Two entry points:** `compute_jacobian_np` (numpy, no graph — LQR/probes) and `compute_jacobian_torch` (keeps the graph via `create_graph=True` so gradients flow into the predictor weights — used by the spectral / PBH / dynSIG training losses).
- **Evaluation slices to the current-state block:** `run_experiment.py` calls `compute_jacobian_np` (which returns the augmented $Wd$ matrices for $W>1$) and slices $A_\text{jac} = A_\text{aug}[(W{-}1)d:,\,(W{-}1)d:]$, $B_\text{jac} = B_\text{aug}[(W{-}1)d:,:]$ for the downstream diagnostics, control, and visualization (an earlier shape mismatch here was fixed in commit `302dfc7`).
- **The minimal trainer sidesteps all of this:** `MinimalTrainer` requires a Markovian predictor (`predictor_window=1`), where $A_\text{aug} = A_\text{jac}$ and no companion form is needed.
