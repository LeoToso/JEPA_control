# SIGReg — Sketched Isotropic Gaussian Regularization

How SIGReg works, as implemented in the [`lejepa`](https://github.com/rbalestr-lab/lejepa) repository (Balestriero & LeCun, *LeJEPA*, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)).

Source files referenced throughout (paths relative to the `lejepa` repo):

| Component | File |
|---|---|
| Univariate Epps–Pulley test | `lejepa/univariate/epps_pulley.py` |
| Random slicing (multivariate → 1-D) | `lejepa/multivariate/slicing.py` |
| Shared univariate base class | `lejepa/univariate/base.py` |
| Self-contained ~20-line version + full training loop | `MINIMAL.md` |

---

## 1. The problem it solves

JEPA-style self-supervised training minimizes a prediction (invariance) loss between embeddings of different views of the same input. That loss has a trivial global minimum: **representation collapse** — the encoder maps every input to the same point (total collapse) or onto a low-dimensional subspace (dimensional collapse), making prediction perfect and the representation useless. The classical fixes are heuristics: stop-gradient (SimSiam), EMA teacher–student (BYOL, I-JEPA), negative pairs (contrastive), variance/covariance penalties (VICReg).

LeJEPA replaces all of them with one principled constraint on the embedding distribution:

$$
z \;\sim\; \mathcal{N}(0,\, I_D).
$$

The paper proves the isotropic Gaussian is the embedding distribution that **minimizes worst-case downstream prediction risk** (for both linear and nonlinear probes). It also rules out collapse for free: every direction has unit variance, no dimension is redundant, and the distribution is maximum-entropy for its scale.

**SIGReg** (*Sketched Isotropic Gaussian Regularization*) is the differentiable training objective that pushes embeddings toward $\mathcal{N}(0, I_D)$. It is built in two steps:

1. **Sketching/slicing** — reduce the $D$-dimensional test to many random 1-D tests (Cramér–Wold).
2. **A 1-D goodness-of-fit test** — compare each projected sample against $\mathcal{N}(0,1)$ via its characteristic function (Epps–Pulley).

The full LeJEPA loss is then a convex combination with a single trade-off hyperparameter $\lambda$:

$$
\mathcal{L}_{\text{LeJEPA}} \;=\; \lambda \cdot \mathcal{L}_{\text{SIGReg}} \;+\; (1-\lambda)\cdot \mathcal{L}_{\text{pred}}.
$$

---

## 2. The math

### 2.1 Step 1 — Slicing (Cramér–Wold)

Testing a $D$-dimensional distribution directly is expensive (multivariate normality tests like BHEP or Henze–Zirkler are quadratic in batch size; the repo ships them in `lejepa/multivariate/` for comparison). The **Cramér–Wold theorem** says a distribution on $\mathbb{R}^D$ is uniquely determined by its one-dimensional projections:

$$
z \sim \mathcal{N}(0, I_D)
\quad\Longleftrightarrow\quad
\langle a, z\rangle \sim \mathcal{N}(0, 1)\ \text{ for every unit vector } a \in \mathbb{S}^{D-1}.
$$

(The forward direction: a linear functional of a Gaussian is Gaussian with variance $a^\top I_D\, a = \lVert a\rVert^2 = 1$.)

So instead of one hard $D$-dimensional test, SIGReg draws $K$ random unit directions $a_1, \dots, a_K$ (the "sketch"), projects the batch onto each, and applies a cheap 1-D test per slice:

$$
\mathcal{L}_{\text{SIGReg}}
\;=\; \frac{1}{K}\sum_{k=1}^{K} T\!\big(\{\langle a_k, z_n\rangle\}_{n=1}^N\big),
$$

where $T$ is a univariate test statistic of departure from $\mathcal{N}(0,1)$. A finite $K$ at a single step only tests $K$ directions — but the directions are **resampled fresh at every training step**, so over the course of SGD the slices cover the sphere and no direction can hide a deviation indefinitely. (The paper bounds the error of the sliced approximation in terms of $K$; more slices = lower-variance estimate, at linear cost.)

### 2.2 Step 2 — The 1-D test: Epps–Pulley via characteristic functions

For each slice, let $x_1, \dots, x_N$ be the projected batch. The **empirical characteristic function (ECF)** is

$$
\hat\varphi(t) \;=\; \frac{1}{N}\sum_{n=1}^N e^{\,i t x_n}
\;=\; \underbrace{\frac{1}{N}\sum_n \cos(t x_n)}_{\text{real part}}
\;+\; i\,\underbrace{\frac{1}{N}\sum_n \sin(t x_n)}_{\text{imaginary part}},
$$

and the characteristic function of the target $\mathcal{N}(0,1)$ is real-valued:

$$
\varphi(t) \;=\; e^{-t^2/2}.
$$

The **Epps–Pulley statistic** is the weighted squared CF discrepancy (docstring of `EppsPulley`, `epps_pulley.py:24–28`):

$$
T \;=\; N \int_{-\infty}^{\infty} \big|\hat\varphi(t) - \varphi(t)\big|^2\, w(t)\, dt,
\qquad w(t) = e^{-t^2/2}.
$$

Expanding the modulus (the target CF is real):

$$
\big|\hat\varphi(t) - \varphi(t)\big|^2
\;=\; \Big(\tfrac{1}{N}\textstyle\sum_n \cos(t x_n) - e^{-t^2/2}\Big)^2
\;+\; \Big(\tfrac{1}{N}\textstyle\sum_n \sin(t x_n)\Big)^2 .
$$

Intuition for the two terms:

- **Real part** captures the *even* structure of the slice distribution — spread, tails, multimodality. A collapsed slice ($x_n \equiv \mu$) gives $\operatorname{Re}\hat\varphi(t) = \cos(t\mu) \neq e^{-t^2/2}$, so collapse is heavily penalized.
- **Imaginary part** is $\mathbb{E}[\sin(t x)]$, which vanishes for any distribution symmetric about $0$. Penalizing it enforces zero mean and symmetry. Note the embeddings are **not centered first** — the test enforces the full $\mathcal{N}(0,1)$, including location.

$T = 0$ in population iff $\hat\varphi \equiv \varphi$, i.e. iff the slice is exactly $\mathcal{N}(0,1)$ (a CF determines the distribution). Combined with Cramér–Wold over all directions, the population minimum of SIGReg is attained exactly at $z \sim \mathcal{N}(0, I_D)$.

**Why a CF test instead of an ECDF test** (Kolmogorov–Smirnov, Anderson–Darling, Cramér–von Mises — all also available in `lejepa/univariate/`)?

- $\cos$ and $\sin$ are smooth with **bounded derivatives**, so the per-sample gradient $\partial T / \partial x_n$ is bounded — no exploding gradients from outliers, stable in bf16, no gradient clipping needed.
- No sorting. ECDF tests require a sort, which is gradient-hostile (piecewise constant ranks) and awkward to distribute. The CF test is three broadcasted tensor ops.
- **Linear time and memory**, $O(N \cdot K \cdot P)$ for $N$ samples, $K$ slices, $P$ quadrature points — vs. $O(N^2)$ for kernel-based multivariate tests.
- DDP-friendly: the ECF is a *mean over samples*, so a single `all_reduce` of $\cos$/$\sin$ means gives the exact global-batch statistic (§3.3).

### 2.3 The quadrature: trapezoid on $[0, t_{\max}]$ with a symmetry trick

The integral is computed numerically. Two observations cut the work in half and bound the domain:

**Symmetry.** For real data, $\hat\varphi(-t) = \overline{\hat\varphi(t)}$, and the target $e^{-t^2/2}$ and weight $w(t)$ are even — so the integrand $|\hat\varphi(t)-\varphi(t)|^2 w(t)$ is an **even function of $t$**. It suffices to integrate over $[0, t_{\max}]$ and double.

**Truncation.** The Gaussian weight $w(t)=e^{-t^2/2}$ makes the integrand negligible beyond $|t| \approx 3$ ($w(3) \approx 0.011$), so the default integration domain is $[0, 3]$.

The implementation uses the trapezoidal rule with $P = 17$ equispaced points $t_p \in [0, 3]$, $\Delta t = 3/16$. Folding the doubling into the weights (`epps_pulley.py:72–78`):

$$
T \;\approx\; N \sum_{p=1}^{P} \omega_p\, e^{-t_p^2/2}\,
\big|\hat\varphi(t_p) - e^{-t_p^2/2}\big|^2,
\qquad
\omega_p =
\begin{cases}
\Delta t & p \in \{1, P\} \ \ (t=0 \text{ and } t=t_{\max})\\[2pt]
2\,\Delta t & \text{otherwise.}
\end{cases}
$$

This is exactly the trapezoid rule for the symmetric extension to $[-t_{\max}, t_{\max}]$: an interior point of the full grid gets weight $\Delta t$; doubling for symmetry gives $2\Delta t$; the shared point $t=0$ must not be doubled, and the endpoints $\pm t_{\max}$ carry trapezoid half-weights $\Delta t/2$ each, totalling $\Delta t$. As `MINIMAL.md` puts it: *"we leverage the symmetric property of the ECF/CF to improve the quadrature efficiency (integrate on `[0, t_max]` and double, instead of integrating on `[-t_max, t_max]`) — improved quadrature for free."*

**The $\times N$ factor** is the classical normalization of ECF-based test statistics: $\hat\varphi - \varphi = O_p(N^{-1/2})$ under the null, so $N\,|\hat\varphi-\varphi|^2 = O_p(1)$ — the statistic has a proper non-degenerate null distribution, making its value comparable across batch sizes. As a loss, it means a fixed distributional deviation contributes proportionally to how many samples certify it.

---

## 3. The code

### 3.1 `EppsPulley` (`lejepa/univariate/epps_pulley.py`)

Constructor — everything that doesn't depend on the data is precomputed once into buffers:

```python
def __init__(self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"):
    super().__init__()
    assert n_points % 2 == 1
    t = torch.linspace(0, t_max, n_points, dtype=torch.float32)   # quadrature nodes, t >= 0
    self.register_buffer("t", t)
    dt = t_max / (n_points - 1)
    weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
    weights[[0, -1]] = dt                                          # no doubling at t=0; half-weights at ±t_max
    self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())   # φ(t) = e^{-t²/2}
    self.register_buffer("weights", weights * self.phi)            # fold weight w(t)=e^{-t²/2} into quadrature
```

Note `phi` plays a double role, matching the math: it is the **target CF** in the residual and the **integration weight** $w(t)$ folded into `weights`.

Forward — input is `(*, N, K)`: N samples, K slices (or K raw dimensions if used without slicing):

```python
def forward(self, x):
    N = x.size(-2)
    x_t = x.unsqueeze(-1) * self.t          # (*, N, K, P)   — t·x for every sample/slice/node
    cos_vals = torch.cos(x_t)
    sin_vals = torch.sin(x_t)

    cos_mean = cos_vals.mean(-3)            # (*, K, P)      — Re φ̂(t), mean over the local batch
    sin_mean = sin_vals.mean(-3)            # (*, K, P)      — Im φ̂(t)

    cos_mean = all_reduce(cos_mean)         # DDP: average ECF across GPUs (differentiable)
    sin_mean = all_reduce(sin_mean)

    err = (cos_mean - self.phi).square() + sin_mean.square()   # |φ̂(t) − φ(t)|², symmetry already in weights

    return (err @ self.weights) * N * self.world_size          # (*, K)  — one statistic per slice
```

Shape flow: `(N, K)` → broadcast against `P` nodes → `(N, K, P)` → mean over the batch dim → `(K, P)` ECF values → quadrature contraction with `weights` → `(K,)` statistics. The `* N * self.world_size` is the $\times N$ scaling of §2.3 with $N$ being the **global** batch size under DDP.

(A `DeprecatedEppsPulley` with explicit complex arithmetic and a two-sided grid lives in the same file for reference; the tests in `tests/test_epps_pulley.py` exercise that interface.)

### 3.2 `SlicingUnivariateTest` (`lejepa/multivariate/slicing.py`)

Wraps **any** univariate test (Epps–Pulley, Anderson–Darling, Cramér–von Mises, … — anything in `lejepa/univariate/`) into a multivariate one via random projections:

```python
def forward(self, x):                                   # x: (*, N, D)
    with torch.no_grad():
        # Synchronize the RNG seed across all DDP ranks
        global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
        seed = global_step_sync.item()
        g = self._get_generator(x.device, seed)         # cached per-device generator

        A = torch.randn((x.size(-1), self.num_slices), device=x.device, generator=g)
        A /= A.norm(p=2, dim=0)                          # unit-norm columns: D×K directions on the sphere
        self.global_step.add_(1)                         # fresh directions next step

    stats = self.univariate_test(x @ A)                  # project → (*, N, K) → per-slice statistics (*, K)
    if self.clip_value is not None:
        stats[stats < self.clip_value] = 0               # optional noise floor
    if self.reduction == "mean":
        return stats.mean()
    elif self.reduction == "sum":
        return stats.sum()
    elif self.reduction is None:
        return stats
```

Key details:

- **Directions are sampled, normalized, and used inside `no_grad`** — they are constants; gradients flow only through the embeddings via `x @ A`.
- **Seed synchronization**: `global_step` is a buffer, all-reduced with `MAX` before seeding, so every DDP rank draws the **same projection matrix** `A`. This is required for §3.3 to be correct — the per-rank ECF means are only averageable if all ranks projected onto the same directions.
- **Fresh directions every call** via the `global_step` increment — the stochastic sweep of the sphere from §2.1.
- Gaussian directions normalized to unit norm = uniform on the sphere $\mathbb{S}^{D-1}$.

### 3.3 Distributed training

The ECF is an average over samples, so for a global batch sharded across $W$ GPUs:

$$
\hat\varphi_{\text{global}}(t) \;=\; \frac{1}{W}\sum_{r=1}^{W} \hat\varphi_{\text{rank } r}(t).
$$

`EppsPulley.forward` therefore all-reduces `cos_mean`/`sin_mean` with `AVG` using `torch.distributed.nn.all_reduce` (`epps_pulley.py:8–13`) — the **differentiable** functional collective, so gradients flow back through the reduction to every rank's embeddings. The statistic each rank computes is then exactly the global-batch statistic (hence `N * self.world_size` = global sample count). This is what lets SIGReg use the *whole* effective batch for the normality test at the cost of communicating just `2·K·P` scalars per step — no gathering of embeddings.

### 3.4 The 20-line self-contained version (`MINIMAL.md`)

The minimal training example inlines slicing + Epps–Pulley into a single module — useful as the "rosetta stone" for the math above:

```python
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)       # quadrature nodes on [0, 3]
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt                                       # symmetry-folded trapezoid
        window = torch.exp(-t.square() / 2.0)                       # φ(t) = e^{-t²/2}
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):                                        # proj: (V, N, D) — views, samples, dims
        A = torch.randn(proj.size(-1), 256, device="cuda")          # 256 fresh random directions
        A = A.div_(A.norm(p=2, dim=0))                              # unit columns
        x_t = (proj @ A).unsqueeze(-1) * self.t                     # (V, N, 256, P)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)            # ×N
        return statistic.mean()                                     # mean over views and slices
```

Line-for-line this is §2.2 + §2.3: project (Cramér–Wold sketch) → ECF on the quadrature grid → squared residual to the Gaussian CF → weighted quadrature → $\times N$ → average over slices. Note it is applied **per view**: the batch mean in `.mean(-3)` runs over samples within each view, so every view's embedding distribution is independently pushed to $\mathcal{N}(0, I)$.

### 3.5 How it sits in the LeJEPA training loss

From the minimal example's training step (`MINIMAL.md`), with `proj` of shape `(V, N, D)` holding the projector outputs of $V$ views:

```python
emb, proj = net(vs)
inv_loss    = (proj.mean(0) - proj).square().mean()        # prediction: each view → mean over views
sigreg_loss = sigreg(proj)                                  # SIGReg on the same projections
lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
```

- The **prediction/invariance term** pulls all views of the same image toward their average (no stop-gradient, no teacher).
- **SIGReg** prevents the trivial solution where everything is pulled to a single point — `inv_loss` would be 0 there, but the embedding distribution would be maximally non-Gaussian.
- $\lambda$ (`lamb`, e.g. `0.02` in the minimal example) is the **only** trade-off hyperparameter.

### 3.6 Library usage (`README.md`)

```python
import lejepa

univariate_test = lejepa.univariate.EppsPulley(n_points=17)        # the 1-D test
loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test,
    num_slices=1024,                                                # K
)
loss = loss_fn(embeddings)    # embeddings: (num_samples, num_dims)
loss.backward()
```

---

## 4. Properties at a glance

| | |
|---|---|
| Population minimum | $\mathcal{L}=0$ iff every slice is exactly $\mathcal{N}(0,1)$ ⇔ $z \sim \mathcal{N}(0, I_D)$ (Cramér–Wold) |
| Complexity | $O(N \cdot K \cdot P)$ time and memory — linear in everything; no sort, no $N{\times}N$ kernel, no eigendecomposition |
| Gradients | bounded per-sample (derivatives of $\cos/\sin$), smooth — stable in bf16 without clipping |
| DDP | exact global-batch statistic via one differentiable `all_reduce` of $2 K P$ scalars; projection seeds synced across ranks |
| Hyperparameters | $K$ slices (256 in the minimal example, 1024 in the README), $P=17$ nodes on $[0,3]$, trade-off $\lambda$ |
| Stochasticity | fresh random directions each step → unbiased coverage of the sphere over training |
| Centering | none — the imaginary-part term enforces zero mean as part of the test |
| vs. VICReg | VICReg constrains only first/second moments; SIGReg constrains the **full distribution** through its CF (the repo's `univariate.VCReg` / `ExtendedJarqueBera` are the moment-based counterparts in the same framework) |
| Modularity | `SlicingUnivariateTest` accepts any 1-D test from `lejepa/univariate/`; Epps–Pulley is the recommended default for its gradient properties |

## 5. References

- R. Balestriero & Y. LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), 2025.
- T. W. Epps & L. B. Pulley, *A test for normality based on the empirical characteristic function*, Biometrika, 1983.
- Cramér–Wold device; sliced distribution comparison: Rabin et al. 2012, Bonneel et al. 2015 (cited in `slicing.py` docstring).
