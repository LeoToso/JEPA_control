# Mirror loss ($\mathcal{L}_\text{mirror}$) — derivation

> **Status:** currently disabled (`lambda_mirror = 0`) in all active configs — the switch to a CNN encoder with `frame_stack=2` changed how sign/direction is captured. The term remains implemented in the full `Trainer`; the minimal trainer (`training/trainer_minimal.py`) omits it entirely. This doc explains what it does when enabled.

`L_mirror` enforces a known symmetry of the cartpole: mirroring the image left↔right corresponds to negating the physical state. It injects the *direction* (sign) of the state into the latent without any state labels.

Implementation: `training/trainer.py:501–507`. Applied to the t=0 frame of **every** trajectory in the batch (all states, not just equilibrium); `z*` is the fixed reflection center, not an input.

$$
\mathcal{L}_\text{mirror} = \big\lVert \mathrm{enc}(o) + \mathrm{enc}(\mathrm{flip}(o)) - 2z^* \big\rVert^2
$$

## 1. What flip does to the state

For an image $o$ showing state $s = (x, \dot x, \theta, \dot\theta)$, the left↔right mirror `flip(o)` is a physically valid frame in the **negated** state:

$$
\text{flip}: \quad s = (x, \dot x, \theta, \dot\theta) \;\longmapsto\; -s = (-x, -\dot x, -\theta, -\dot\theta).
$$

## 2. The equilibrium is the flip's fixed point

The upright, centered configuration is its own mirror image, $\text{flip}(o^*) = o^*$, with state $s = 0$ (and $-0 = 0$). The encoder maps it to $z^*$ by definition.

## 3. The desideratum: antisymmetric latent displacement

Define the latent displacement from equilibrium

$$
\delta(o) \;=\; \mathrm{enc}(o) - z^*.
$$

Since flipping negates the state ($s \mapsto -s$), a sign-faithful latent should negate its displacement under the flip:

$$
\delta(\text{flip}(o)) \;=\; -\,\delta(o).
$$

## 4. That condition *is* the loss target

Writing out both displacements and adding them,

$$
\mathrm{enc}(o) - z^* = \delta, \qquad \mathrm{enc}(\text{flip}(o)) - z^* = -\delta,
$$

$$
\mathrm{enc}(o) + \mathrm{enc}(\text{flip}(o)) - 2z^* = 0
\;\;\Longrightarrow\;\;
\mathrm{enc}(o) + \mathrm{enc}(\text{flip}(o)) = 2z^*.
$$

So "$= 2z^*$" just states that **$z^*$ is the midpoint of $\mathrm{enc}(o)$ and $\mathrm{enc}(\text{flip}(o))$** — the frame and its mirror sit on opposite, equidistant sides of the equilibrium latent.

```
        enc(flip(o)) •────────•────────• enc(o)
                            z*
              (the two are reflections of each other through z*)
```

## 5. Why this term earns its keep

It forbids the **sign-blind** encoder. If the encoder mapped a frame and its mirror to the same point — encoding only the tilt magnitude $|\theta|$, not its direction — then $\mathrm{enc}(o) = \mathrm{enc}(\text{flip}(o)) = v$, and the constraint forces $2v = 2z^* \Rightarrow v = z^*$, i.e. total collapse onto $z^*$, which the prediction loss won't allow. To satisfy $\mathcal{L}_\text{mirror}$ *without* collapsing, the encoder must place $+\theta$ and $-\theta$ on opposite sides of $z^*$ — i.e. encode the sign. This breaks the $|\theta|$-vs-$\theta$ degeneracy that is otherwise tempting near equilibrium, where $\pm\theta$ frames look nearly identical.

## Notes

- **Computed batch-wide, not at equilibrium only.** At the equilibrium the constraint is trivially satisfied ($z^* + z^* = 2z^*$, zero gradient); all the real signal comes from non-equilibrium frames where $o$ and $\text{flip}(o)$ are genuinely different images of $+s$ and $-s$.
- **Global assumption.** The antisymmetry is enforced over the whole data distribution (all angles/positions), not just near equilibrium — justified here because the cartpole's flip symmetry is exact. Unlike $\mathcal{L}_\text{local}$ / $\mathcal{L}_\text{temp}$, it is not a near-equilibrium-only term.
- **Label-free.** The sign structure comes purely from the image symmetry; no ground-truth state is used.
