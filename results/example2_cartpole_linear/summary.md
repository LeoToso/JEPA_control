# cartpole_linear

true eigvals(A): [1.         1.         1.09736611 0.9112729 ]

spectral radius: 1.0974  (open-loop unstable)

| config | stabilizable (latent) | unstable-mode R^2 | success rate | mean frac stable | final state dist |
|---|---|---|---|---|---|
| pred_only | True | -0.003 | 0.0% | 0.276 | 584.8627 |
| pred_sigreg | True | 0.048 | 0.0% | 0.276 | 560.4367 |
| pred_actrecon | True | 1.000 | 100.0% | 1.000 | 0.0339 |
| oracle | True | 1.000 | 100.0% | 1.000 | 0.0241 |
