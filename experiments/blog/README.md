# Scripts behind the simm.is post on probabilistic programming

[Probabilistic programming for Clojure programmers](https://simm.is/blog/write-the-model-once/)
explains probabilistic programs as constraint programs with soft constraints and runs one
forecasting model under several inference methods.  These scripts produce its numbers.
Run them from the repository root.

| Script | What it computes | Time on a laptop |
|---|---|---|
| `dice.clj` | the two dice under exact enumeration: a hard constraint (sum 7) and a soft one (a noisy reading of 7) | seconds |
| `demand.clj` | the demand model: SMC and importance sampling over ten seeds, four MH chains, PMMH for the drift, and the per-week SMC band of the figure | about 10 min |
| `particle_gibbs.clj` | the fix for MH: four particle Gibbs chains, R-hat and bulk ESS of the final and the first week's level | about 4 min |

The data are twenty simulated weeks (level starting at 100, drift 4, noise 8), listed in the
scripts.  The exact answers come from a Kalman filter: final level 102.89, log evidence
−77.49, and for an unknown drift with a Gamma(2, scale 2.5) prior a posterior mean of 2.85
with a 90% interval from 0.61 to 6.28 (quadrature over the filter's evidence).

Results on main, 2026-10-15 (foerster 0.1.105):

| Method | Final level | Log evidence | Check |
|---|---|---|---|
| SMC, 1000 particles, 10 seeds | 102.92 ± 0.23 | −77.44 ± 0.14 | 114 to 134 distinct histories at week 0 |
| Importance sampling, 1000, 10 seeds | 102.9 ± 1.8 | −77.7 ± 0.5 | 2 to 10 effective samples |
| MH, 4 chains × 4000, burn 1000 | 103.4 | | R-hat 1.03, bulk ESS 114 |
| Particle Gibbs, 4 chains × 600, 50 particles, burn 100 | 102.93 | | R-hat 1.002, bulk ESS 1508 (week 0: 1.002, 659) |
| PMMH, drift, 2000 iterations, 100 particles | drift 2.82, 90% 0.61–6.09 | | 5.4 min |
