# foerster against PyMC, Gen.jl and raster

Matched models, the same data (`data/`, made by `make_data.py`), and exact
answers to check against. Machine: 8 cores, 31 GB, JVM 25 (GraalVM), Julia
1.11.6, PyMC 5.28.4. Run order and commands are at the top of each script.

| | workload | exact reference |
|---|---|---|
| W1 | bootstrap particle filter, Gaussian random walk, T = 300 | Kalman log evidence |
| W2 | NUTS, radon varying intercepts (89 parameters), 4 × 1000 draws after 1000 warm-up | 2-D grid over the scales, the rest in closed form |
| W3 | random-walk MH (sd 0.02, one site per move), golf logistic, 4 chains × 22 000 moves | 2-D grid |
| W4 | tempered SMC, golf logistic, 2000 particles | 2-D grid evidence (−191.932) |

## Results (2026-10-05, foerster main with spindel 0.1.103)

W1 particle filter — particle-steps/s (N = 100 / N = 1000), log-evidence error (mean ± sd over seeds, N = 1000):

| system | N = 100 | N = 1000 | logZ error |
|---|---|---|---|
| numpy, vectorized (`pymc/pf_numpy.py`) | 2.4 M | 11.3 M | −0.07 ± 0.13 |
| raster `deftm` (`raster/pf.clj`) | 2.9 M | 6.9 M | −0.01 ± 0.19 |
| Gen static `Unfold` | 73 k | 66 k | −0.01 ± 0.41 |
| foerster `:smc` | 16.5 k | 15.7 k | +0.06 ± 0.18 |
| Gen dynamic DSL (re-executes per step) | 2.1 k | — | — |

W2 NUTS radon — time (warm / first run incl. compilation), ESS/s:

| system | warm | first | β ESS/s | σ_a ESS/s | β mean (exact −0.663) |
|---|---|---|---|---|---|
| PyMC | 4.1 s | 25.4 s | 737 | 276 | −0.661 |
| foerster (block, hand gradient) | 15.0 s | 17.1 s | 283 | 102 | −0.663 |

W3 random-walk MH golf — time, ESS of a, ESS/s (exact a = 2.2247 ± 0.0583):

| system | time | ESS(a) | ESS/s | mean a |
|---|---|---|---|---|
| Gen (same proposal) | 3.2 s | 653 | 207 | 2.228 |
| PyMC Metropolis (tuned proposal) | 10.0 s | 2277 | 227 | 2.224 |
| foerster `:rmh` | 37.2 s | 483 | 13 | 2.228 |

W4 tempered SMC golf — time, log evidence (exact −191.932):

| system | time | log evidence |
|---|---|---|
| PyMC `sample_smc` (4 runs × 2000) | 1.4 s | −191.94 … −192.03 |
| foerster `:tempered` (1 run × 2000) | 8.3 s | −192.19 |

## Reading

- raster alone runs a compiled filter at numpy speed, ~40× Gen.
- foerster's particle filter is 4.5× slower than Gen's incremental Unfold
  and 7× faster than Gen's dynamic DSL; per-site world bookkeeping.
- foerster's MH is 12× slower per move than Gen: every move replays the
  program suffix through savepoints. The largest gap, and structural.
- foerster's NUTS is 2.6× slower than PyMC warm (faster cold): the loop,
  not the gradient.
