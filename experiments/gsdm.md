# Native GSDM amortized abduction: CPU pilot

This experiment connects Raster's graph transformer to Foerster's causal
world API. It is research code, outside the library dependency tree. No
Python package is used for training, sampling, or validation.

## Model and reference answer

Independent standard Gaussian disturbances generate

```
z = uz
x = z + ux
y = 2x + z + uy
```

Only x and y are observed. The network receives three variables:
`[z, x, y-2x]`, with z latent and the other two clamped. The residual feature
is an invertible transformation of the observed pair, using the declared
structural coefficient; it is not an inferred latent value or a posterior
oracle. This is deliberately an easy first problem with a fixed conditioning
mask, not a demonstration of arbitrary simulators or automatic graph extraction.

The exact posterior is `z | x,y ~ Normal((y-x)/3, sqrt(1/3))`.
For every sampled z, reconstruct `ux=x-z`, `uy=y-2x-z`. Foerster then replays
the model under interventions with those factual disturbances:

- `do(z=0)` gives `y_cf=y-3z`, a nontrivial counterfactual distribution.
- `do(x=0)` gives `y_cf=y-2x`, a deterministic counterfactual here.

GSDM is trained on fresh joint simulator draws, without reference-posterior
samples or oracle denoising targets. Analytic formulas are used for evaluation
and explicitly labeled oracle controls only. Equal-weight learned abductions
remain approximate; exact structural replay does not make them exact posterior
draws.

## Architecture and execution

- CPU, Raster JVM-AOT, double precision, JDK 25. No GPU or native C backend
  was selected. `params/compile-train-step` fuses the forward, backward and
  Adam updates; `params/compile-aot` compiles prediction separately.
- Two GSDM blocks: residual MLP and graph attention, 16-dimensional
  embeddings, two attention heads, timestep MLP and value/state/position
  embeddings. **4,401 trainable parameters**.
- Seven directed attention edges, including self-edges, on three nodes.
  These are real edge-indexed operations, but this tiny graph establishes
  neither sparse scaling nor large-problem performance. Normalization and
  learned message passing must not be interpreted as proofs of independence.
- 64-step cosine schedule; fixedlarge DDPM reverse variance. Both x0 and v
  prediction were tried. The v variant is implemented in this experiment's
  training-target and sampling adapter. Raster's public `gsdm-sample` still
  expects x0 predictions.
- A resident REPL retains compiled functions. Initialization and optimizer
  arrays can be reset with `fresh` without compiling the architecture again.

## Recorded results

The [summary](results/gsdm/summary.edn) records settings, timings, source
provenance, numeric checks and statistical diagnostics. The
[inference checkpoint](results/gsdm/consolidated.edn) includes the trained
weights. Earlier failures and intermediate checkpoints are retained beside it.

| Run | Absolute mean error over four fixed contexts | Posterior variance / reference variance |
|---|---:|---:|
| x0, 2 blocks / 16 dimensions, 100,000 updates | 0.038–0.293 | 0.580–1.005 |
| v, same architecture and 100,000 updates | 0.043–0.094 | 0.889–1.066 |
| v, another 100,000 updates at lower rate | 0.003–0.053 | 0.984–1.019 |

The matched 100,000-update runs use the same initialization, simulator seed,
learning rate and diffusion schedule. This is one training realization, not
evidence of general objective superiority. The earlier 1-block / 8-dimension
x0 runs also show why falling denoising loss is insufficient.

At the final v checkpoint:

- Compilation/preparation took 22.8 seconds. The two training chunks took
  29.3 and 27.5 seconds. These are local single-run CPU observations.
- At x=1, y=2.5, 2,048 posterior draws give mean **0.5285** and variance
  **0.33394**, versus **0.5** and **1/3** analytically.
- On 64 fresh simulator cases with 128 posterior draws each, central 90%
  empirical coverage is **87.5%**, conditional-mean RMSE **0.0601**, and
  average variance ratio **1.069**. This small calibration check is not a
  certification. The oracle denoiser on the same cases has 85.9% coverage,
  mean RMSE 0.0492 and variance ratio 1.023, illustrating Monte Carlo variation.
- Actual Foerster replay over 128 learned abductions is completely coupled;
  maximum disagreement with the independent SCM equation is **8.9e-16**.
- Thirteen coordinates of the actual fused training gradient agree with
  independent finite differences, maximum scaled error **9.8e-10**. Forward
  MSE and the training step's returned loss agree exactly at the audited point.
- Raster's public x0 sampler and the compiled experiment sampler produce
  **identical seeded draws** in the 16-draw check; observed values stay clamped.
- Restoring the checkpoint reproduces 32 seeded draws exactly. Optimizer
  moments are not saved, and the restored checkpoint refuses resumed training.
- Independent quadrature, diffusion-target and causal-adapter tests pass:
  **4 tests, 36 assertions**.

## Reproduce and iterate

From the Foerster checkout, with `../raster` available:

```sh
# Full recorded v training recipe and held-out checks, fresh process:
clojure -M:gsdm -m org.replikativ.foerster.gsdm /tmp/gsdm-reproduction.edn

# Independent model/adapter tests; no network training required:
clojure -M:gsdm:test -d experiments -n org.replikativ.foerster.gsdm-test

# Or start a resident REPL with both projects:
clojure -M:gsdm:repl
```

```clojure
(require '[org.replikativ.foerster.gsdm :as experiment])

;; Compile once, then iterate without another compiler invocation.
(def system (experiment/prepare {:prediction :v :emb-dim 16 :n-layers 2}))
(def restored
  (experiment/restore system "experiments/results/gsdm/consolidated.edn"))
(experiment/posterior-check restored {:x 1.0 :y 2.5} 2048 304)
(experiment/check-causal-query! restored {:x 1.0 :y 2.5} 128 501)
(experiment/numerical-audit system 801)

;; Same compiled architecture, independent parameters and optimizer state.
(def comparison (experiment/fresh system :x0))
(experiment/train! comparison 100000 {:seed 103 :rate 0.0003})
(experiment/public-sampler-check comparison {:x 1.0 :y 2.5} 16 601)
```

The optional `:gsdm` alias uses the local Raster checkout. Its tracked dirty
diff fingerprint was unchanged during the recorded runs. This does not pin
an immutable Raster release; source revisions and the fingerprint are in the
receipt. Reproduction against ongoing Raster changes needs the same numerical
and statistical checks. The resident session for this experiment uses port
41465, recorded in `/tmp/foerster-gsdm-repl-port`.

## Consolidation route for Raster

The CPU checks provide an initial regression contract:

1. Keep public and compiled forward/sampling paths consistent under identical
   weights, data and RNG streams. Expose a compiled forward model directly;
   this experiment currently reuses the prediction expression from the public
   `gsdm-loss-body` generator.
2. Make prediction parameterization and forward/reverse schedules explicit
   across training and sampling. Move v support upstream only with matched
   tests; the state embedding alone does not implement intervention semantics.
3. Preserve independent value/gradient checks, inference checkpoint replay,
   held-out conditional accuracy and calibration. Check these after compiler
   and kernel changes, rather than relying on a short loss-reduction test.
4. For larger graphs, qualify GPU values/gradients against CPU before comparing
   training or posterior results. Record precision, transfers, resident state,
   compilation and synchronization costs. **GPU is not tested by this pilot.**

Next scientific checks include multiple seeds, raw observation inputs, varying
conditioning masks, larger sparse systems and a nonlinear/non-Gaussian problem.
The current tests do not establish general marginalization or causal inference
through Raster's `INTERVENED` embedding; causal actions use Foerster's SCM replay.

## Densities, free-form flows and SBI

The DDPM adapter supplies samples, not a normalized marginal log density for
those samples. It is not registered as an importance proposal distribution.

[Free-form flows](https://proceedings.mlr.press/v238/draxler24a.html) are a
promising alternative for keeping the graph architecture while changing the
training objective. Their learned encoder/decoder inverse relation must be
checked: training's efficient log-determinant-gradient estimator is not itself
an evaluated proposal log density. In small dimensions, explicit Jacobians,
inverse consistency and density normalization are feasible checks. Scoring an
encoder density does not automatically score samples of an imperfect decoder.
Likewise, a probability-flow ODE density is associated with that continuous
flow and its numerical implementation, not automatically with the finite-step
DDPM sampler above.

The local `../sbi` checkout is a reference for amortized versus sequential
inference, posterior predictive checks, SBC, TARP and local conditional tests.
Its vector-field posterior `log_prob` explicitly builds a probability-flow ODE.
SBI is not a runtime dependency, and no cross-package performance comparison
was performed here.
