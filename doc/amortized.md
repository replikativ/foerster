# Amortized inference from simulator source

Foerster can compile a fixed simulator instance into a conditional-sampling
task, derive its RV representations and attention mask, and train an optional
Raster GSDM backend. The deployment artifact samples an **approximate**
posterior. It does not require importance correction or claim an evaluated
posterior density.

## Write the simulator, then ask an inference question

An ordinary source-backed `defn` works:

```clojure
(require '[org.replikativ.foerster.effects :refer [sample]]
         '[org.replikativ.foerster.dist :as d]
         '[org.replikativ.foerster.amortized :as amortized]
         '[org.replikativ.foerster.amortized.compiler :as compiler]
         '[org.replikativ.foerster-raster.amortized :as gsdm]
         '[org.replikativ.spindel.effects.await :refer [await]]
         '[org.replikativ.spindel.spin.cps :refer [spin]])

(defn simulator []
  (spin
   (let [z (sample (d/normal 0 1) :id :z)
         x (sample (d/normal z 1) :id :x)
         y (sample (d/normal (+ (* 2 x) z) 1) :id :y)]
     {:z z :x x :y y})))

(def task (compiler/compile #'simulator {:targets [:z] :causal? true}))

;; These are derived from the program, not authored masks:
(:schema task)
(:attention-edges task)
(:compilation task)

(spin
 (let [fitted (await (amortized/train task (gsdm/backend)
                     {:simulations 4096 :seed 811
                      :stages [{:updates 100000 :rate 0.0003}
                               {:updates 100000 :rate 0.00005}]}))
       posterior (await (amortized/infer fitted {:x 1.0 :y 2.5} {:draws 512}))]
   (await (amortized/worlds fitted posterior {:x 1.0 :y 2.5}
            [[:z-zero {:z {:do 0.0}}]]))))
```

The body is ordinary Foerster code. In a REPL, where source for a `defn` may
not be available, use `(compiler/model [args] body...)`; like `analysis/model`,
this wrapper already supplies the surrounding spin and retains its source.
Existing `analysis/model` functions also work. Public `core/sample` and
`core/observe` aliases are recognized.

The JVM compiler uses Foerster's existing source analysis. Data dependencies
and each conditional factor's scope become attention connections, including
connections between co-parents. Opaque/escaping calls get conservative dense
connectivity when source analysis cannot justify sparsity. This is an
inference graph, not causal discovery. The compiler never establishes that
the program's dependencies are causal facts.

Named sites retain their IDs. For ordinary inference, unnamed sites may get
names from directly returned map fields, or generated site names. Known-size
loops are expanded with `:args`; hierarchical IDs such as `[:state i]` work.
Automatic causal replay initially requires explicit stable sample IDs.

## Observation masks and inference targets

Without `:targets`, every derived RV can be queried or conditioned on. With
`:targets [:z]`, z is always latent during training and the remaining RVs
become possible observations. `:known-inputs` marks variables that must always
be supplied. These are inference questions, not hand-written neural masks.

The default conditioning policy is `:any`. Training samples masks independently
of the data without enumerating all 2^n subsets. An explicit list of sets is
available when an application needs only a few masks:

```clojure
(compiler/compile #'simulator
  {:targets [:z] :conditionings [#{:x :y} #{:x}]})
```

Supporting a mask means the network was trained under that policy, not that
its approximation is accurate for every possible observation pattern. Evaluate
the patterns used by the application. Missing-not-at-random observations need
an explicit missingness model.

`:targets` on an inference request projects the learned joint's draws. Other
latent coordinates are retained internally for causal reconstruction. This is
a valid marginal of the learned distribution, but does not reduce neural work.
We do not yet remove nodes using a learned marginalization mask. Dropping a
node and its edges alone can lose induced dependencies.

## RV types and observation encoders

| Type | Numerical representation | Boundary |
|---|---|---|
| Real scalar or fixed tensor | real coordinates | fixed shape, finite values |
| Positive scalar/tensor | logarithm | strictly positive |
| Interval scalar/tensor | logit in declared bounds | interior values; numerical overflow is refused |
| Simplex | k-1 additive log ratios | strictly positive components summing to one |
| Finite domain, including booleans | one-hot, then argmax decoding | approximate continuous relaxation; no evaluated categorical mass |
| Embedding observation/input | versioned feature encoder | conditioning on features, no inverse string generator |

Common scalar distributions derive these representations automatically.
Some vector distributions with literal sizes are recognized. Unknown supports,
unbounded counts, symbolic shapes and custom distributions need an explicit
schema override. Values are checked against the fixed schema before training.
The GSDM backend records its finite-value relaxation; pass
`:discrete-targets :reject` to refuse latent finite variables.

For example, an observation encoder may call `pretrained.embed/embed-texts`
using an already loaded model from `../pretrained-rstr`. A variable descriptor
can carry:

```clojure
{:name :text :role :observation :type :embedding :shape [384]
 :encoder {:id :sentence-encoder :version "weights-and-pooling-version"
           :encode (fn [text] (encode-with-your-loaded-model text))}}
```

The callback may return nested vectors or native numeric arrays. Runtime encoder
code and model handles are rebound at load time; encoder identity and version
are part of the artifact contract. No encoder weights are downloaded by this
integration. High-dimensional embeddings currently expand into scalar tokens;
a projected context-token architecture is useful future work for efficiency.

Conditioning on a deterministic text embedding is a well-defined inference
problem over coarsened observations. It is not automatically equivalent to
conditioning on the full text; that depends on what information the embedding
retains. Generating strings or evaluating their likelihood is separate work.

## Simulator limits and the general boundary interface

Automatic source compilation currently needs a fixed stochastic site structure,
known loop sizes, readable distribution constructors (or schema overrides), and
no arbitrary factor-scored target. It checks runtime traces against the source
instance rather than guessing a schema from one branch. Randomness that changes
which sites exist requires a different representation.

Pass changing simulator configuration through `:args`. The source fingerprint
does not fully identify mutable globals, captured runtime state or external
implementations; these need explicit versioning and validation.

A simulator-boundary task can still have branching, native solvers or opaque
stochastic internals if its exported parameters/state/observations have fixed
shapes:

```clojure
(amortized/task simulator
  {:id :opaque-simulator/v1
   :schema [{:name :parameters :role :target :type :real :shape [4]}
            {:name :measurements :role :observation :type :real :shape [8]}]})
```

The simulator returns that map; a `:project` callback may adapt its result.
The task falls back to dense attention without supplied structure. This is
the escape hatch for outputs whose shapes/dependencies the source compiler
cannot yet derive, including many solver-produced return values. No simulator
gradient is required.

The collector draws original priors, ignores inference guides, and does not
evaluate sample-site densities. It therefore supports implicit distributions
that can draw but cannot score. Source compilation explicitly generates fresh
observations at observe sites. Boundary tasks opt in with
`:simulate-observed? true`. Fixed observations and arbitrary factors cannot
silently become an unweighted training dataset. A normalized external joint
example bank can be supplied with `:examples`.
Density/target blocks used by HMC are not automatically generative priors:
their draw capability may be only a proposal. They need a separate generative
simulator or joint example bank, and are refused by the collector.

Density-free simulation does not itself provide counterfactual noise replay.
Custom implicit distributions need a compatible structural mechanism or an
explicit causal reconstruction adapter for that additional capability.

Stay fixed-dimensional for now. A padded sequence plus presence indicators is
a possible next representation, but presence, observation and marginalization
masks have different semantics. Padding needs explicit loss/readout rules;
they are not implemented by silently treating padding as observed zero.
Trans-dimensional diffusion remains a separate research extension.

## Causal abduction and intervention inference

`:causal? true` is an explicit declaration of autonomous mechanisms. The
compiler then reconstructs noise from the learned factual assignments using
Foerster's mechanism protocols and replays the original program. A projected
query retains other learned latent draws so reconstruction does not lose them.
Source IDs must remain stable, and supported mechanism/noise couplings are
required. Fixed or deterministic assignments that contradict a learned draw
are refused; a custom free-coordinate adapter may be needed.

`amortized/worlds` evaluates counterfactuals inside the declared model. Complete
coupling does not make an approximate factual posterior exact. Backend-neutral
tasks can provide `:abduct` or asynchronous `:reconstruct` adapters and a
`:model` replay factory when source reconstruction is unsuitable.

Intervention inference can model a prior over intervention families directly:

```clojure
(defn intervention-simulator []
  (spin
   (let [intervention (sample (d/flip 0.5) :id :intervention)
         x (sample (d/normal (if intervention 2.0 0.0) 0.5) :id :x)
         y (sample (d/normal (* 3 x) 0.5) :id :y)]
     {:intervention intervention :x x :y y})))
```

This defines baseline and shifted-mechanism regimes. The compiler derives the
finite/continuous encoding and connectivity. Querying `:intervention` given
`:y` estimates which declared regime generated the observation. It is Bayesian
inference under that intervention prior, not proof of a unique cause. Unknown
targets and intervention strengths can be additional latent variables, with
their own priors. Observationally indistinguishable regimes remain ambiguous.

## Training budgets, validation and continuation

Validation here means checking the approximation, not applying importance
sampling correction. For a tractable simulator, compare conditional means,
variances and distributions with an analytic posterior or an independently
checked reference sampler, using the observation patterns the application needs.
For simulator-generated test cases, also examine posterior ranks, interval
coverage and fresh-observation predictions. Keep these cases separate from both
training and the denoising-validation bank used to select checkpoints.

Each check has limits. Prior-averaged rank/coverage checks alone can miss a
network that ignores the observations; predictive agreement can hide ambiguous
latent explanations. Conditional reference comparisons and application-specific
decision errors therefore matter alongside calibration. Choose tolerances for
the intended use and retain the individual results instead of reducing everything
to one training-loss threshold.

Importance weights would require a target density and the learned proposal
density. This GSDM adapter supplies samples without an evaluated proposal density,
and implicit simulators need not supply a likelihood either. A future flow with
a tractable proposal density would not, by itself, supply a missing simulator
likelihood. Importance correction is an optional additional algorithm when those
quantities are available, not a prerequisite for these empirical checks.

The recorded experiments have reference and predictive checks. Their deployment
status remains `:unvalidated` because no application-specific acceptance criterion
has been applied; this does not mean that no measurements were made. The observed
moment discrepancies are also a reason to avoid claiming a calibrated release.

Training completion is explicit. Inspect `:training-outcome`, `:deployment-status`
and `:warnings` on the returned artifact; inference diagnostics carry them too.
The backend receipt at `[:payload :training]` includes the validation history,
losses by observation mask, selected optimizer step and elapsed time.

```clojure
(spin
 (let [fitted (await (amortized/train task (gsdm/backend)
                     {:simulations 4096 :validation-simulations 256
                      :updates 100000 :validation-every 1000
                      :validation-cases 128 :patience 10 :min-delta 0.001
                      :max-seconds 120
                      :on-progress #(prn (select-keys % [:status :updates :validation-loss]))}))]
   {:outcome (:training-outcome fitted)
    :deployment (:deployment-status fitted)}))
```

The trainer stops with `:budget-exhausted`, `:time-limit`,
`:validation-plateau` or `:numerical-failure`. It never labels a denoising-loss
plateau as posterior convergence. Numerical failures restore a retained state
for diagnosis/continuation, but inference refuses the failed artifact.
The time limit is cooperative: it covers backend preparation, validation and
optimization, checked between updates; it cannot interrupt native compilation
or an update. Simulator-data generation happens before that backend budget.
Progress callbacks run synchronously at validation intervals and on completion;
callback failures propagate to the caller.

Generated training banks get 128 additional held-out simulator draws by default,
with non-overlapping simulator seeds. `:validation-simulations 0` disables them.
For supplied `:examples`, supply a separate `:validation-examples` bank; there
is no automatic split or assumption that supplied rows are independent.
Validation uses normalization fitted on training rows only, with fixed noise,
timesteps and observation masks, so measurements are comparable across updates.
The receipt reports losses by sampled mask and their case counts. With many
possible masks, a finite validation bank cannot cover every pattern.

By default the best validated-loss state under the improvement rule is retained,
including its Adam moments and step. `:min-delta` sets the required improvement and `:patience`
counts unsuccessful validation checks. Set `:restore-best? false` to retain the
last state instead. Reusing the validation bank for checkpoint selection can
overfit that bank; final posterior checks need separate data.

`(amortized/resume fitted {:updates 50000 :rate 0.00005})` returns a spin and
continues from the selected checkpoint. This also works after EDN export/thaw.
The same training/validation banks, simulator count, seed and validation-case
count must be reproduced; explicit banks must be supplied again. Resume checks
the task, data fingerprints and architecture, preserves normalization and Adam
state, and uses reproducible step-local randomness. Changing the learning-rate
schedule is allowed. Previously attached posterior validation is cleared after
further training. Older sampling-only checkpoints remain readable, but lack the
optimizer state needed for continuation.

Posterior validation is separate. `(amortized/validate fitted validator)` returns
a spin. The validator receives the artifact and returns a spin containing
`{:scope :posterior :passed? boolean :checks ...}`. It should declare its tests,
observation patterns, data, tolerances and results. It can check rank/coverage,
reference moments, posterior predictive behavior or application decisions.
Alternatively pass it as `:posterior-validator` to `train`.
Only such an explicit assessment sets `:deployment-status` to
`:validation-passed` or `:validation-failed`; otherwise it is `:unvalidated`.
Passing applies to the declared checks, not every observation or future dataset.
A caller attaching an exploratory report without a pass/fail criterion does
not silently certify the artifact. Failed posterior checks remain visible as
warnings; numerical failures block inference.

## Deployment and numerical backends

`train`, `infer`, `abduct`, and `worlds` return spins. `infer` returns an
`ApproximateMeasure` supporting ordinary Foerster summaries. Equal weights do
not certify accuracy; `log-marginal` returns nil rather than a fabricated
evidence estimate. Training information and optional validation accompany it.

For prospective control, `amortized/predictive` accepts a posterior and a
function from its target map to a new simulator spin. This explicitly binds
inferred parameters/state and draws fresh future randomness. The usual
trace-based `core/predictive` refuses trace-less learned draws instead of
silently reverting to the prior. Retrospective counterfactuals use `worlds`
and reconstructed factual noise instead.

`freeze` exports data: schema, identities, masks, transforms, learned parameters,
normalization, settings and validation. It rejects closures and device handles.
`thaw` binds the artifact to a matching task and backend, including encoder
versions and source fingerprint. The application supplies executable callbacks.
Attach exploratory reports with `(assoc fitted :validation report)` before
exporting. Use `amortized/validate` to update live pass/fail status as well.

Foerster's schema/interface is ClojureScript portable. The optional backend is
staged in [integration/](../integration/org/replikativ/foerster_raster/amortized.clj)
under the foerster-raster namespace and loaded by `:gsdm`; Raster is not a core
dependency. The current qualified backend is CPU/JVM-AOT, double precision.
Requests for other devices, compilation targets or float precision are rejected.
GPU and efficient resident training remain separate qualification work.

Compilation can be retained explicitly using `gsdm/prepare` and `:compiled`.
Reuse requires matching architecture **and exact attention layout**, not just
the same RV count. Reusing the earlier 7-edge engine on a 9-edge graph exposed
a vector-buffer bounds failure. Fresh compilation worked; a regression check
now refuses that incompatible reuse. There is no adapter-owned compiler cache.

## Experiments and current accuracy

[Executable examples](../experiments/org/replikativ/foerster/amortized_examples.clj)
train ordinary source-backed simulators without manual schemas or masks.
[Recorded artifacts](../experiments/results/amortized/) retain settings,
learned weights and diagnostics.

- The focused Gaussian artifact gives mean **0.524**, variance **0.268** for
  `z | x=1,y=2.5`, versus **0.5**, **1/3** analytically. With only x observed,
  it gives mean **0.546**, variance **0.478**, versus **0.5**, **0.5**. Its
  counterfactual mean is **0.929**, versus **1.0**. These are 512-draw results.
  The remaining underdispersion is material; this is not a calibrated release.
- The finite intervention artifact gives probabilities **0.008, 0.160, 0.512,
  0.838, 0.992** at y=0,2,3,4,6, versus **0.001, 0.083, 0.5, 0.917, 0.999**.
  Each uses 512 draws. The largest discrepancy is about 0.079 in this one run.
- The earlier [specialized GSDM pilot](../experiments/gsdm.md), with a fixed
  conditioning mask and a hand-chosen residual feature, is more accurate.
  Its accuracy must not be substituted for this more general integration.

Training error is useful for optimization. Held-out conditional distributions,
calibration, and application-specific control/decision quality remain distinct
checks. Neither the current experiments nor the API assert universal accuracy.

```sh
clojure -M:gsdm -m org.replikativ.foerster.amortized-examples gaussian-z /tmp/gaussian-z.edn
clojure -M:gsdm -m org.replikativ.foerster.amortized-examples interventions /tmp/interventions.edn
clojure -M:gsdm:test -d experiments -n org.replikativ.foerster.amortized-integration-test
```

## Loop-generated scaling qualification

[The state-space experiment](../experiments/org/replikativ/foerster/amortized_scaling.clj)
unrolls 2, 8 and 32 time steps into 4, 16 and 64 scalar RVs. It derives 3, 15
and 63 undirected dependencies, represented by 10, 46 and 190 directed attention
edges including self edges. Each model uses an 8-dimensional, two-layer,
two-head GSDM with 64 diffusion steps on CPU/JVM-AOT double precision.

Each run uses 2,048 training simulations, 128 independent denoising-validation
simulations and 30,000 updates. Posterior checks use another 16 simulations,
128 posterior draws per simulation, both complete and alternating sensor masks,
and exact Kalman/RTS smoothing marginals. Fresh replicated sensor observations
also test posterior predictive moments and interval coverage.

The 30,000-update runs all stopped with `:budget-exhausted`. These checks use
one training realization per size:

| RVs | Sensor observations | Mean RMSE | Mean variance / reference | Latent 90% coverage | Replicate 90% coverage |
|---|---|---|---|---|---|
| 4 | all | 0.137 | 1.60 | 100.0% | 93.8% |
| 4 | alternating | 0.081 | 1.63 | 96.9% | 93.8% |
| 16 | all | 0.162 | 1.52 | 93.0% | 92.2% |
| 16 | alternating | 0.251 | 1.68 | 96.9% | 96.1% |
| 64 | all | 0.130 | 1.18 | 90.8% | 88.3% |
| 64 | alternating | 0.244 | 1.28 | 90.8% | 87.7% |

The sparse integration executes through 64 RVs, but the remaining variance
inflation and missing-observation mean errors matter. Coverage alone would hide
some of these errors. Each larger model supplies more target coordinates per
optimizer update, so this is not evidence that larger problems are easier.

[Recorded results and checkpoints](../experiments/results/amortized/scaling.edn)
are exploratory. States within one trajectory are correlated, so pooled
coverage counts are not independent trials. The experiment does not establish
joint-distribution calibration or performance for arbitrary large simulators.
Sparse two-layer attention also needs further study for long-range dependencies.

```sh
clojure -M:gsdm -m org.replikativ.foerster.amortized-scaling /tmp/scaling.edn 30000
```
