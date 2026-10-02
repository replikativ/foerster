# Coming from Stan, PyMC, Turing, Gen, Anglican or WebPPL

A translation of concepts, and for each system what foerster does not have
yet. Throughout, `infer` is `org.replikativ.foerster.core`, `m`
`foerster.measure`, `diagnostics` `foerster.diagnostics`, and a model is a
function returning a spindel `spin` ([language](language.md)).

Three differences hold for all of them:

- **A model is a program run, not a graph.** Sites are named with `:id` and
  reached by ordinary control flow. There is no vectorized `~` over arrays:
  observations are sites in a `loop`/`recur` (`:id [:y i]`), and effects
  inside closures passed to `map` or `for` are not seen by the `spin` macro
  ([the rules](language.md#the-rules-of-the-spin-macro)).
- **Inference returns a spin resolving an empirical measure** — weighted
  particles, each with its value and trace ([posteriors](posteriors.md)).
  At the REPL deref it inside `sp/with-context`; inside a spin `await` it.
- **No gradients are derived from the model.** HMC runs on *block* sites
  whose log density and gradient you supply (or compile with
  [foerster-raster](https://github.com/replikativ/foerster-raster));
  everything else is gradient-free (random-walk and single-site MH, the
  particle methods, BBVI with score-function gradients).

## Stan

| Stan | foerster |
|---|---|
| `parameters { real mu; }` and `mu ~ normal(0, 1);` | `(sample (dist/normal 0.0 1.0) :id :mu)` |
| `real<lower=0> sigma;` with `sigma ~ normal(0, 1);` | `(sample (dist/half-normal 1.0) :id :sigma)`: the support comes from the distribution |
| `y ~ normal(mu, sigma);` (data) | `(observe (dist/normal mu sigma) y :id :y)` |
| `target += lp;` | `(factor lp)` |
| `transformed parameters`, `generated quantities` | ordinary `let` bindings; `(deterministic v :id :a)` records a value in the trace |
| `log_lik` in `generated quantities` | `(diagnostics/pointwise-log-likelihood measure)` |
| `y_rep` in `generated quantities` | `(infer/predictive model measure n)` |
| `sample(chains=4)` (NUTS) | `(infer/infer model {:method :nuts :iterations … :chains 4 :burn …})` on a block site, or `:rmh` on ordinary sites |
| R-hat, ESS, MCSE in the summary | `(diagnostics/summary measure :mu)` — the same rank-normalized split R-hat and bulk/tail ESS |
| bridge sampling (evidence) | `(m/log-marginal measure)` of an SMC or importance-sampling run |
| ADVI | `{:method :bbvi}`: mean-field, score-function gradients |

Not in foerster yet:

- Gradients derived from the model. NUTS (`{:method :nuts}`, with Stan's
  step-size and diagonal-metric adaptation) and HMC run on numerical blocks
  whose gradient is hand-written or compiled by raster; constrained
  latents (`:positive`, `[:interval a b]`) are transformed automatically
  when the block is written in natural coordinates, but there is no
  simplex or correlation-matrix transform yet.
- Optimization (`optimize`, MAP estimates) and Laplace approximations.
- Stan's speed on the models it is built for. A model with all observations
  at the end and smooth continuous latents fits in Stan in seconds; in
  foerster it takes a random walk or a hand-supplied gradient.

## PyMC

| PyMC | foerster |
|---|---|
| `with pm.Model():` | `(defn model [data] (spin …))` |
| `pm.Normal("mu", 0, 1)` | `(sample (dist/normal 0.0 1.0) :id :mu)` |
| `pm.Normal("y", mu, sigma, observed=y)` | `(observe (dist/normal mu sigma) y :id :y)`, one per datum |
| `pm.Potential("p", lp)` | `(factor lp)` |
| `pm.Deterministic("d", x)` | `(deterministic x :id :d)` |
| `pm.sample(chains=4)` | `(infer/infer model {:method :rmh :chains 4 …})` |
| `pm.sample_smc()` | `{:method :tempered}` — also tempered SMC; or `{:method :smc}` in program order |
| `pm.sample_prior_predictive()` | `(gfi/run-policy model (trace/policy {:simulate-observed? true}))` ([workflow](workflow.md#1-simulate-from-the-prior)) |
| `pm.sample_posterior_predictive(idata)` | `(infer/predictive model measure n)` |
| `az.summary`, `az.rhat`, `az.ess` | `diagnostics/summary`, `rhat`, `ess-bulk`, `ess-tail`, `mcse` |
| `pm.compute_log_likelihood`, `az.loo` | `diagnostics/pointwise-log-likelihood`, then ArviZ or R's `loo` |
| `pm.do(model, {"x": 1})` | `{:policy (trace/policy {:interventions {:x {:do 1}}})}` on a particle method |

Not in foerster yet: automatic gradients (NUTS runs on blocks with a
supplied or raster-compiled gradient), simplex and other matrix transforms, ADVI with reparameterization gradients, Gaussian-process and
other random-process building blocks, and vectorized distributions over
arrays. PSIS-LOO and WAIC are computed outside foerster.

## Turing

| Turing | foerster |
|---|---|
| `@model function f(y) … end` | `(defn f [y] (spin …))` |
| `mu ~ Normal(0, 1)` | `(sample (dist/normal 0.0 1.0) :id :mu)` |
| `y ~ Normal(mu, 1)` with `y` an argument | `(observe (dist/normal mu 1.0) y :id :y)` |
| `@addlogprob! lp` | `(factor lp)` |
| `model \| (; mu = 0.3)` | the particle methods' `:policy (trace/policy {:constraints {:mu 0.3}})`, or `gfi/generate` |
| `sample(m, MH(), n)` | `{:method :mh :iterations n}` |
| `sample(m, SMC(), n)`, `PG(n)`, `IS()` | `{:method :smc}`, `{:method :pgibbs}`, `{:method :importance}` |
| `Gibbs(:a => HMC(…), :b => PG(…))` | `k/block-gibbs-kernel`, or kernels composed by `k/cycle` / `k/mixture` ([algorithms](algorithms.md#kernels)) |
| `MCMCThreads(), 1000, 4` | `:chains 4` |
| `predict(m_missing, chain)` | `(infer/predictive model measure n)` |
| `generated_quantities` | `deterministic` sites, or `infer/query` with a function of the value |
| `pointwise_loglikelihoods` | `diagnostics/pointwise-log-likelihood` |
| MCMCChains summary | `diagnostics/summary` |

Not in foerster yet: automatic differentiation of the model (NUTS runs on
blocks), Bijectors beyond log and interval transforms, particle Gibbs as a component of a Gibbs sampler
(`k/cycle` and `k/mixture` compose Markov-chain kernels only: MH, block
Gibbs and HMC), and
MAP/MLE optimization. Conditioning from
outside (`|`) works for the particle methods and the generative function
interface; the Markov-chain methods take no `:policy`, so write the
observation into the model.

## Gen

foerster implements Gen's generative function interface over its traces
(`foerster.gfi`). Each operation is a CPS operation: `await` it in a spin.

| Gen | foerster |
|---|---|
| `@gen function f() … end` | `(defn f [] (spin …))` |
| `{:x} ~ normal(0, 1)` | `(sample (dist/normal 0.0 1.0) :id :x)` |
| `{:x => i => :y} ~ …` | `:id [:x i :y]`, or `with-scope` ([addresses](language.md#addresses)) |
| `{:sub} ~ g()` | `(await (g))` inside `with-scope [:sub]` |
| `choicemap((:x, 1.0))` | a map `{:x 1.0}` |
| `simulate(f, args)` | `(gfi/simulate (f))` |
| `generate(f, args, constraints)` | `(gfi/generate (f) constraints)` → `{:trace :weight}` |
| `assess(f, args, choices)` | `(gfi/assess (f) choices)` |
| `update(tr, args, argdiffs, constraints)` | `(gfi/update tr constraints)` → `{:trace :weight :discard}` |
| `regenerate(tr, args, argdiffs, sel)` | `(gfi/regenerate tr selection)` |
| `select(:x)` | `#{:x}`, or a spindel selector (`select/id`, `select/path`, `select/prefix`) |
| `mh(tr, sel)` | `(gfi/mh tr selection)` → `{:trace :accepted?}` |
| `mh(tr, proposal, args, involution)` | `(involutive/step tr {:propose :log-q :involution})` |
| `get_choices`, `get_retval`, `get_score` | `trace/choices`, `(:trace/result t)`, `trace/log-joint` |
| `importance_sampling`, `particle_filter_*` | `infer/importance-sampling`, `infer/smc-infer`, `smc/stream` |
| SMCP3 | `smc-infer` with `:anchors` and `:smcp3` |

Traces hold worlds: give them back with `gfi/close!`, and after `update` or
`regenerate` release the one you do not keep (`spindel.trace/release!`).

Not in foerster yet:

- **Arguments and argdiffs.** A model's arguments are closed over when its
  spin is built; `update` and `regenerate` cannot change them, and there
  are no argdiffs for incremental re-execution.
- **Combinators** (`Map`, `Unfold`, `Recurse`, `Switch`) and the static
  modeling language. Replay restarts from the earliest changed site.
- Moves that add or remove sites (reversible jump) are not supported.
  (`mh(tr, proposal, args)` with a proposal program is `gfi/mh-proposal`.)
- **Trainable parameters** (`@param`, `train!`) and amortized training.
  `foerster.learn` exports trajectories as training data; training happens
  elsewhere.
- `propose`, `project`, trace translators, and enumeration.

## Anglican

foerster keeps Anglican's design — `sample` and `observe` as checkpoints of
a continuation-passing program, inference as a handler — and most of its
algorithms.

| Anglican | foerster |
|---|---|
| `(defquery q [data] …)` | `(defn q [data] (spin …))` |
| `(defm f [x] …)` | `(defn f [x] (spin …))`, called with `(await (f x))` |
| `(sample (normal 0 1))` | `(sample (dist/normal 0.0 1.0) :id :mu)` — name sites you will read |
| `(observe (normal mu 1) y)` | `(observe (dist/normal mu 1.0) y)` |
| `(predict :mu mu)` | the program's return value, or `(deterministic mu :id :mu)` |
| `(store …)`, `(retrieve …)` | state in the particle's world (spindel atoms and signals fork with it) |
| `(doquery :smc q [data] :number-of-particles 1000)` | `(infer/infer (q data) {:method :smc :particles 1000})` |
| `:importance`, `:smc`, `:pimh`, `:pgibbs`, `:pgas`, `:ipmcmc`, `:bbvi` | the same keywords for `infer/infer` |
| `:lmh` | `{:method :mh}` |
| `:pcascade` | `(cascade/cascade model n opts)` |
| `(gamma shape rate)` | `(dist/gamma shape scale)`: **scale**, not rate |
| `with-primitive-procedures` | not needed: a model calls any Clojure function |

`doquery` returns a lazy, unbounded sequence of samples; `infer/infer`
resolves one finite measure.

Not in foerster yet:

- Random processes beyond the Chinese restaurant process (`process/mem`
  and `process/crp-draw` exist; Dirichlet and Gaussian processes are built
  on them by hand). Nested inference is `infer/conditional`.
- Higher-order functions with sites inside them. Anglican transformed
  `map` and `reduce` inside a query; in foerster an `observe` inside a
  closure passed to `map` is not seen by the macro — use `loop`/`recur`.
- `:almh` (adaptive site choice), and an anytime lazy stream of samples.

## WebPPL

| WebPPL | foerster |
|---|---|
| `var model = function() { … }` | `(defn model [] (spin …))` |
| `sample(Gaussian({mu: 0, sigma: 1}))` | `(sample (dist/normal 0.0 1.0) :id :mu)` |
| `observe(Gaussian({mu, sigma: 1}), y)` | `(observe (dist/normal mu 1.0) y)` |
| `factor(s)` | `(factor s)` |
| `condition(b)` | `(factor (if b 0.0 ##-Inf))` |
| `Infer({method: 'SMC', particles: 1000, model})` | `(infer/infer (model) {:method :smc :particles 1000})` |
| `Infer({method: 'MCMC', kernel: 'MH'})` | `{:method :mh}` |
| `Infer({method: 'forward'})` | `gfi/simulate`, or `{:method :importance}` on a model without data |
| `expectation(d, f)` | `(:mean (infer/query measure f))` |
| `d.normalizationConstant` | `(m/log-marginal measure)` |

WebPPL's sources include an asynchronous particle filter and particle MCMC,
prior art for foerster's cascade and particle-MCMC methods
([literature](literature.md)).

Not in foerster yet: exact enumeration (`'enumerate'`), rejection sampling,
incremental MH, `mem`, nested `Infer` (a posterior used as a distribution
inside another model), and variational inference with guide programs and
trainable parameters (`'optimize'`); foerster's BBVI learns a mean-field
guide only.

## What foerster has that these do not all have

- Particles that run in forks of a world with side effects
  (`:world-policy :fork`; [worlds](worlds.md)).
- Particle MCMC (PIMH, particle Gibbs, PGAS, IPMCMC), PMMH and SMC² over
  general programs, with evidence estimates from every SMC method.
- SMC without barriers: arrival-batched SMC and the particle cascade.
- Streaming SMC with a population that can be forked.
- Interventions and twin-world counterfactuals
  ([counterfactuals notebook](https://replikativ.github.io/foerster/foerster.counterfactuals.html)).
- Steering a process that has no density by SMC over scored steps, with the
  trajectories as training data.
- The same draws for a seed on the JVM and in JavaScript.
