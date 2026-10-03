# Posteriors

Every inference method resolves an **empirical measure**: weighted samples
of the program. `org.replikativ.foerster.measure` (`m`) reads it.

## Particles

```clojure
(m/get-particles measure)   ; [[particle log-weight] …]
(m/get-contexts measure)    ; the particles alone
(m/get-log-weights measure) ; the log weights alone
```

A particle is a `Sample` for the particle methods (and for the Markov
chains' `:samples :all`), or a projected context (a chain's final state):

```clojure
(m/get-value particle)        ; the program's value
(m/get-trace particle)        ; {address {:value :distribution :log-prob :observed?}}
(m/site-value particle :bmi)  ; the value of one site, a `deterministic` one included
```

Weights are log-scale and unnormalized; `(m/normalize-log-weights lws)`
turns them into probabilities. A canonical-world particle also carries its
world's descriptor (`:world-descriptor`; `(m/world-descriptors measure)`
collects them).

## Summaries

```clojure
(infer/query measure identity)   ; {:mean :variance :std-dev … :samples :weights}
(infer/query measure :mu)        ; a field of a map value
(infer/query measure #(* 2 %))   ; a function of the value
(infer/predict measure m/get-value 1000)   ; 1000 resampled draws
```

For a site's posterior rather than the value's,
`(m/measure-stats measure #(m/site-value % :mu))` gives the same statistics.

`query` needs a numeric function of the value; its `:samples` and
`:weights` are the per-particle values and normalized weights, which any
plotting or statistics library can use; its `:quantiles` (`:p025`, `:p50`,
`:p975`) are weighted too (`m/weighted-quantiles` for others). `predict`
resamples by weight and passes each drawn particle to its function.

## Diagnostics

**Markov chains** — `kernel-infer` keeps each chain's draws together (the
measure's `:chain-lengths`), and `org.replikativ.foerster.diagnostics`
computes the convergence diagnostics of Vehtari et al. (2021) exactly as
Stan and ArviZ do (checked against ArviZ to 10⁻⁹): rank-normalized split
R-hat, bulk and tail effective sample sizes, and the Monte Carlo standard
error of the mean.

```clojure
(diagnostics/summary measure :mu)   ; a site, a map key, or a function of the value
;; => {:mean 0.71 :sd 0.33 :quantiles {…} :rhat 1.002 :ess-bulk 1830.0
;;     :ess-tail 1590.0 :mcse 0.008 :chains 4 :draws 4800}
```

Run several chains (`:chains 4` with `infer/infer`); R-hat above 1.01 or a
bulk or tail ESS below about 100 per chain means the chains have not
mixed: run longer, start elsewhere, or change the kernel. `chains` gives the
draws by chain for plots.

**Effective sample size** — `(m/effective-sample-size measure)` =
(Σw)² / Σw² over the normalized weights: how many equally weighted samples
the weighted ones are worth. It is meaningful for importance sampling and
before resampling. After SMC resamples, the weights are equal again and it
counts particles, though many may be copies of a few; for equally weighted
MCMC output it counts samples, not their autocorrelation.

**History** — an SMC measure's `:history` has one map per observation
barrier: `:ess` before resampling, `:resampled?`, `:log-mean-weight` (that
barrier's factor of the evidence) and, with `:genealogy? true`, the
`:ancestors` each particle was resampled from — enough to see where the
population degenerated and to trace lineages.

**Evidence** — `(m/log-marginal measure)` estimates log p(data), the
normalizer of the posterior, for importance sampling and the SMC methods
(tempered and batched SMC, the cascade, SMC²): compare models by it. For
pooled particle-MCMC output it is not an evidence estimate.

**Surviving histories** — `(diagnostics/distinct-count measure [:x 0])`:
how many distinct values an early site still takes after SMC resampled —
the number of histories the particles descend from, which the weight-based
ESS does not show.

## Checking the model

**Predictive draws** — `(infer/predictive model measure n)` replays `n`
posterior draws (particles drawn by weight) with their latent choices held
and every observed site drawing a fresh value from its distribution
instead of scoring the data; it resolves `[{:value v :observations {address
x}} …]`. Compare the drawn observations with the data (a posterior
predictive check). For a prior predictive check, run the model with
`(gfi/run-policy model (trace/policy {:simulate-observed? true}))`:
`gfi/simulate` scores the data instead of drawing it
([workflow](workflow.md#1-simulate-from-the-prior)). `model` may be the program on new inputs — its latent
sites are held at the posterior draw and observed sites draw fresh, so it
forecasts — and `(infer/predictive model measure n {:interventions {:b
0.0}})` sets sites by the do-operator first: a scenario under the
posterior.

**Pointwise log-likelihood** — `(diagnostics/pointwise-log-likelihood
measure)`: one map `{address log-p}` of the observations per draw, the
input of PSIS-LOO and WAIC (equally weighted draws, as Markov chains give;
resample a weighted measure first).

## Training data

The particles' traces are also training data. For a steered program
(`foerster.steer/model`), which records its states under `[:steer/state t]`
and its reward under `:steer/reward`, `org.replikativ.foerster.learn` reads
them back:

```clojure
(learn/trajectory particle)   ; {:states [s₀ s₁ …] :reward r}
(learn/trajectories measure)  ; every particle's, with :weight and :log-weight
(learn/draws measure 256)     ; 256 resampled by weight: unweighted draws from the target
```

`draws` is what a model trained on plain examples needs; `trajectories`
keeps the weights for weighted losses (see
[extending](extending.md#learned-twists-and-proposals)).

## Fitting parameters

A model's parameters are its arguments. `learn/maximize-evidence` fits them
by maximum marginal likelihood (empirical Bayes): Adam on the evidence
estimate of a particle method, the gradient by central differences whose
two sides share a seed (common random numbers), so it is not lost in
Monte Carlo noise.

```clojure
(learn/maximize-evidence (fn [{:keys [log-sigma]}] (model (Math/exp log-sigma)))
                         {:log-sigma 0.0}
                         {:method :smc :particles 300}
                         {:steps 120 :rate 0.05})
;; => {:params {:log-sigma …} :history [{:params … :log-evidence …} …]}
```

Each step costs 2·d + 1 inferences for d parameters, so it suits a few
hyperparameters; a guide network or many parameters train with gradients
in finetune-rstr and raster. Particle methods take `:seed` for such
reproducible runs.
