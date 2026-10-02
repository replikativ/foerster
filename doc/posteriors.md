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
