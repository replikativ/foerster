# Checking a Model and its Inference

An inference run returns numbers whether or not the model says what you
meant and whether or not the sampler found the posterior. This guide is a
short Bayesian workflow (after Gelman et al. 2020) in foerster's terms:
simulate from the prior, fit with more than one method, diagnose the run,
check the fit against the data, compare models, and calibrate the sampler.

The examples use this model, with five observations of a normal of unknown
mean and scale:

```clojure
(require '[org.replikativ.foerster.core :as infer]
         '[org.replikativ.foerster.diagnostics :as diagnostics]
         '[org.replikativ.foerster.measure :as m]
         '[org.replikativ.foerster.gfi :as gfi]
         '[org.replikativ.foerster.trace :as trace]
         '[org.replikativ.foerster.dist :as dist]
         '[org.replikativ.foerster.random :as random]
         '[org.replikativ.foerster.effects :refer [sample observe]]
         '[org.replikativ.spindel.core :as sp]
         '[org.replikativ.spindel.effects.await :refer [await]]
         '[org.replikativ.spindel.spin.cps :refer [spin]])

(def world (sp/create-execution-context))

(defn model [ys]
  (spin
   (let [mu (sample (dist/normal 0.0 1.0) :id :mu)
         sigma (sample (dist/half-normal 1.0) :id :sigma)]
     (loop [i 0]
       (when (< i (count ys))
         (observe (dist/normal mu sigma) (nth ys i) :id [:y i])
         (recur (inc i))))
     mu)))

(def ys [1.2 0.4 2.1 1.5 0.9])
```

Name the sites (`:id`): every step below reads them back by address.

## 1. Simulate from the prior

Before conditioning on data, check that the model generates data that look
plausible. Run it with every observed site drawing a fresh value from its
distribution instead of scoring the data — a policy option of
`foerster.trace`:

```clojure
(defn prior-draw []
  (spin
   (let [t (await (gfi/run-policy (model ys) (trace/policy {:simulate-observed? true})))
         c (trace/choices t)]          ; latents and simulated observations
     (await (gfi/close! t))
     c)))

(sp/with-context world @(prior-draw))
;; => {:mu -0.72, :sigma 1.34, [:y 0] 0.25, [:y 1] -3.49, …}
```

The data passed in are ignored under `:simulate-observed?`; only their
count matters here. `gfi/simulate` runs the model with the prior for the
latents but scores the observations, so it gives prior draws of the
parameters, not of the data. Look at the simulated data in the units of
the problem: if the prior puts much of its mass on data you know to be
impossible, change the prior before fitting.

## 2. Fit with more than one method

`infer/infer` runs any method with one call shape
([algorithms](algorithms.md#one-call-shape-and-which-method)), so fitting
the same model twice costs a line:

```clojure
(def smc (sp/with-context world
           @(infer/infer (model ys) {:method :smc :particles 1000})))

(def chains (sp/with-context world
              @(infer/infer (model ys) {:method :rmh :iterations 2000 :chains 4
                                        :burn 500 :step-size 0.3})))
```

Two methods that rest on different assumptions — particles in program
order, a random walk over complete traces — and agree on the posterior
summaries are much stronger evidence than either alone. When they
disagree, at least one has not found the posterior; the diagnostics below
say which.

## 3. Diagnose the run

### Markov chains

Run several chains (`:chains 4` is the default for `:mh` and `:rmh`) and
read the summary of each quantity you care about:

```clojure
(diagnostics/summary chains :mu)
;; => {:mean 1.06 :sd 0.31 :quantiles {…} :rhat 1.011 :ess-bulk 315.0
;;     :ess-tail 528.0 :mcse 0.017 :chains 4 :draws 6000}
```

These are the diagnostics of Vehtari et al. (2021), computed as Stan and
ArviZ compute them: rank-normalized split R-hat, bulk and tail effective
sample sizes, and the Monte Carlo standard error of the mean.

- **R-hat** above 1.01: the chains have not mixed. The run above sits just
  over the line; run longer, start elsewhere, or change the kernel (the
  step size of `:rmh`, HMC on a block, `k/cycle` of kernels).
- **ESS** (bulk and tail) below about 100 per chain: too few effectively
  independent draws for the quantiles to be reliable.
- **MCSE**: the standard error of the reported mean. Report the mean only to
  the digits the MCSE supports.

Check every quantity you report — `summary` takes a site address, a key of
a map value, or a function of the value — not only one. `diagnostics/rhat`,
`ess-bulk`, `ess-tail` and `mcse` take draws by chain directly, and
`(diagnostics/chains chains :mu)` gives those draws for trace plots.

### Particle methods

SMC's weights carry the information, and its diagnostics are different:

- **Evidence** — `(m/log-marginal smc)`. Rerun with several seeds
  (`random/set-seed!`); a log-evidence that varies by more than about 1
  across runs says the population is too small for the model.
- **The `:history`** — one map per observation barrier with `:ess` before
  resampling, `:resampled?` and `:log-mean-weight`. A barrier whose ESS
  collapses to a handful of particles is where the proposal (the prior, by
  default) and the data disagree; a site `:proposal` or more particles help
  there.
- **Surviving histories** — after resampling the weights are equal again,
  so the weight-based ESS counts particles, many of which are copies.
  `(diagnostics/distinct-count smc :mu)` counts the distinct values an early
  site still takes: the number of histories the population descends from.
  A static parameter with a few distinct values left is degenerate; use
  resample-move (`:anchors`, `:rejuvenate`), `:pmmh` or SMC²
  ([algorithms](algorithms.md)).

`(diagnostics/summary smc :mu)` gives the weighted mean, sd and quantiles
with `:ess`, the weight-based effective sample size. The pooled
particle-MCMC methods (`:pimh`, `:pgibbs`, `:pgas`, `:ipmcmc`) are MCMC
estimates; their `m/log-marginal` is not an evidence estimate.

### Against an exact answer

When a version of the model has a closed-form posterior — conjugate priors,
a discrete model small enough to sum by hand — fit that version and compare.
For a N(0, 1) prior on μ and one observation y = 1 from N(μ, 1), the
posterior is N(0.5, 0.707²) and the evidence is the density of y under
N(0, √2):

```clojure
(defn toy []
  (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
          (observe (dist/normal mu 1.0) 1.0 :id :y)
          mu)))

(let [s (sp/with-context world @(infer/infer (toy) {:method :smc :particles 2000}))]
  [(:mean (infer/query s identity))                     ; ≈ 0.5
   (m/log-marginal s)                                   ; ≈ -1.52
   (dist/logpdf (dist/normal 0.0 (Math/sqrt 2.0)) 1.0)]) ; -1.5155
```

`foerster.conjugate` has the conjugate families
([algorithms](algorithms.md#conjugate-parameters-never-sampled)); a model
that carries its parameter's posterior there is also an exact reference for
the same model sampled.

## 4. Check the fit: posterior predictive draws

A posterior that is computed correctly can still be a poor description of
the data. `infer/predictive` replays posterior draws with their latent
choices held and every observed site drawing a fresh value:

```clojure
(def pp (sp/with-context world @(infer/predictive (model ys) smc 1000)))
(first pp)
;; => {:value 0.77 :observations {[:y 0] 1.04, [:y 1] 2.33, …}}
```

Compare a statistic of the replicated data with the same statistic of the
observed data — the minimum, the maximum, the spread, a count of outliers:

```clojure
(let [stat (fn [obs] (apply max (vals obs)))
      observed (apply max ys)]
  (/ (count (filter #(>= (stat (:observations %)) observed) pp)) (count pp)))
;; the posterior predictive p-value of the maximum
```

A value near 0 or 1 says the model cannot reproduce that feature of the
data. Choose statistics the model does not fit directly (a normal model
always reproduces the mean).

## 5. Compare models

Two quantities, which answer different questions:

- **The evidence** p(data | model), from importance sampling and the SMC
  methods (`m/log-marginal`). The difference of two models' log evidences
  is their log Bayes factor. It depends on the whole prior, so a vague prior
  on a parameter the data barely constrain is penalized; it says which model
  explains the data better as a whole, prior included.
- **Expected predictive accuracy**, estimated by leave-one-out
  cross-validation (PSIS-LOO) or WAIC (Vehtari, Gelman & Gabry 2017). These
  need the log density of each observation under each posterior draw:

  ```clojure
  (diagnostics/pointwise-log-likelihood chains)
  ;; => [{[:y 0] -0.74, [:y 1] -1.48, …} …]   one map per draw
  ```

  The draws must be equally weighted, as Markov chains give. For a weighted
  measure, resample first:

  ```clojure
  (diagnostics/pointwise-log-likelihood
   (m/empirical (mapv (fn [[p _]] [p 0.0]) (m/sample-measure smc 1000))))
  ```

  `loo` and `waic` compute the estimates from them as ArviZ's `az.loo`
  and `az.waic` do, and `compare` ranks models over the same observations:

  ```clojure
  (diagnostics/loo chains)
  ;; => {:elpd-loo -7.87 :se 0.73 :p-loo 0.55
  ;;     :pareto-k {[:y 0] 0.01 …} :good-k 0.7 :pointwise {[:y 0] -1.1 …}}

  (diagnostics/compare {:free free-chains :pinned pinned-chains})
  ;; => [{:name :free   :elpd-loo -7.87  :elpd-diff 0.0  :dse 0.0 …}
  ;;     {:name :pinned :elpd-loo -30.54 :elpd-diff 22.66 :dse 5.04 …}]
  ```

  A higher `:elpd-loo` predicts held-out data better; a difference of a
  few `:dse` is clear. An observation whose Pareto k̂ exceeds `:good-k` has
  an unreliable estimate: the posterior moves a lot without it, so refit
  without it or use a more robust model. `:p-loo` is the effective number
  of parameters; far above the actual number says the model is
  misspecified. `:reff` (default 1) is the relative efficiency ESS/S the
  Pareto smoothing assumes, ArviZ's estimate from chains.

## 6. Calibrate the sampler

The checks above find a run that went wrong. Simulation-based calibration
(Talts et al. 2018) checks the sampler itself: draw θ* from the prior,
simulate data from the model at θ*, fit, and record the rank of θ* among
the posterior draws. If the sampler is correct, the ranks are uniform; a
skewed or U-shaped histogram shows bias or under- and over-dispersion.

`experiments/org/replikativ/foerster/sbc.clj` does this for HMC on block
sites, with two models: a two-dimensional Gaussian mean and a two-parameter
logistic regression. Each simulation draws θ* from the prior, simulates the
data, runs one HMC chain (100 burn-in moves, then 99 draws thinned by 5),
and ranks θ* in each dimension among the draws. The ranks (0 … 99) go into
10 bins per dimension, tested for uniformity by a χ² test at level 0.01,
Bonferroni-corrected over the dimensions tested. It is a seeded experiment,
not part of the test suite:

```bash
clojure -M:test:sbc 200            # 200 simulations; exits 1 if a test fails
clojure -M:test:sbc 200 control    # negative control: a block assuming the
                                   # wrong noise, which must fail
```

The negative control matters: it shows that the test can fail.

For your own model, the same loop is short to write: simulate a dataset as
in step 1 (`:simulate-observed? true`, reading θ* and the data from
`trace/choices`), fit `(model data)`, and rank θ* among draws that are
close to independent — thin a chain by its effective sample size, or use
`infer/predict` on a particle measure. A few hundred simulations detect
gross errors; subtle ones need more.

## Further reading

- A. Gelman et al. *Bayesian Workflow.* 2020.
  [arXiv:2011.01808](https://arxiv.org/abs/2011.01808)
- A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, P.-C. Bürkner.
  *Rank-normalization, folding, and localization: an improved R̂ for
  assessing convergence of MCMC.* Bayesian Analysis 2021.
- A. Vehtari, A. Gelman, J. Gabry. *Practical Bayesian model evaluation
  using leave-one-out cross-validation and WAIC.* Statistics and Computing
  2017.
- S. Talts et al. *Validating Bayesian Inference Algorithms with
  Simulation-Based Calibration.* 2018.

See [posteriors](posteriors.md) for reading a measure and
[literature](literature.md) for the rest.
