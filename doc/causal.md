# Causal queries and identifiability

Foerster evaluates causal queries in a model whose mechanisms the author
declares. Posterior inference, graphical identification, algebraic target
identifiability, and attribution are separate operations. An answer from one
is not evidence that a different question is identified.

| Operation | Entry point | Information required |
|---|---|---|
| Fit and check mechanisms | `causal-model/fit`, `causal-model/evaluate` | a declared causally sufficient DAG, complete numeric training and held-out rows |
| Association and posterior prediction | `infer/infer`, `infer/predictive` | a probabilistic model and evidence |
| Intervention | `trace/policy` `:interventions` | declared autonomous causal mechanisms |
| Counterfactual and nested worlds | `counterfactual/query` | a factual posterior plus a mechanism/noise coupling |
| Graphical effect identification | `causal/identify` | a declared acyclic directed mixed graph and an observed joint law |
| Incident attribution | `attribution/attribute-noise` | abducted incident noise, a baseline population and an outcome score |
| Mechanism-change attribution | `attribution/attribute-change` | old/new mechanisms and a mixed-model value function |
| Repair ranking | `attribution/repairs` | finite candidate interventions and a success predicate |
| Actual causality | `attribution/actual-cause` | a finite deterministic SCM at fixed exogenous noise, candidate events and domains |

The [amortized inference integration](amortized.md) can learn factual conditional
samplers from simulator source and reconstruct causal noise for supported models.
Its attention masks are inferred computational dependencies, not discovered causal
graphs. Learned posterior error remains separate from the correctness of replay.

## From observations to an executable causal model

The [causal workflow notebook](../notebooks/foerster/causal_workflow.clj)
fits a confounded synthetic system, checks held-out predictions, estimates
an intervention effect against known truth, reconstructs an incident, ranks
repairs and allocates excess latency. It also shows a nonlinear mechanism
that the linear model's residual checks reject.

```clojure
(require '[org.replikativ.foerster.causal-model :as cm])

(def fitted
  (cm/fit {:directed [[:load :queue] [:load :latency] [:queue :latency]]}
          training-rows))

(cm/evaluate fitted held-out-rows {:seed 42 :permutations 399})
(def executable (cm/model fitted))
(def incident (cm/abduct fitted {:load 1.0 :queue 3.0 :latency 200.0}))
;; `incident` can be passed to cf/query as :abductions [incident].
(cm/replay fitted (:noise incident) {:queue 0.0})
```

Rows are ordinary maps. The default fitter uses a linear conditional mean
and Gaussian noise with independent standardized residual coordinates.
Least squares uses centered/scaled, reorthogonalized QR; residual scale is
the maximum-likelihood estimate. Constant or collinear parents,
insufficient rows and effectively deterministic residuals are refused.
Declare deterministic equations in a program rather than hiding them behind
an arbitrary residual floor. This fitting entry point requires all nodes
observed and refuses bidirected edges. It does not fit latent-variable SCMs.
The general program/inference APIs still support author-declared latent models.

`cm/model` exposes one ordinary sample site per graph node, using the node
identifier as its address, and returns the node-value map. Existing trace
policies, posterior inference, counterfactual worlds and attribution work
with this model. `cm/simulate` generates IID observations under optional hard
interventions. `cm/replay` evaluates fixed noise synchronously; its intervention
map contains raw hard values, unlike trace policy maps containing `{:do …}`.

`cm/abduct` reconstructs noise from a **complete** observation at fixed fitted
parameters. For the built-in invertible mechanisms this is exact. It does
not infer a parameter posterior. For incomplete evidence use the executable
model with `cf/query` or ordinary inference. A custom discrete mechanism
samples conditional noise, so repeated abductions may differ.

Fitting is extensible without an external runtime:

```clojure
(cm/fit graph rows
  {:fitters
   {:latency
    (fn [node ordered-parents training-rows]
      ;; Fit parameters using these parents and training rows.
      {:family :custom
       :distribution (fn [parent-values] your-structural-distribution)})}})
```

The callback's distribution must implement `Distribution` and `PMechanism`.
Its runtime input contains only declared parent values. Fitters are
responsible for respecting that parent set and for a correct noise law and
conditional abduction. Diagnostics require scalar finite noise; a custom
`:noise-score` callback may project other noise to a scalar. Such a projection
can miss dependence in other coordinates. Custom families get log scores
and noise checks; Gaussian RMSE/coverage summaries apply to the built-in fit.

`cm/evaluate` must receive separate held-out IID rows. It reports log scores,
noise moments and Gaussian predictive coverage. Permutation tests examine
noise–parent and noise–noise dependence using the largest absolute
correlation across linear and squared features. Holm-adjusted p-values
account for multiple checks. These are limited moment-based tests, not
omnibus independence tests or full graph falsification. Small permutation
budgets limit the smallest attainable p-value. The report contains rejected
checks, never a flag declaring the causal model valid. Markov-equivalent
linear Gaussian graphs can pass while disagreeing on interventions.

`cm/bootstrap fitted original-training-rows statistic` resamples IID rows,
refits all mechanisms and returns a percentile interval for a scalar
statistic. The notebook shows both an intervention effect and a noise
contribution. Refitting errors are propagated, not silently discarded.
Deterministic fitting/statistic callbacks and `:seed` give reproducible draws.
Intervals condition on the graph, families, coupling and the reference
population used by the statistic. They do not establish coverage under
misspecification or address unobserved confounding. Dependent time-series
data require a different resampling scheme.

This follows the useful fit–evaluate–query separation of
[DoWhy's GCM workflow](https://www.pywhy.org/dowhy/main/user_guide/modeling_gcm/model_evaluation.html).
It does not yet provide DoWhy's general estimator/refutation collection,
kernel independence tests, graph falsification, propensity diagnostics or
automatic model-family selection.

## Abduct once, evaluate several worlds

The namespaces used below are:

```clojure
(require '[org.replikativ.foerster.counterfactual :as cf]
         '[org.replikativ.foerster.causal :as causal]
         '[org.replikativ.foerster.attribution :as attribution]
         '[org.replikativ.foerster.mechanism :as mechanism])
```

`cf/query` returns a CPS operation, as GFI does. Invoke it with success and
failure callbacks, or compose it with the GFI CPS operations. It does not
return an empirical measure synchronously.

```clojure
((cf/query model
   {:posterior posterior
    :worlds
    [[:untreated {:treatment {:do 0}}]
     [:direct (fn [worlds]
                {:treatment {:do 1}
                 :mediator {:do (get-in worlds [:untreated :mediator])}})]]})
 (fn [population]
   (println (cf/expectation population #(get-in % [:worlds :direct :outcome]))))
 (fn [error] (println error)))
```

The supplied model must expose the sites named in these interventions and
return the values used by the world specification. The example represents
a nested counterfactual; its validity depends on the declared structural
equations and their shared exogenous variables.

Supply exactly one factual source:

- `:posterior`: any ordinary empirical measure; its existing weights are
  retained. Already-posterior MCMC draws are not likelihood-weighted again.
- `:abductions`: previously reconstructed draws. `abduct-trace` accepts a
  live GFI trace; `abduct-posterior` accepts measures storing legacy traces.
- `:evidence`: a map of site values; `:particles` importance-weighted factual
  runs are generated, as in the earlier `counterfactual` helper.

Each reusable draw contains `:factual`, `:noise`, `:unsupported` and `:weight`
(a log weight). Each result contains `:worlds`, `:weight`, `:unaligned` and
`:status`. `:factual` is the first named world. Later specifications may
refer to earlier worlds. Names must be unique. `:event`, a predicate of the
world map, conditions on factual or hypothetical outcomes, including events
in several worlds. This is Monte Carlo reconditioning: rare events may need
many draws. Empty or zero-mass results are errors.

For compiled or block models, supply `:reconstruct`, a function from a
posterior particle to the reusable draw fields. Internal likelihood terms
hidden in a block are not automatically exposed as causal mechanisms. The
replay program must expose equivalent mechanisms or supply a structural
interpretation of the block. The MMM notebook's residual reconstruction is
an example of the required model-specific work.

## Coupling is part of a mechanism

`PMechanism` gives a noise draw, a forward transformation and posterior
noise reconstruction. Normal sites retain standardized residuals. Discrete
sites retain a threshold/CDF coordinate sampled conditional on their factual
value. Category order is part of that convention. Observational probabilities
do not select this convention uniquely.

`mechanism/mechanism` wraps any distribution with explicit `:noise`, `:push`
and `:abduct` functions. Their forward law must agree with the distribution;
`abduct` must sample conditional noise, not choose an arbitrary inverse.
`mechanism/inverse-cdf` supplies the quantile convention for distributions
with implemented CDFs and quantiles. For discrete laws it additionally needs
a function returning P(X < x). Declaring a CDF protocol alone does not imply
that a quantile implementation exists.

`cf/query`, `cf/predict` and `cf/worlds` require complete coupling by default.
An unsupported or newly reached stochastic site is refused. With
`:strict? false`, it draws fresh noise, including at observed sites, and is
reported in `:unaligned`; the result is `:partial`. Factual unsupported sites
that are removed by intervention need not prevent a fully coupled answer.
The earlier `cf/counterfactual` helper retains its permissive behavior for
compatibility; use `cf/query` for the strict interface.

Interventions support hard `:do` values, `:policy` values computed from
choices so far, distribution replacements `:dist`, and translations `:shift`.
Translations preserve the original noise coordinate. A replacement also
preserves it by default; `:coupling :fresh` explicitly introduces independent
noise instead. The author must choose compatible coordinates for replacements.
Named deterministic sites support `:do` and `:policy` interventions.

Model exogenous noise, posterior parameter uncertainty, numerical solver
randomness and inference RNG are different objects. Forking application
state does not declare a causal model. Preserving a log density through
compilation does not establish that joint counterfactual answers are preserved.

## Identification from a causal graph

```clojure
(def g
  (causal/graph
   {:directed [[:advertising :awareness] [:awareness :sales]]
    :bidirected [[:advertising :sales]]}))

(causal/identify g {:outcomes #{:sales} :treatments #{:advertising}})
```

This is the front-door structure. Directed edges declare causal relationships;
bidirected edges declare possible hidden common causes. `identify` implements
ID and conditional IDC, returning `:identified` and an expression in the
observational joint law, or `:not-identified` and the failing subgraphs as a
witness. `:given` specifies conditioned variables. Query sets must be disjoint.
Isolated nodes must be supplied in `:nodes`.

Expressions contain probability terms, products, ratios and marginalizations.
`causal/evaluate` evaluates them for finite domains, given an oracle accepting
event and conditioning maps. It refuses undefined ratios/positivity failures.
The symbolic ID/IDC procedure is not restricted to finite variables; the
provided numerical evaluator is. Causal discovery, measurement error,
transportability and identification of arbitrary nested counterfactuals are
outside this entry point. Source dependency analysis does not establish the
causal graph's assumptions.

References: Shpitser and Pearl, [ID (2006)](https://cdn.aaai.org/AAAI/2006/AAAI06-191.pdf)
and [conditional identification](https://arxiv.org/abs/1206.6876).

## Algebraic target identifiability is separate

Analysis of parameter fibers under a polynomial observation map is deferred.
It is not required by the native graphical ID/IDC implementation or by
counterfactual world composition. Both run in Clojure and ClojureScript
without an external algebra process.

If algebraic analysis is added later, it must distinguish an observed-marginal
map from a complete-data conditional-law map with unknown latent values.
Local versus global identification, finite ambiguity, denominator conditions
and unresolved resource limits must remain explicit. The inference/query
interfaces do not require a particular computer algebra system.

## Attribution, repair and actual causality

`attribution/shapley` evaluates an arbitrary coalition value function, with
exact or seeded permutation estimation and an evaluation budget.
`attribute-noise` constructs the coalition game from one abducted incident
and a supplied baseline noise bank: selected nodes keep incident noise,
others receive baseline noise. It requires fully coupled replay and computes
exact coalition attributions for the supplied finite bank. Average incident
reports with their posterior weights when abduction is uncertain.

`attribute-change` constructs a coalition game by swapping old mechanisms for
new ones. Its value callback can evaluate a simulator's mean or a chosen
distribution discrepancy. Mechanism fitting, anomaly-score calibration and
robust estimation from datasets are separate tasks.

`repairs` ranks explicitly supplied candidate interventions by posterior
success probability. Include joint repairs to cover interacting faults.
This does not establish that the winning repair is a minimal actual cause.

`actual-cause` checks the modified Halpern-Pearl conditions in a finite
deterministic SCM with fixed exogenous noise: factual occurrence, a
counterfactual witness with other nodes held at factual values, and exhaustive
subset minimality. Its `evaluate` callback maps interventions to all endogenous
values. The caller supplies candidate domains; witness sets can be restricted.
A resource limit returns `:unresolved`, not `:not-cause`.

The regression examples include preemption: the first throw breaks a bottle;
the second would have broken it in its absence. A single but-for test misses
the first throw's causal role. Holding the second hit at its factual value
supplies the witness. See [Halpern's modified definition](https://www.cs.cornell.edu/home/halpern/papers/modified-HPdef.pdf).

## Current boundary

These APIs cover causal simulation, shared-noise and nested counterfactuals,
graphical ID/IDC, coalition attribution,
finite repair ranking and finite actual-cause search. They are not a claim of
complete DoWhy parity. Native dataset fitting currently defaults to linear
Gaussian mechanisms, with custom fitting hooks. Broader dataset effect-estimator
families, refutation/sensitivity suites, sharp partial-identification bounds,
counterfactual identification from graphs and automatic block reconstruction
remain separate work. Their absence must not be hidden behind a posterior
estimate from a fully specified model.
