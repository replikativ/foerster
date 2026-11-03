# Documentation

Start with the [README](../README.md) or choose a path in
[Concepts and Reading Paths](concepts.md).

## Tutorials

Clay notebooks in [notebooks/foerster/](../notebooks/foerster/), rendered at
**[replikativ.github.io/foerster](https://replikativ.github.io/foerster/)**.
Each runs as it renders, in CI on every pull request.

| Notebook | Topics |
|---|---|
| [Getting Started](../notebooks/foerster/getting_started.clj) | a first model, `sample` and `observe`, SMC, the posterior and the evidence |
| [Inference in a Running Application](../notebooks/foerster/queue.clj) | a live queue, speculative application state, SMC checked against enumeration |
| [Choosing an Inference Algorithm](../notebooks/foerster/algorithms.clj) | importance sampling, SMC, particle MCMC, MCMC and BBVI on known posteriors; which to use when |
| [Inference Methods](../notebooks/foerster/methods.clj) | every inference method foerster implements — what it does, when to use it, its options and cost, a run on a model with a known answer, and the papers it comes from |
| [Models in Worlds](../notebooks/foerster/worlds.clj) | `:world-policy :fork`, systems in the world, budgets, what cannot be copied |
| [Blocks and HMC](../notebooks/foerster/blocks.clj) | numerical blocks, HMC-within-Gibbs, checking a gradient |
| [Streaming SMC](../notebooks/foerster/streaming.clj) | online filtering against the Kalman filter |
| [Trace Programming and Custom Moves](../notebooks/foerster/programmable.clj) | the generative function interface, MH by selection, involutive MCMC |
| [Composing Inference](../notebooks/foerster/composing.clj) | proposals, kernels, pairs and inference programs plugged into each other, each against a known posterior |
| [Interventions and Counterfactuals](../notebooks/foerster/counterfactuals.clj) | seeing and doing, interventions by selector, twin-world counterfactuals |
| [From Data to Counterfactual Repairs](../notebooks/foerster/causal_workflow.clj) | native mechanism fitting, held-out checks, bootstrap effect/attribution uncertainty and incident repairs against known truth |
| [A Gallery of Classic Models](../notebooks/foerster/gallery.clj) | small models checked against exact answers, including branching and label switching |
| [From the PyMC Gallery](../notebooks/foerster/pymc_gallery.clj) | golf putting (logistic and geometric models compared by LOO) and radon (partial pooling, NUTS on 89 parameters), each against an exact posterior; stochastic volatility (a non-centred path by NUTS, checked by a particle filter) |
| [Marketing Mix Modeling](../notebooks/foerster/mmm.clj) | adstock and saturation fit by NUTS on a block, ROAS by counterfactual and in the twin world, a lift test, budget splits by expected response and CVaR — against a known truth |
| [Steering a Process](../notebooks/foerster/steering.clj) | SMC over scored steps of a process without a density, value estimates as twists, trajectories as training data |

## Guides

| Guide | |
|---|---|
| [Concepts and Reading Paths](concepts.md) | worlds, traces, measures, populations, blocks and their boundaries |
| [The Programming Model](programming-model.md) | proposals, kernels, pairs, population programs and contracts: where each plugs in, what is checked, what is not unified yet |
| [Incremental Trace Inference](incremental-inference.org) | complete keyed traces, finite Gibbs, resumable SCMC and the proposed reactive boundary |
| [Causal Queries and Identifiability](causal.md) | reusable abduction, nested worlds, ID/IDC, attribution, repairs and actual causality |
| [Amortized Inference](amortized.md) | automatic RV schemas and masks, conditional sampling artifacts, causal reconstruction and intervention inference |
| [Language](language.md) | sites and their options, addresses, selectors, the rules of the spin macro |
| [Algorithms](algorithms.md) | every inference entry point and its options; foerster in an interpreter sandbox |
| [Posteriors](posteriors.md) | reading a measure: particles, traces, summaries, ESS, evidence, training data |
| [Checking a Model and its Inference](workflow.md) | prior predictive simulation, several methods, R-hat/ESS/MCSE and SMC diagnostics, posterior predictive checks, model comparison, simulation-based calibration |
| [Coming from Other PPLs](coming-from.md) | Stan, PyMC, Turing, Gen, Anglican and WebPPL concepts in foerster, and what foerster does not have yet |
| [Inference Programs](inference-programs.org) | scoped custom population programs, targets, weight accounting and Markov transitions |
| [Extending](extending.md) | your own distributions, policies, proposals, MH moves and kernels; learned twists and proposals |
| [Worlds](worlds.md) | canonical worlds, their lifecycle, failure and recovery, resources |
| [Distributions and Reproducibility](distributions.md) | `foerster.dist`, writing a distribution, seeds and streams |
| [Design](design.md) | savepoint-based traces, numerical capabilities and the shared inference engines |
| [Literature](literature.md) | the systems and papers foerster builds on, by method |

## Reference

The API reference is on [cljdoc](https://cljdoc.org/d/org.replikativ/foerster),
generated from the docstrings.

- [Incremental trace inference](incremental-inference.org): keyed execution, finite Gibbs, ownership restrictions and the proposed reactive-data boundary.
