# Documentation

For an overview see the [README](../README.md).

## Tutorials

Clay notebooks in [notebooks/foerster/](../notebooks/foerster/), rendered at
**[replikativ.github.io/foerster](https://replikativ.github.io/foerster/)**.
Each runs as it renders, in CI on every pull request.

| Notebook | Topics |
|---|---|
| [Getting Started](../notebooks/foerster/getting_started.clj) | a first model, `sample` and `observe`, SMC, the posterior and the evidence |
| [Choosing an Inference Algorithm](../notebooks/foerster/algorithms.clj) | importance sampling, SMC, particle MCMC, MCMC and BBVI on known posteriors; which to use when |
| [Models in Worlds](../notebooks/foerster/worlds.clj) | `:world-policy :fork`, systems in the world, budgets, what cannot be copied |
| [Blocks and HMC](../notebooks/foerster/blocks.clj) | numerical blocks, HMC-within-Gibbs, checking a gradient |
| [Streaming SMC](../notebooks/foerster/streaming.clj) | online filtering against the Kalman filter |
| [Programmable Inference](../notebooks/foerster/programmable.clj) | the generative function interface, MH by selection, involutive MCMC |
| [Interventions and Counterfactuals](../notebooks/foerster/counterfactuals.clj) | seeing and doing, interventions by selector, twin-world counterfactuals |
| [From the PyMC Gallery](../notebooks/foerster/pymc_gallery.clj) | golf putting (logistic and geometric models compared by LOO) and radon (partial pooling, NUTS on 89 parameters), each against an exact posterior |
| [Marketing Mix Modeling](../notebooks/foerster/mmm.clj) | adstock and saturation fit by NUTS on a block, ROAS by counterfactual and in the twin world, a lift test, budget splits by expected response and CVaR — against a known truth |
| [Steering a Process](../notebooks/foerster/steering.clj) | SMC over scored steps of a process without a density, value estimates as twists, trajectories as training data |

## Guides

| Guide | |
|---|---|
| [Language](language.md) | sites and their options, addresses, selectors, the rules of the spin macro |
| [Algorithms](algorithms.md) | every inference entry point and its options |
| [Posteriors](posteriors.md) | reading a measure: particles, traces, summaries, ESS, evidence, training data |
| [Checking a Model and its Inference](workflow.md) | prior predictive simulation, several methods, R-hat/ESS/MCSE and SMC diagnostics, posterior predictive checks, model comparison, simulation-based calibration |
| [Coming from Other PPLs](coming-from.md) | Stan, PyMC, Turing, Gen, Anglican and WebPPL concepts in foerster, and what foerster does not have yet |
| [Extending](extending.md) | your own distributions, policies, proposals, MH moves and kernels; learned twists and proposals |
| [Worlds](worlds.md) | canonical worlds, their lifecycle, failure and recovery, resources |
| [Distributions and Reproducibility](distributions.md) | `foerster.dist`, writing a distribution, seeds and streams |
| [Design](design.md) | inference as handlers of spindel savepoints; how each algorithm works |
| [Literature](literature.md) | the systems and papers foerster builds on |

## Reference

The API reference is on [cljdoc](https://cljdoc.org/d/org.replikativ/foerster),
generated from the docstrings.
