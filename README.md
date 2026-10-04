# foerster

[![Clojars Project](https://img.shields.io/clojars/v/org.replikativ/foerster.svg)](https://clojars.org/org.replikativ/foerster)
[![CircleCI](https://circleci.com/gh/replikativ/foerster.svg?style=shield)](https://circleci.com/gh/replikativ/foerster)
[![cljdoc](https://cljdoc.org/badge/org.replikativ/foerster)](https://cljdoc.org/d/org.replikativ/foerster)
[![Slack](https://img.shields.io/badge/slack-join_chat-brightgreen.svg)](https://clojurians.slack.com/archives/CB7GJAN0L)

> A "bit" of information is definable as a difference that makes a
> difference. — Gregory Bateson, *Steps to an Ecology of Mind*

> Act always so as to increase the number of choices. — Heinz von Foerster

Probabilistic inference in forkable worlds, for Clojure.

A *Förster* is a forester: someone who tends a forest, grows branches, prunes
them, and decides which trees continue. That is what inference over
[spindel](https://github.com/replikativ/spindel) worlds does. Particles branch
from a world, resampling prunes the unlikely ones and copies the likely ones,
and Markov chains walk the tree of worlds. The name also honours
Heinz von Foerster, whose cybernetics put the observer inside the system it
observes, as agents inferring inside the worlds they act in are here.

## What it does

A probabilistic program is a spindel `spin` with `sample`, `observe` and
`factor` sites. Every site is a spindel *savepoint*, so an inference algorithm
is a handler that decides, scores, forks, copies or abandons those savepoints:

- **one call shape** for every method: `(infer/infer model {:method :smc
  :particles 1000})`
- **exact enumeration** of finite discrete models over forked worlds
- **SMC** and streaming SMC, importance sampling, tempered SMC,
  resample-move and SMCP3; arrival-batched SMC and the **particle cascade**,
  which do not wait for the slowest particle
- **particle MCMC**: PIMH, particle Gibbs, PGAS, IPMCMC; **PMMH** and
  **SMC²** for static parameters
- **MCMC over traces**: single-site and random-walk Metropolis–Hastings,
  block Gibbs, involutive MCMC with reversible jump, custom proposal
  programs; **NUTS** (Stan's adaptation) and HMC on numerical blocks, with
  constrained latents; kernels composed by `k/cycle` and `k/mixture`
- **MAP** (L-BFGS) and the **Laplace approximation** on blocks; **BBVI**
  (black-box variational inference); evidence maximization for a model's
  parameters (`foerster.learn`)
- **Anglican's toolkit**: `mem`, Chinese restaurant processes, nested
  inference (`infer/conditional`)
- **checking**: R-hat, bulk and tail ESS and MCSE as ArviZ computes them,
  PSIS-LOO, WAIC and model comparison, posterior predictive checks
  (also on new inputs and under interventions), export to ArviZ
- **steering** a process — a language model's turns, a simulator — by SMC
  over scored steps, twisted by a value estimate (`foerster.steer`), with
  the trajectories as training data (`foerster.learn`)
- **guides**: a sample site's `:proposal` draws from a distribution the
  program computes, weighted by log p − log q
- **counterfactuals** and interventions (Pearl's `do`), selectors over sites
- a Gen-style generative function interface

```clojure
(require '[org.replikativ.foerster.core :as infer]
         '[org.replikativ.foerster.dist :as dist]
         '[org.replikativ.foerster.effects :refer [sample observe]]
         '[org.replikativ.spindel.core :as sp]
         '[org.replikativ.spindel.spin.cps :refer [spin]])

;; programs run in a spindel world (an execution context)
(def world (sp/create-execution-context))

(defn model []
  (spin
   (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
     (observe (dist/normal mu 1.0) 1.0 :id :y)
     mu)))

;; inference returns a spin; at the REPL, deref it
(def posterior (sp/with-context world @(infer/infer (model) {:method :smc :particles 1000})))

(:mean (infer/query posterior identity))   ; ≈ 0.5, the posterior is N(0.5, 0.707²)
```

Inside another spin, `await` it instead of dereferencing
(`org.replikativ.spindel.effects.await`).

## Documentation

- **Tutorials** — runnable notebooks, rendered at
  [replikativ.github.io/foerster](https://replikativ.github.io/foerster/):
  getting started, choosing an algorithm, models in worlds, blocks and HMC,
  streaming SMC, programmable inference, interventions and counterfactuals,
  steering a process, a gallery of classic models, models from the PyMC
  gallery, and marketing mix modeling — each checked against an exact
  answer or a known truth.
- **Guides** — [doc/](doc/README.md): the language, the algorithms,
  posteriors, [checking a model and its inference](doc/workflow.md),
  [coming from Stan, PyMC, Turing, Gen, Anglican or WebPPL](doc/coming-from.md),
  extending, worlds, distributions and reproducibility, the design, the
  literature.
- **API reference** — [cljdoc](https://cljdoc.org/d/org.replikativ/foerster).

## Worlds

Pure models run in fresh worlds (`:world-policy :fresh`, the default). The
particle methods (importance sampling, SMC, particle MCMC, BBVI) also run in
**canonical forks** of the caller's world (`:world-policy :fork`), for models
that read or change the systems of the world they run in (databases,
repositories, a business book); Markov chains run in fresh worlds only:

- every particle is a frozen copy of the caller's world and runs the whole
  model, so nothing a particle writes reaches the caller;
- on the JVM particles are *copies* of worlds: systems that must not be
  duplicated (live handles, unsettleable linear state) are refused before the
  model runs, and with a resource authority particles split the inference's
  budget instead of multiplying it (in ClojureScript particles are forks,
  without these checks);
- however inference ends, every world is discarded before the result is
  delivered, and the posterior keeps each particle's world descriptor.

See [the worlds guide](doc/worlds.md).

## Numerical blocks

A block is a group of latents sampled at one site, with a log density and its
gradient. HMC moves blocks jointly. [foerster-raster] compiles block densities
with raster (JVM, WASM, GPU) and reverse-mode AD.

## Platforms

The sources are `.cljc`, and inference runs on the JVM and in JavaScript.
Distributions (`foerster.dist`) are portable Clojure, named and
parameterized like raster's, and the random generator draws the same
numbers for a seed on both platforms. Copying worlds for canonical particles is
JVM-only; in ClojureScript canonical particles are forks.

## History

Monte Carlo began on ENIAC: Stanisław Ulam's idea (1946), John von Neumann's
design, Klára Dán von Neumann's code for the first runs (1948), and Nicholas
Metropolis's name for it. The Metropolis algorithm followed in 1953.

> The idea was to try out thousands of such possibilities and, at each stage,
> to select by chance, by means of a "random number" with suitable
> probability, the fate or kind of event, to follow it in a line, so to speak,
> instead of considering all branches. — Stanisław Ulam

## Lineage

foerster stands on earlier probabilistic programming systems:

- [Anglican](https://github.com/probprog/anglican), probabilistic
  programming in Clojure: a model is a program whose `sample` and
  `observe` are checkpoints, and inference is a continuation-passing
  interpreter over them (SMC, PIMH, PGibbs, PGAS, IPMCMC, LMH, BBVI).
  foerster's sites are those checkpoints as spindel savepoints, and its
  benchmarks reuse Anglican's ground truths.
- [Daphne](https://github.com/plai-group/daphne), a probabilistic
  programming compiler in Clojure, following
  [An Introduction to Probabilistic Programming](https://arxiv.org/abs/1809.10756):
  first-order programs compiled to graphical models, and those to amortized
  inference networks.
- [Gen](https://www.gen.dev/) ([Gen.jl](https://github.com/probcomp/Gen.jl)),
  programmable inference through the generative function interface —
  `simulate`, `generate`, `assess`, `update`, `regenerate` over traces — which
  foerster implements (`foerster.gfi`), and involutive MCMC.

foerster grew inside spindel and was split out with its history.

## License

Copyright © 2026 Christian Weilbach. Apache License 2.0.

[foerster-raster]: https://github.com/replikativ/foerster-raster
