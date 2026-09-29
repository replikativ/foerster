# foerster

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

- **SMC** and streaming SMC, importance sampling
- **particle MCMC**: PIMH, particle Gibbs, PGAS, IPMCMC
- **MCMC over traces**: single-site and random-walk Metropolis–Hastings,
  block Gibbs, involutive MCMC, **HMC** on numerical blocks
- **BBVI** (black-box variational inference)
- **counterfactuals** and interventions (Pearl's `do`), selectors over sites
- a Gen-style generative function interface

```clojure
(require '[org.replikativ.foerster.core :as infer]
         '[org.replikativ.foerster.effects :refer [sample observe]]
         '[org.replikativ.foerster.measure :as m]
         '[org.replikativ.spindel.spin.cps :refer [spin]]
         '[org.replikativ.spindel.effects.await :refer [await]]
         '[org.replikativ.foerster.dist :as dist])

(defn model []
  (spin
   (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
     (observe (dist/normal mu 1.0) 1.0 :id :y)
     mu)))

(spin
 (let [posterior (await (infer/smc-infer (model) 1000))]
   (infer/query posterior identity)))   ; mean ≈ 0.5
```

## Worlds

Pure models run in fresh worlds (`:world-policy :fresh`, the default). A model
that reads or changes the systems of the world it runs in (databases, repos,
a business book) runs in **canonical forks** of the caller's world
(`:world-policy :fork`):

- every particle is a frozen copy of the caller's world and runs the whole
  model, so nothing a particle writes reaches the caller;
- systems that must not be duplicated (live handles, unsettleable linear
  state) are refused before the model runs;
- with a resource authority, particles split the inference's budget instead of
  multiplying it;
- however inference ends, every world is discarded before the result is
  delivered, and the posterior keeps each particle's world descriptor.

See [docs/worlds.md](docs/worlds.md).

## Numerical blocks

A block is a group of latents sampled at one site, with a log density and its
gradient. HMC moves blocks jointly. [spindel-raster] compiles block densities
with raster (JVM, WASM, GPU) and reverse-mode AD.

## Platforms

The sources are `.cljc`, and inference runs on the JVM and in JavaScript.
Distributions (`foerster.dist`) are portable Clojure, named and
parameterized like raster's (and Distributions.jl), and a seed draws the same
numbers on both platforms. Copying worlds for canonical particles is
JVM-only; in ClojureScript canonical particles are forks.

## History

Monte Carlo began on ENIAC: Stanisław Ulam's idea (1946), John von Neumann's
design, Klára Dán von Neumann's code for the first runs (1948), and Nicholas
Metropolis's name for it. The Metropolis algorithm followed in 1953.

> The idea was to try out thousands of such possibilities and, at each stage,
> to select by chance, by means of a "random number" with suitable
> probability, the fate or kind of event, to follow it in a line, so to speak,
> instead of considering all branches. — Stanisław Ulam

foerster grew inside spindel and was split out with its history.

## License

Copyright © 2026 Christian Weilbach. Apache License 2.0.

[spindel-raster]: https://github.com/replikativ/spindel-raster
