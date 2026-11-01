# Concepts and reading paths

Foerster runs probabilistic programs on Spindel's forkable execution runtime.
These five objects have different roles and lifetimes:

| Object | Meaning | Boundary |
|---|---|---|
| **World** | An execution context containing managed application state and registered systems. Canonical particle methods copy the caller's world; fresh-world methods start without it. | Forking isolates state supported by the runtime or a system adapter. Ordinary host atoms and arbitrary I/O are outside that boundary. |
| **Trace** | One run's addressed random choices, observations, scores and replay information. Trace operations can propose changes and re-execute a model. | Live replay resources have ownership and cleanup obligations. An arbitrary collection of values is not a replayable trace. |
| **Measure** | A settled inference result, with weighted values, summaries and method-specific diagnostics. | It is not a collection of live application worlds. Normalizer/evidence availability depends on the estimator. |
| **Population** | Intermediate inference state over a target, with weights and retained trace or numeric values. Population programs compose target changes, resampling and moves. | A move's invariance, a proposal correction and an evidence estimate are separate contracts. A common result interface does not certify all three. |
| **Numerical block** | A site exposing a joint numeric state and explicit density, gradient or other sampler capabilities. Raster can compile those supplied functions. | Foerster does not automatically extract a differentiable density from an arbitrary simulator. |

A typical particle run starts from a world, records traces, maintains a
population, then settles a measure and retires its owned worlds. Numerical
methods can instead operate on block values. Read the [algorithm guide](algorithms.md)
for each entry point's model and world-policy requirements.

## Choose a path

- **First probabilistic program:** [Getting Started](../notebooks/foerster/getting_started.clj),
  then [Choosing an Inference Algorithm](../notebooks/foerster/algorithms.clj)
  and [checking a model](workflow.md).
- **Reuse Clojure application code:** [Inference in a Running Application](../notebooks/foerster/queue.clj),
  then [Models in Worlds](../notebooks/foerster/worlds.clj) and the
  [language scope rules](language.md#the-rules-of-the-spin-macro).
- **Program inference:** [Inference Programs](inference-programs.org) for
  weighted populations; [Trace Programming and Custom Moves](../notebooks/foerster/programmable.clj)
  for the generative function interface and custom trace proposals.
- **Use numerical inference:** [Blocks and HMC](../notebooks/foerster/blocks.clj),
  then [From the PyMC Gallery](../notebooks/foerster/pymc_gallery.clj).
  [foerster-raster](https://github.com/replikativ/foerster-raster) supplies
  compiled block functions and automatic differentiation.

[Causal inference](causal.md), [amortized inference](amortized.md) and
[incremental trace inference](incremental-inference.org) each have additional
model assumptions and current restrictions. In particular, counterfactuals
need structural mechanisms and a declared noise coupling; an amortized
sampler is an approximation whose training loss alone does not validate its
posterior; fixed-target trace reuse does not implement a general reactive
posterior controller.
