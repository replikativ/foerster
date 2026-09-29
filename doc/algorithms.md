# Algorithms

A reference for every inference entry point in `org.replikativ.foerster.core`
(`infer` below). Each returns a spin resolving an `EmpiricalMeasure` (see
[posteriors](posteriors.md)); at the REPL deref it inside
`sp/with-context`, inside a spin `await` it. For which to use when, see the
[algorithms notebook](https://replikativ.github.io/foerster/foerster.algorithms.html).

## Options every particle method takes

`smc-infer`, `importance-sampling`, `pimh-infer`, `pgibbs-infer`,
`pgas-infer`, `ipmcmc-infer`, `bbvi-infer`, and `kernel-infer` with a
non-Markov-chain kernel:

| Option | Default | Meaning |
|---|---|---|
| `:world-policy` | `:fresh` | `:fork` runs the particles in canonical forks of the caller's world ([worlds](worlds.md)) |
| `:world-opts` | `{}` | fork options for `:fork`: `:systems` (`:all`, `:none` or a set of system ids), `:rights`, `:snapshots` |
| `:authority`, `:grant` | — | budgets under `:fork` ([worlds](worlds.md#budgets)) |
| `:executor` | spindel's default | the executor the worlds run on |
| `:resample-threshold` | `0.5` | SMC resamples when the effective sample size is below this fraction of the particles |
| `:policy` | the prior | a `foerster.trace/policy` deciding the sites: constraints, interventions, custom proposals ([extending](extending.md)) |

An option a method does not take is refused (`::infer/unknown-options`), and
so are `:world-opts`, `:authority` and `:grant` under `:world-policy :fresh`
(`::infer/fork-only-options`): an option that would be ignored is a mistake.
`importance-sampling` takes no `:resample-threshold`, BBVI no `:policy` or
`:resample-threshold` (it decides both), and the Markov-chain kernels only
`:executor` and `:world-policy :fresh`.

## Particle methods

| Function | What it does |
|---|---|
| `(smc-infer model n opts)` | Sequential Monte Carlo with `n` particles, resampling at observations. The measure's `m/log-marginal` estimates the evidence. |
| `(importance-sampling model n opts)` | `n` weighted runs, never resampled. |
| `(pimh-infer model n iterations opts)` | Particle independent MH: `iterations` SMC sweeps of `n` particles, each accepted on its evidence estimate; every sweep's particles pooled. |
| `(pgibbs-infer model n iterations opts)` | Particle Gibbs: iterated conditional SMC, each sweep keeping a trajectory drawn from the previous one; sweeps pooled. |
| `(pgas-infer model n iterations opts)` | PGAS: particle Gibbs whose retained particle redraws its ancestor at every observation. Costs a replay of the retained future per particle per observation. |
| `(ipmcmc-infer model n iterations opts)` | Interacting particle MCMC: `:num-nodes` (default 8) SMC nodes, `:num-csmc-nodes` (default half) of them conditional, exchanged by Gibbs updates on their evidence; `:all-particles?` (default `true`) emits every node's particles, `false` one per node. |
| `(bbvi-infer model n iterations opts)` | Black-box variational inference: learns a mean-field q in `iterations` updates from `n` samples each. Each site's q starts as its prior when first reached, then moves by gradient only (it does not follow a prior that depends on other latents). `:base-lr` (1.0), `:robbins-monro` (0.0; step size `base-lr / (t+1)^robbins-monro`), `:adagrad` (`true`: AdaGrad; γ ∈ (0,1): RMSprop; `false`: plain steps). The measure holds `n` importance samples from the final q; `(infer/get-variational-dists measure)` returns q, `{address distribution}`. |

The pooled particle-MCMC measures are MCMC estimates: their weights are
normalized per sweep, and `m/log-marginal` of the pool is not an evidence
estimate.

## Kernels

`(kernel-infer model kernel n opts)` runs a kernel from
`org.replikativ.foerster.kernel` (`k`):

| Kernel | Runs |
|---|---|
| `(k/prior-kernel)` | SMC from the prior (`:barrier-policy :none`: importance sampling) |
| a custom `PInferenceKernel` | SMC whose latent sites take the kernel's `step` value ([extending](extending.md)) |
| `(k/single-site-mh-kernel iterations opts)` | `n` independent Markov chains of single-site MH, proposals from the prior |
| `(k/random-walk-mh-kernel iterations {:step-size 0.1})` | random-walk MH on continuous sites |
| `(k/block-gibbs-kernel iterations selector block-kernels classifier opts)` | block Gibbs: `classifier` `(fn [address entry])` assigns sites to block ids, `selector` (`k/round-robin-selector`, `k/random-selector`) picks a block per iteration, `block-kernels` `{block-id kernel}` moves it (`k/prior-block-kernel`, `k/random-walk-block-kernel`) |
| `(k/hmc-kernel iterations {:step-size 0.1 :steps 10})` | HMC-within-Gibbs on block sites ([blocks](https://replikativ.github.io/foerster/foerster.blocks.html)) |

The Markov-chain kernels run in fresh worlds only (they refuse `:fork`) and
take output options: `:samples :final` (default) emits each chain's last
state; `:samples :all` emits every state after the first `:burn` moves,
equally weighted.

## Streaming

`(smc/stream model n opts)` (`org.replikativ.foerster.smc`) is a CPS
operation resolving a step `{:measure :push :done? :close}` once every
particle waits at its next `:stream` site; `((:push step) value)` scores the
value and runs on ([streaming notebook](https://replikativ.github.io/foerster/foerster.streaming.html)).

## Programmable inference and counterfactuals

- `org.replikativ.foerster.gfi`: `simulate`, `generate`, `assess`, `update`,
  `regenerate`, `mh`, `close!` — CPS operations on traces of a model spin.
- `org.replikativ.foerster.involutive/step`: an involutive MCMC move.
- `(counterfactual/counterfactual model {:evidence :interventions
  :particles})`: twin-world counterfactuals of the model spin.

See the [programmable](https://replikativ.github.io/foerster/foerster.programmable.html)
and [counterfactuals](https://replikativ.github.io/foerster/foerster.counterfactuals.html)
notebooks.
