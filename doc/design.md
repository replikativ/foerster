# Design

foerster is inference built from one primitive of
[spindel](https://github.com/replikativ/spindel): the **savepoint**. This
guide explains how, for readers who want to write an algorithm or understand
the ones there are. The savepoint itself — its laws, sessions, worlds — is
described in spindel's
[savepoints design](https://github.com/replikativ/spindel/blob/main/docs/savepoints.md).

## Sites are savepoints

A spindel savepoint is a point a program *publishes*: the rest of the
computation, offered to a handler, which may resume it (once), **fork** it
into new worlds (any number of times), **copy** it (consuming it), or
**abandon** it. Worlds are copy-on-write, so forking is cheap and a fork
continues exactly where the savepoint was, with the world as it was.

`sample`, `observe` and `factor` publish savepoints when the world they run
in handles their site, and simulate forward otherwise:

| Site | Payload | A policy resumes it with |
|---|---|---|
| `:inference/choose` (`sample`) | `{:dist d :options …}` | a draw from `d`, a kept value, a constraint, or a proposal's draw — having added `log d(v) − log q(v)` to the weight |
| `:inference/choose` (`observe`) | `{:dist d :observed? true :value y}` | `y`, having added `log d(y)` |
| `:inference/factor` | `{:log-weight w}` | `nil`, having added `w` |

Every inference algorithm is a handler over these sites, and they share the
same programs. This is Anglican's design — a model's `sample` and `observe`
are checkpoints of a continuation-passing interpreter — with the
continuation now a forkable world.

## Traces and policies

A **trace** (`spindel.trace`, `foerster.trace`) records one run: every
site's address, value and notes. The notes of an entry are `:dist`,
`:log-prob` — the log density **of every site, sampled ones included**, not
only observations — `:log-proposal` when the value was drawn from a
proposal, and flags: `:observed?`, `:constrained?`, `:kept?`, `:symmetric?`,
`:factor?`, `:intervened?`. The importance weight accumulates at
`[:inference :log-weight]` in the world.

A **policy** (`trace/policy`) decides sites: constraints, kept values from
an earlier trace, draws from custom proposals, interventions, counterfactual
noise, and the prior for everything else. `trace/run` and `trace/replay`
run a program under a policy.

## The algorithms as handlers

- **SMC** (`foerster.smc`) forks the first site into N particle worlds (in
  canonical worlds: copies the root at a start site, so every particle runs
  the whole model), runs each until it parks at an observation, and when all
  have arrived resamples if the effective sample size is below
  `:resample-threshold`·N: the chosen ancestors' parked savepoints are
  forked (or copied, in canonical worlds) into the next generation, the
  others abandoned. No coordinator and no barrier thread: the barrier is the
  arrival of the last particle. **Streaming** SMC parks particles at
  `:stream` sites until a value is pushed. In fresh worlds the session is
  closed before the measure is delivered, so a caller that stops its
  executor on the result cuts no cleanup short; in canonical worlds every
  world is discarded first (below).
- **Importance sampling** is the same with `:resample-threshold` 0: particles
  still park at each observation, but the population is never resampled.
- **A `PInferenceKernel`** (`kernel-infer`) runs as SMC whose latent sites
  take the value the kernel's `step` gives, its `:log-weight-delta` added to
  the weight; the prior kernel draws from the prior.
- **The particle cascade** (`foerster.cascade`) has no barrier: each
  savepoint arriving at an observation is decided on its own, against the
  running mean weight of those that arrived there so far, itself included —
  abandoned, continued, or forked into several children (Paige et al. 2014,
  Eq. 14).
- **Steering** (`foerster.steer`) is SMC over a program whose steps are
  random without a sample site and whose value estimates are barrier
  factors: twisted SMC, with the telescoping correction as the last factor.
- **Conditional SMC** keeps one particle on a retained trajectory: particle
  Gibbs, and with **ancestor sampling** (PGAS) the retained particle redraws
  its past, each candidate's future scored by forking its parked savepoint
  under a scoring handler.
- **PIMH** proposes a whole SMC sweep and accepts it on the ratio of evidence
  estimates; **IPMCMC** runs several SMC and conditional-SMC nodes and
  exchanges which of them are conditional by Gibbs updates on their
  evidence.
- **Metropolis–Hastings** is `replay` plus an accept step (`trace/mh-step`,
  `mh-chain`). A move selects a set of target sites (one for single-site
  MH, a block for block Gibbs), replays from the earliest, proposes at the
  targets, and keeps every other site's value rescored under its
  distribution as it is now:

      log a = [log p(new) − log p(old)]
            + [log q(old | new) − log q(new | old)]
            + [log s(targets | new) − log s(targets | old)]

  where `q(new | old)` is what the move drew afresh, `q(old | new)` what the
  reverse move would have to draw of the old trace, and `s` the probability
  of selecting the targets, which changes when the move changed how many
  sites there are. Single-site MH proposes from the prior; **random-walk
  MH** perturbs continuous sites with a normal step and is symmetric;
  **block Gibbs** selects blocks of sites by a selector and moves each with
  its own kernel. A kept value that fell out of its site's support is drawn
  again; a move whose reverse would keep the new value (overlapping
  supports) is refused. Limits: the reverse move is scored under the prior,
  so a custom proposal must be the prior or symmetric, and a block's
  membership must not depend on the move.
  Chain kernels compose: `k/cycle` runs several in turn, `k/mixture` one
  picked at random per iteration; both are kernels again and nest.
- **HMC-within-Gibbs** moves block sites along the gradient of the block's
  density — plus one other latent site by single-site MH per iteration — and
  accepts on the **full** trace's log joint, so an incomplete block density
  costs mixing, not correctness. Under the particle methods a block site is
  drawn from its `:sample` capability and weighted by its density over the
  density of the draw (`:sample-log-density`, required there).
- **Involutive MCMC** (`foerster.involutive`): an auxiliary draw and an
  involution on (choices, aux) with its log Jacobian.
- **BBVI** learns a mean-field q by stochastic gradient ascent on the ELBO
  with score-function gradients (`foerster.gradient`) and control variates;
  its samples are importance samples from q.
- **The generative function interface** (`foerster.gfi`) — `simulate`,
  `generate`, `assess`, `update`, `regenerate` — is Gen's, on the same traces.
- **Counterfactuals** (`foerster.counterfactual`) read every site as a
  structural equation of exogenous noise (`foerster.mechanism`): abduction
  infers the noise, action intervenes, prediction replays.

## Canonical worlds

With `:world-policy :fork` the model runs in a frozen fork of the caller's
world, the root, and its session's worlds join the root's scope; every
particle runs the whole model, is a copy of a world (grades checked, budgets
split), and every world is discarded before the result is delivered. See
[worlds](worlds.md). Markov chains run in fresh worlds only.

## Not implemented yet

- `:parents` in a site's payload: which earlier choices a site read — which
  no address scheme can recover — making a trace a graph (variables, edges,
  observed mask, values), the training datum of graphically structured
  amortized inference. (`:proposal`, a guide distribution the program
  computes, is implemented: a fresh draw comes from it and the weight takes
  log p − log q.)
- Markov chains in canonical worlds.
- Resample-move inside conditional SMC with ancestor sampling (PGAS),
  refused: PGAS redraws the retained particle's past from the other
  particles' pre-move states, and its exact construction with moves is not
  worked out.

## Incremental re-execution (Gen's combinators and argdiffs)

Gen's `Map`, `Unfold` and argdiffs let an update re-run only the parts of a
program whose inputs changed, which turns an MH sweep over n independent
items from O(n²) into O(n). foerster's replay today restarts at the
earliest changed site: the program before it is shared (the anchor is a
forked world), the rest re-runs with every other site kept at its value. In
a program that awaits one sub-spin per item, an update of item 1's site
re-runs the bodies of items 2 … n though nothing they read changed — the
result is right, the work is the rest of the program.

spindel already reuses unchanged work across worlds: a replay names the old
world its reuse source, and a computation spin registering again adopts the
old node — result, deps and children — when its subtree is clean and every
input outside it is identical. A spin that reached a savepoint is a reuse
barrier, deliberately: its decisions belong in the new trace, and adopting
the spin would skip recording them.

The increment is to adopt such a spin **with its trace**: when a sub-spin's
captured inputs and outside deps are identical, copy its subtree's trace
entries (addresses, values, notes) and their log-probability contributions
into the new trace and world instead of re-running its body. The pieces it
needs: the trace entries of a spin's subtree (spindel's trace knows which
savepoint belongs to which spin), a world-level splice of those entries and
of the weight they carry, and the reuse barrier relaxed to "adopt with
trace" when the trace layer can splice. Per-item sub-spins then behave as
`Map`, a loop of awaited steps as `Unfold`, and the captured-locals check is
the argdiff.

## What else the same programs admit

| Method | How |
|---|---|
| reweighted wake-sleep, inference compilation | per-site `:log-prob` and `:log-proposal` in the trace; gradients by the embedding |
| nested inference | an inner world with its own handler table; nesting is by world |
