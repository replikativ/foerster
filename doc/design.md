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
  `:stream` sites until a value is pushed.
- **Importance sampling** is the same with `:resample-threshold` 0: particles
  still park at each observation, but the population is never resampled.
- **A `PInferenceKernel`** (`kernel-infer`) runs as SMC whose latent sites
  take the value the kernel's `step` gives, its `:log-weight-delta` added to
  the weight; the prior kernel draws from the prior.
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
- Resample-move inside conditional SMC (particle Gibbs, PGAS), refused
  today: keeping the retained particle unmoved while the others move leaves
  the conditional target off (an exact enumeration of one sweep puts its
  stationary law off by 2·10⁻⁴ to 3·10⁻³ in total variation). The exact
  construction treats the retained path as the state *after* each barrier's
  move and draws its state before the move backwards through the move's
  reversal (for a reversible MH kernel, the kernel itself) — exact to 10⁻¹⁵
  in the same enumeration. It needs the retained path drawn backwards
  through every barrier's move before a sweep, and slot 0 switched from its
  pre-move to its post-move values at each barrier.

## What else the same programs admit

| Method | How |
|---|---|
| asynchronous SMC, particle cascade | no barrier: decide per arriving savepoint whether to fork, continue or abandon |
| twisted SMC, process rewards | `factor` with a value estimate; the telescoping correction is a second factor |
| reweighted wake-sleep, inference compilation | per-site `:log-prob` and `:log-proposal` in the trace; gradients by the embedding |
| nested inference | an inner world with its own handler table; nesting is by world |
