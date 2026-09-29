# Worlds

A pure model only computes. A model that runs in an application — an agent
reasoning about a room, a simulation over a database — also reads and writes
the **systems** of the world it runs in (databases, repositories, a business
book), registered with spindel's yggdrasil. foerster's particle methods run
such models in **canonical forks** of the caller's world, so that no particle
writes reach the caller. The [worlds notebook](https://replikativ.github.io/foerster/foerster.worlds.html)
walks through it with runnable examples.

## World policies

| `:world-policy` | Where particles run |
|---|---|
| `:fresh` (default) | fresh, empty worlds — for pure models |
| `:fork` | canonical forks of the caller's world |

`:fork` is supported by the particle methods: `smc-infer`,
`importance-sampling`, `pimh-infer`, `pgibbs-infer`, `pgas-infer`,
`ipmcmc-infer`, `bbvi-infer`, and `kernel-infer` with the prior kernel or a
custom `PInferenceKernel`. Markov-chain kernels (`single-site-mh-kernel`,
`random-walk-mh-kernel`, `block-gibbs-kernel`, `hmc-kernel`) run in fresh
worlds only and refuse `:fork`.

```clojure
(sp/with-context world
  @(infer/smc-infer (model) 500
     {:world-policy :fork
      :world-opts {:systems #{"knowledge" "repository"}}}))
```

Options for `:fork`:

| Option | Meaning |
|---|---|
| `:world-opts` | fork options: `:systems` — `:all` (default), `:none` or a set of system ids visible in the particles; `:rights`; `:snapshots` |
| `:authority` | a spindel `world.scope/PResourceAuthority` (budgets, below) |
| `:grant` | what the inference draws from the caller's wallet |
| `:executor` | the executor the worlds run on (default: the caller's) |

## How a canonical inference runs

```text
caller's world
  -> fork the root, owned by the inference's world scope
  -> copy the root into N frozen particles, each running the whole model
  -> run until every particle waits at an observation
  -> resample: copy each selected ancestor as often as it was selected,
     abandon the others
  -> resume, …
  -> collect values and traces into an EmpiricalMeasure of Samples
  -> discard every world, then deliver the result
```

- **Every particle runs the whole model.** In pure inference, the part of a
  model before its first random choice runs once and is shared; a model in a
  world may make effects that are random without a sample site (a call to a
  language model, a tool), and sharing that prefix would give every particle
  the same answer.
- **Particles are copies of worlds** (spindel `effects.savepoint/copy`,
  `world.scope/copy!`): three copies of an ancestor are three independently
  writable worlds, and the ancestor's world continues in them instead of
  unwinding. An ancestor selected by none is abandoned: its computation
  unwinds (its `finally` blocks run) in its own world.
- **A method of several sweeps** (particle MCMC, IPMCMC, BBVI) runs each
  sweep in canonical worlds of its own.
- **However inference ends** — a result, a failure of the model, or the
  cancellation of the inference spin — every world is discarded before the
  outcome is delivered. The measure holds `Sample`s: each particle's value,
  trace and its world's settled descriptor (`measure/world-descriptors`),
  never a live context or settlement authority.

## What cannot be copied

Particles are copies, so the structural **grades** of the world's systems
apply (`ygg/register!` `:grade`). Copying is refused — before the model runs
— for a world holding a system that is

- `:affine` (a live handle: an open connection, a device),
- `:divisible` (a conserved quantity kept in the world rather than by a
  resource authority),
- `:linear` without `:realize` or `:intents` (legal numbers, seals, sends
  that do not settle by replay), or
- `:shared` (not forked at all).

A system that settles by replaying intents — a book whose legal numbers are
assigned when a world is merged — may be copied. See spindel's
[forking guide](https://github.com/replikativ/spindel/blob/main/docs/forking.md).

## Budgets

A fork copies state; it must not copy authority to spend. With `:authority`
and `:grant`:

- the root draws `:grant` from the caller's wallet;
- every particle gets an even share of the root's wallet, and at each
  resampling every copy an even share of what its ancestor has left
  (`PResourceAuthority/balance`) — integer amounts are rounded down, and the
  remainder stays with the ancestor's world;
- what a world has not spent goes back to the wallet it came from when the
  world is discarded, so after inference the caller's wallet holds everything
  that was not spent.

Because shares are re-split at every resampling, a lineage that is copied
often runs on less; size the grant for the number of particles and the
expected spend per particle.

## ClojureScript

Copying worlds is JVM-only. In ClojureScript canonical particles are forks
instead: grades are not checked, and budgets are not split — particles get no
wallet of their own.

## Failure and recovery

A failure of the model fails the inference: the other particles are
cancelled and every world is discarded once quiescent. The error
(`::infer/inference-failed`, with the model's error as its cause) carries
process-local recovery operations for the case that the automatic cleanup
itself failed:

```clojure
(:world/recovery (ex-data error))
;; => {:status          the world scope's status (:open, :discarding, :discarded)
;;     :manager         the world scope (process-local)
;;     :await-quiescent a CPS operation resolving when no world is busy
;;     :cancel!         (fn []) requesting cancellation of every world
;;     :discard!        (fn []) returning a CPS operation that discards them
;;     :descriptors     portable descriptors: the root, then the particles}
```

Descriptors are safe durable and audit projections; the manager and the
operations are process-local. Cleanup is idempotent: concurrent callers share
one result, and a read-only preflight failure permits a retry.

## State placement

Particle-local program state belongs in the particle's world: signals,
spindel atoms, continuations, the trace and weight all fork with it. Durable
application records should store world descriptors, never contexts or
handles.
