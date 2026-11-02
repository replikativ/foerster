# Worlds

A pure model only computes. A model that runs in an application — an agent
reasoning about a room, a simulation over a database — also reads and writes
the **systems** of the world it runs in (databases, repositories, a business
book), registered with spindel's yggdrasil. foerster's inference methods run
such models in **canonical forks** of the caller's world, so that no
particle's or chain's writes reach the caller. The [worlds notebook](https://replikativ.github.io/foerster/foerster.worlds.html)
walks through it with runnable examples.

## World policies

| `:world-policy` | Where particles run |
|---|---|
| `:fresh` (default) | fresh, empty worlds — for pure models |
| `:fork` | canonical forks of the caller's world |

Coverage, exactly:

| Entry point | `:fork` |
|---|---|
| `smc-infer`, `importance-sampling` (`:smc`, `:importance`) | yes, `:batch` included; under `:fork` SMC refuses `:anchors`, `:rejuvenate` and `:smcp3` (resample-move and SMCP3 replay from anchors, which copied worlds do not keep yet) |
| `pimh-infer`, `pgibbs-infer`, `pgas-infer`, `ipmcmc-infer` | yes: every sweep (every node's) in canonical worlds of its own, all forked from one frozen fork of the caller |
| `bbvi-infer` | yes: every iteration's importance sweep, as above |
| `kernel-infer` with the prior kernel or another non-Markov `PInferenceKernel` (particle kernels) | yes |
| `:enumerate` (`infer/infer`) | yes: the model runs in a copy of the root, and every branch is a copy of its world |
| `kernel-infer` with a Markov-chain kernel (`single-site-mh-kernel`, `random-walk-mh-kernel`, `finite-gibbs-kernel`, `block-gibbs-kernel`, `hmc-kernel`, `nuts-kernel`, `delayed-acceptance-kernel`, `cycle`, `mixture`, …), `:mh`, `:rmh`, `:nuts` | yes: every chain in a world of its own; no `:authority` yet |
| `tempered-infer`, `parallel-tempering-infer` | yes: every particle (PT: the program's one run) in a world of its own; no `:authority` yet |
| `:pmmh` (`infer/infer`) | yes: every iteration's SMC in canonical worlds; no `:authority` yet |
| `program/run` with `program/trace-target` | yes: `(program/run {:world-policy :fork} …)` with `(trace-target model {:world-policy :fork})` |
| `smc2/smc2` | refused: its inner filters fork running populations, which copied particle worlds do not support yet |
| `:rmh` with `:compiled? true` | refused: compiled code has no worlds; read application state once and pass it to the model as data |
| `cascade/cascade`, `smc/stream` | no `:world-policy` option: always fresh worlds |

Every entry point above that takes no `:world-policy :fork` refuses it, so
it is never silently ignored.

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
- **Enumeration** runs the same way without resampling: the model runs in a
  copy of the root, and at each latent site its world is copied once per
  value of the site's support; every branch sees the caller's state and its
  own writes only, and splits its world's budget with its siblings.
- **However inference ends** — a result, a failure of the model, or the
  cancellation of the inference spin — every world is discarded before the
  outcome is delivered. The measure holds `Sample`s: each particle's value,
  trace and its world's settled descriptor (`measure/world-descriptors`),
  never a live context or settlement authority.

## Methods that run several times

A method that runs the model several times — particle MCMC (a sweep per
iteration), IPMCMC (a node per sweep), BBVI (a sweep per iteration), a
Markov chain (a world per chain), tempered SMC (a world per particle), PMMH
(an SMC per iteration), an inference program — forks the caller's world
**once**, when inference starts, and copies that fork once (so a world
that may not be copied is refused before any model runs). Every run forks
its own world from that frozen source, with a world scope of its own
(chains and tempered particles keep no descriptor of the worlds they
release). So every run sees the caller's state as it was when inference
started, whatever the caller does meanwhile, and `:grant` is drawn from the
caller's wallet once: a sweep takes what is left of it, IPMCMC's concurrent
nodes an even share. The source is discarded, and its wallet given back,
however inference ends.

## Incremental reuse

Pure inference reuses work between a trace and its proposals: a move on a
site of a keyed loop re-runs only that key (`spindel.trace/update`), and a
replay adopts computations of the replayed world that are unchanged. Both
assume that a computation depends on its arguments and choices only. A
model that reads state it — or a later part of it — wrote breaks that:
a keyed update re-runs the key in the world the trace *ended* in, so the
key's own earlier writes are applied again, and a replay may adopt a
computation that read a value since changed. Spindel does not track which
world state a computation read, so in canonical worlds foerster reuses
nothing incrementally: a move replays from the latest anchor before its
sites (the world as it was at that site), and a keyed model, whose sites
take no anchors, re-executes from its start. Reuse returns when Spindel can
tell which computations' state a world changed. (Under `:fresh` the same
hazard applies to a keyed model that writes its own state; keep such
writes out of keyed bodies, or run it under `:fork`.)

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
and `:grant` (SMC, importance sampling, enumeration, particle MCMC, IPMCMC,
BBVI; Markov chains, tempering, PMMH and programs refuse them for now,
since every move forks a world):

- the root draws `:grant` from the caller's wallet (a method of several
  runs: its frozen source does, once; each sweep's root then takes what is
  left of it, or an even share for IPMCMC's concurrent nodes);
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

## Cost

Canonical particles import managed caller state, whereas fresh worlds start
without it. Cost depends on the attached systems, retained state, mutations,
particle count and runtime revision. Forking, copying, trace retention and
retirement all contribute. There is no universal slowdown ratio; measure the
application and dependency versions you intend to use. Earlier measurements
on Spindel 0.1.112 are historical and do not characterize the current engine.

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
