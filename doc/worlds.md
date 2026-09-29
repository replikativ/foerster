# Inference Worlds

Spindel's inference combinators execute probabilistic programs as `Spin`
values. Pure models can keep the default `:world-policy :fresh`, under which
the particle methods (`smc-infer`, `pimh-infer`, `pgibbs-infer`, `pgas-infer`,
`ipmcmc-infer`) run as savepoint handlers (`inference.smc`): particles are
frozen forks of savepoints, and the measure holds `Sample`s (result and
trace). A model that reads or changes registered room systems can request
canonical worlds:

```clojure
(smc-infer model 32
  {:world-policy :fork
   :world-opts {:systems #{:knowledge :repository}}
   :resample-threshold 0.5})
```

The model runs in a frozen `ygg/fork!` of the ambient execution context, the
root, owned by a world scope of the inference that the session's worlds join. Every particle method
(`smc-infer`, `importance-sampling`, `pimh-infer`, `pgibbs-infer`, `pgas-infer`,
`ipmcmc-infer`, `bbvi-infer`, `kernel-infer` with a PInferenceKernel) then runs
savepoint SMC there; a method of several sweeps runs each in canonical worlds of
its own. Each
particle is a frozen copy of the root and runs the whole model: a model that
reads or changes room systems may make effects that are random without a
sample site (a model call), so particles never share a prefix, as pure
inference does up to the first random choice. At resampling, each selected
ancestor is copied as many times as it was selected
(`effects.savepoint/copy`, `world.scope/copy!`): three copies are three
independently writable worlds, not three aliases to one context, and the
ancestor's world continues in them instead of unwinding. An ancestor selected
by none is abandoned: its computation unwinds (its `finally` blocks run) in
that world.

Particles are copies, so the structural grades of the world's systems apply
(`ygg/register!` `:grade`): a world holding a system that may not be copied —
a live handle (`:affine`), a `:linear` system that does not settle by
intents, a `:shared` system — is refused before the model runs. With
`:authority` (a `world.scope/PResourceAuthority`) and `:grant`, the root is
granted `:grant` from the caller's wallet, every particle an even share of the
root's, and every copy an even share of what its ancestor has left
(`PResourceAuthority/balance`); what a world has not spent goes back when it
is discarded. Copying is JVM-only; in ClojureScript particles are forks.

This makes inference a composition of existing Spindel operations:

```text
ambient world
  -> fork the root
  -> copy the root into N frozen particles, each running the model
  -> run until a probabilistic checkpoint
  -> score and select ancestors
  -> copy each selected ancestor, abandon the others
  -> resume
  -> project values and traces into an EmpiricalMeasure of Samples
  -> discard the speculative world tree, then the root
```

The `EmpiricalMeasure` holds `Sample`s: each particle's result, its trace and
its world's settled descriptor (`measure/world-descriptors`). It retains
neither contexts, the resampling ancestry, nor settlement authority. However
inference ends — a result, a failure, or the cancellation of its Spin — every
world is discarded before the outcome is delivered.

## Failure and recovery

Model failures are fail-fast. Spindel cooperatively cancels sibling particles,
tracks their terminal callbacks, and discards their worlds automatically once
they are quiescent. The thrown exception also contains actionable process-local
recovery operations:

```clojure
(:world/recovery (ex-data error))
;; => {:status :open
;;     :manager <process-local capability>
;;     :await-quiescent <CPS operation>
;;     :cancel! <host function>
;;     :discard! <host function>
;;     :descriptors [<portable fork descriptors> ...]}
```

`:descriptors` lists the root, then the particle worlds. Descriptors are
safe durable/audit projections. The manager, operations, and
its live handles are process-local capabilities. A supervising host can await
quiescence and retry cleanup if automatic settlement encountered a recoverable
preflight failure. Cleanup is idempotent, concurrent callers share one result,
and a read-only preflight failure permits a later retry; failures after mutation
remain terminal for explicit substrate recovery.

## State placement

Particle-local program state belongs in the particle `ExecutionContext`:
signals, Spindel atoms, continuations, trace, score, and checkpoint state all
fork with the world. The manager contains only host lifecycle capabilities for
the entire inference execution. Durable application records should store fork
descriptors, not contexts or handles.

The optional top-level `:executor` is used by every particle and forwarded to
its canonical fork. Without it, world-backed inference shares the ambient
world's executor, preserving scheduler ownership. The legacy `:fresh` policy
keeps its existing inference-local shared executor behavior.

## Resources are not copied authority

Forking a Kontor ledger or another registered system creates a hypothetical
branch of its state. It does not grant duplicate authority to spend external
compute, tokens, money, or network capacity. Pass `:authority` (a spindel
`world.scope/PResourceAuthority`) and `:grant`: the root draws `:grant` from
the caller's wallet, every particle an even share of it, and every copy an
even share of what its ancestor has left; what a world has not spent goes
back when it is discarded. Simulations can instead receive stubbed effects,
cheaper models, or a forked accounting scenario.

World scopes, Monte Carlo tree search and recursive SCI interpreters are
spindel's; see its docs.
