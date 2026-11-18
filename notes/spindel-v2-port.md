# Porting foerster to spindel v2

Working note for the port (branch `feat/spindel-v2-port`); to be removed before merge.

## What foerster uses today

foerster depends on the old engine in four ways (counts over `src/`):

1. **Runtime state as its own store**: ~80 `rtp/get-state` / `rtp/swap-state!` calls, almost all
   under `[:inference …]` (log-weight, slot, multiplicity, barriers, proposals, moves, base cells),
   plus `[:random/free]`.
2. **The savepoint session layer** (`effects.savepoint`): `open!`/`start!`/`close!`, `seed`,
   `derive-seed`, `pending?`, `resume`, `abandon`, `fork`, `copy`, `release-world!`,
   `in-callers-world`, `publish!` (choose, observe and factor are effects that *are* savepoints),
   `install-handlers!`.
3. **Traces** (`spindel.trace`): `run`, `replay`, `update`, `record!`, `earliest`, `anchor-id`,
   `release!`, `entries`, `policy`. A library over (2): anchors are forks of a pending savepoint.
4. **Engine plumbing**: `create-execution-context`/`stop-context!`, executors (`thread-pool-executor`,
   `spread!`), `spin-core/resume` inside effect handlers, `keyed/spin-of`, `world.scope` (authority,
   leases, `fork!`, `copy!`, `discard-when-quiescent!`, `await-quiescence`), and the engine key
   `[:engine/reuse-source]`.

## Mapping

| Old | v2 | Where the work is |
|---|---|---|
| `create-execution-context`, `stop-context!` | `v2/world`, `release-world!` | foerster |
| `rtp/get-state w [:inference …]`, `swap-state!` | `get-in-world`, `update-in-world!` (world state forks by value) | foerster, mechanical |
| `[:engine/reuse-source]` (clear the replay link) | nothing: v2 forks share rows copy-on-write; drop the calls | foerster |
| `spin`, `await` (`spin.cps`, `effects.await`) | `v2/spin`, `v2/await`, `v2/task` | foerster |
| custom effects resuming via `spin-core/resume` | `register-effect!` / `custom-effect!` with `k`/`ek` | foerster |
| `sp/publish!` inside choose/factor | the effect body calls `savepoint-effect!` (S1) | foerster |
| `sp/resume` | `resume!` (S2) | session layer |
| `sp/fork`, `sp/copy` | `instantiate` + `activate!` (S3/S4) | session layer |
| `sp/abandon` | `abandon!` (S7) | session layer |
| `open!`/`start!`/`close!`, terminal sites, `seed`, `derive-seed`, `pending?`, `release-world!` | **missing in v2** | new: spindel v2 session layer |
| `spindel.trace` (`run`, `replay`, `update`, …) | **missing in v2**; `replay` = instantiate the anchor's publication; `update` = rerun (spec §4.1) | new: spindel v2 trace library |
| `trace/update` re-running chosen spins | rerun sources (§4.1): **no public `rerun!` in v2 yet** | spindel core (small) |
| `keyed/spin-of` | creation keys exist (`creation-key!`); no public lookup | spindel core, check need |
| `in-callers-world` | `send-world!` (H4 cross-world ingress) | foerster |
| `world.scope` authority/leases/quiescence | scopes + fork markers (`open-scope!`, `fork!`, `release-world!`); quiescence wait to check | foerster + maybe spindel |
| executors, `spread!`, multi-chain HMC/NUTS | one world per chain (H4, spec §12); `offload` for blocking work (`compiled.clj`) | foerster |

## Plan

Spindel side, general (no foerster specifics, per "spindel stays general"):

- **S-a** `rerun!` on a computation (write to its rerun source, §4.1) — small core addition.
- **S-b** savepoint session layer on v2 (`org.replikativ.spindel.v2.savepoint`): sessions, terminal
  sites, seeds, `pending?`, `fork`/`copy` via instantiate/activate, `release-world!`.
- **S-c** trace library on v2 (`org.replikativ.spindel.v2.trace`), porting `spindel.trace`; its
  tests port with it.

foerster side, in dependency order:

- **F-1** runtime + state: worlds instead of contexts, world state instead of `rtp`, drop
  `reuse-source`.
- **F-2** effects: choose/observe/factor as v2 custom effects publishing savepoints.
- **F-3** runtime/trace, gfi, program on S-b/S-c.
- **F-4** inference algorithms (smc, smc2, scmc, tempering, pmcmc, mcmc moves, nuts), canonical/scope.
- **F-5** multi-chain drivers: one world per chain.
- Tests: 72 test namespaces touch spindel; run the full suite plus the benchmark comparison (perf
  #225/#226) at the end.

## Design: S-b session layer and S-c traces on v2

Constraints from v2 that shape it:
- A savepoint is supported only in a **task** body (not a computation, not a nested slice). A
  probabilistic program therefore runs as a task; re-execution from a changed site is S5's
  instantiate route, not T11.
- A publication keeps its frozen basis after `resume!` (S2) and can be instantiated any number of
  times (S3/S4). **An anchor is just the publication**: no "fork before resume" world per site, which
  the old layer needed. A held publication pins only its basis pages.
- Handlers run as host callbacks after the publishing commit, not inline in the effect.

Addresses: a site's address must survive re-execution with different control flow (otherwise MH
pairs the wrong choices). Effect indices shift when a branch adds choices; v2's creation keys
(site × per-run ordinal) do not. Proposal (spindel core, small): `register-effect!`'s `ctx` gains
`:key`, the same site/ordinal key computations use. Address = explicit `:id`, else `[site ordinal]`.

`org.replikativ.spindel.v2.savepoint` (S-b), the old API on publications:

| Old | v2 session layer |
|---|---|
| `open! world opts` | installs handlers + `[:savepoint/seed]` in world state, returns a session (host record: root world, worlds it made) |
| `start! session task` | runs the task in the root; observes it: result / error / abandoned terminal sites |
| `sp` map (`:savepoint/world :address :payload :seq :site`) | the same keys over a publication |
| `resume sp v` | `resume!` (in its own world) or `activate!` (an instance not yet born) |
| `fork sp` | `instantiate` → an `sp` pending in the new world; seed derived as before |
| `abandon sp` | `abandon!`; for an unborn instance, `release-world!` (nothing runs, S7) |
| `copy sp k` | `k` instances, then the source world released without running its finalizers |
| `pending? sp` | publication state `:pending` (or an unborn instance not yet activated) |
| `seed`, `derive-seed` | world state + the same content hash |
| `release-world!`, `close!` | `release-world!` per world; `close!` abandons pending publications and releases the session's worlds |
| `in-callers-world` | `send-world!` |
| `publish!` from an effect | the effect handler returns `[:savepoint site payload]` |

`org.replikativ.spindel.v2.trace` (S-c), `spindel.trace` on S-b: entries per address as before;
`replay` instantiates the anchor publication with the new value; `update` uses `rerun!`;
`release!` abandons the anchors the other trace does not share.

### Review (gpt-6.1-sol medium, 2026-11-14) and proposed decisions

The review (`spindel-v2-land/.internal/reviews/sb-design-sol.md`) found that the mapping is not yet
equivalent; these need spec decisions (§8.2) before S-b:

| # | Finding | Proposal |
|---|---|---|
| D1 | `copy` = instantiate k + release source would cancel the source (finalizers run) and, by S4, release the activated copies with their owner | new transition: consume a pending publication into k instances without unwinding the source run; the instances are owned by the session's owner scope, not the publication world |
| D2 | `trace/update` = `rerun!` does not fit: choices live in tasks, tasks cannot rerun | `update` becomes replay from the earliest changed address (instantiate its anchor); work before the site is reused (S5 adoption). Measure on the benchmarks; savepoints in computations only if replay is too slow |
| D3 | `[site ordinal]` addresses collide across task runs and shift when an earlier occurrence of the same site is skipped | address = enclosing frame (the creation keys of the entity chain, explicit scope ids) + site + ordinal within that frame; count the publishing effect before freezing |
| D4 | anchors: a resumed publication's basis is never disposed; `Publication` holds its world strongly | `dispose!` on a publication drops its basis; the publication holds only what instantiate needs |
| D5 | `close!` cannot wait for cleanup: T21 only requests release | `close!` returns a completion after the session's worlds are released and quiescent; needs the planned world join, interim: a quiescence wait in the session layer |
| D6 | instances see the publication-time state, not writes made while parked | writes for a particle go into its instance before `activate!` (S3 allows it); foerster's "write into the pending world, then fork" call sites change accordingly |
| D7 | grants / budget splitting / refusal without consumption, spend-key per occurrence | session-layer ledger (host data), not engine; spend-key = [world, publication occurrence] |

## Open

- `world.scope` authority/grant/balance (canonical.cljc, sandbox): which part foerster still needs
  on v2, and whether v2 scopes cover it.
- Whether `keyed/spin-of` (2 uses) is needed or can be replaced by holding the handle.
- `[:random/free]` and seeds: keep as world state in the session layer.
