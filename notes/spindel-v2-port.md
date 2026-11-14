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

## Open

- `world.scope` authority/grant/balance (canonical.cljc, sandbox): which part foerster still needs
  on v2, and whether v2 scopes cover it.
- Whether `keyed/spin-of` (2 uses) is needed or can be replaced by holding the handle.
- `[:random/free]` and seeds: keep as world state in the session layer.
