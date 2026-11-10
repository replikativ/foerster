# Bounded-memory streaming SMC: design note

Status: decided (2026-11-09): (a) and (d). (a) is in `inference/smc.cljc`. (d) is on
spindel branch `feat/flat-forks` (worktree `../spindel-flat-forks`, off
`main` at 0.1.200). Nothing is committed or released yet.

## Outcome

Heap growth in MB after GC, every 500 pushes up to 4000 (random walk, N =
40, JVM):

| spindel                | `:retain :all`          | `:retain :filtering` |
|------------------------|-------------------------|----------------------|
| 0.1.200                | 2 6 8 12 13 16 18 21    | 1 5 6 9 10 12 14 16  |
| `feat/flat-forks`      | 0 2 2 4 5 6 7 8         | flat (−1 … 0)        |

Bounded memory needs both: the spindel fix removes the lineage term, and
`:retain` removes the trace term.

What spindel does (`feat/flat-forks`):

- A frozen fork's fork-time view of its parent is a `FrozenViewBackend`. It
  reads as the `ImmutableBackend` it replaces (`backend-type :immutable`),
  but holds its state map in a cell.
- When `world.scope/release!` releases a world, and no following fork reads
  through it, `state-backend/retire!` empties the world's overlay and cached
  view and drops its frozen view.
- Lineage is untouched. Descendants still hold the released world's context
  (`:parent-ctx`), so `fork-depth`, lineage and `:causal-follow` work as
  before. The context is now a few small records rather than a state.
- Reading a released world finds nothing, which the missing-state policy
  reports.

## Question

`smc/stream` is meant to run for as long as a page is open (the
christian-site pointer tracker pushes at 15–30 Hz). Its memory grows with the
number of pushes. A filter for a Markov model needs only the current state.
What should a bounded-memory mode look like?

## What actually grows (measured)

Random walk `x_t ~ N(x_{t-1}, 1)`, `y_t ~ N(x_t, 1)`, N = 40, JVM. Heap after
GC, every 500 pushes, while the stream is held:

| mode                    | MB after 500 … 4000 pushes      |
|-------------------------|---------------------------------|
| full traces (`:all`)    | 4 7 9 13 14 19 20 23            |
| `:retain :filtering`    | same slope (≈ 2.8 MB / 500)     |
| `:retain {:last 20}`    | 4 7 11 12 14 18 19 21           |

**Windowing the traces does not bound memory.** A heap-histogram diff over
3000 pushes shows what is held: one `OverlayBackend`, two `ExecutionContext`s,
two `SpinNode`s and one cancellation exception with its stack trace per
*resampling generation* (≈ 1 000 of each, for ≈ 1 000 resamplings).

The cause is the fork lineage in spindel. A forked world's context keeps
`:parent-ctx`, and its overlay backend keeps `:parent-backend`: reads of
paths the child never wrote fall through to the parent. A particle's world
is a fork of its ancestor's savepoint at every resampling, so the live
population holds a chain of every ancestor back to the root. The ancestral
lines coalesce, so the cost is about one world per generation, not N. The
measured depth of a live world's `:parent-ctx` chain went from 690 to 789
over 300 pushes.

That is ≈ 11 KB per resampling generation, ≈ 6 KB per push here, so roughly
10 MB a minute at 30 Hz. Per-generation cost is independent of N.

The traces share structure along the same lineage. Once the lineage is cut,
they become the next linear term. So both are needed.

## Options

### (a) `:retain` window in the stream options: prototyped

`:retain :all` (default) | `:filtering` | `{:last k}`. At every idle
barrier, each waiting particle's trace drops the sites before its last k
stream steps. A step ends at a stream site, so `:filtering` (k = 1) keeps
the step just observed (the state it observed, the observation, what
followed) and what was drawn since. For the walk that is `x_t, y_t, x_{t+1}`.
For the pointer model it is `s_{t-1}, p_t, z_t, s_t, p_{t+1}`. The measure's
`:history` keeps the last k barriers. `:measure` is then the posterior over
the last k steps' sites. Its marginals are the filtering marginals, which the
docstring would have to say.

- Cost: each idle barrier drops only the step that just aged out (a
  transient `dissoc!` over the dropped addresses) and copies the window's
  order vector. It must not use `subvec`, which would keep the history alive.
- Exactness: windowing changes no draw. With the same seed, the per-step
  posterior means of x_t are bit-identical across `:all`, `:filtering` and
  `{:last 3}` (tested).
- Lost: spindel's duplicate-address check for addresses that aged out, and
  `site-value` of an aged-out site (nil).
- Alone it bounds the trace but not the memory, because of the lineage
  above.

### (b) Model-declared state: not recommended

The program already carries forward exactly its loop variables: a spindel
continuation holds only those. The history that grows is the trace and the
world lineage, and neither is reachable from the model's future in a way the
model could declare more precisely than "the last k steps". (b) would add a
site kind and a reachability analysis to say what (a) says with one number.
Revisit only for a model whose stream steps are not the right unit, such as
irregular sub-steps.

### (c) Supported `restart` from the weighted population: useful as a fallback

The model is written as a function of its carried state, `(fn [state]
model)`. `smc/restart` takes a step and `(fn [sample] state)`. It closes the
population and starts a fresh session whose model first draws an ancestor
index from the current normalized weights, then runs from that particle's
state. That is a multinomial resampling step, so the restarted population is
an unbiased representation of the filtering distribution. The new measure's
`:log-normalizer` starts at the old one.

This differs from christian-site's hand-rolled `restart`, which moment-matches
the cloud into one Gaussian and is lossy for a multimodal posterior (the
mode mixture).

It bounds everything today, with no spindel change: a new session starts a
new lineage. Its cost is that the model must take its state as an argument,
and that a restart is an extra resampling.

### (d) Flat forks in spindel: needed for (a) to bound memory

A fork mode in which the child does not keep its parent:

- its overlay is based on a materialized snapshot of the parent's state
  rather than on the parent backend;
- it keeps no `:parent-ctx`, which spindel uses only to settle (merge) a
  fork into its parent, and SMC particles never merge back.

Materializing per fork costs the state size. Practical variants:

- flatten only when the parent is abandoned (the usual case at a resampling
  barrier, since the old generation is abandoned right after forking);
- or flatten when the chain depth exceeds a bound, amortizing the cost.

It would be a session fork option, set by `smc/run-particles` next to
`:fork-opts {:systems :none}`. Cheap, unrelated win in the same place:
`cancellation-error` is control flow and need not capture a stack trace
(≈ 10 % of the retained bytes per generation).

This is spindel work and needs its maintainers' view of the overlay
invariants (entity tombstones, `replaced-top-key?`, view caches).

## Interactions

| option                    | (a) `:retain`                                      | (c) restart                       | (d) flat forks         |
|---------------------------|----------------------------------------------------|-----------------------------------|------------------------|
| `:fork` (population copy) | works: children copy windowed traces (tested)      | works: fork the restarted stream  | works                  |
| `:anchors`/`:rejuvenate`  | refused in the prototype; possible with `:lag` ≤ k | works: anchors start afresh       | anchors are forks too: flatten them alike |
| `:smcp3`                  | refused (needs `:anchors`)                         | works                             | works                  |
| `:retained` (cSMC)        | refused: conditions on the whole trajectory        | n/a                               | works                  |
| pgibbs / PGAS             | n/a: they run `smc`, which refuses stream sites    | n/a                               | works                  |
| `:enumerate`              | works                                              | works                                 | works                  |

`:rejuvenate` with `:lag L ≤ k` could be allowed later. A move replays from
an anchor within the last L steps, and the entries upstream of the replayed
address cancel in the acceptance ratio, so a window covering the lag holds
everything a move reads. It needs an anchor-GC check that a windowed entry's
anchor is released with it.

## Recommendation

1. Take (a) as specified. It is cheap and exact, and it is the right
   user-facing knob.
2. Take (d) to spindel. Without it (a) does not bound memory. With it,
   `:retain :filtering` gives constant memory and constant push cost.
3. Offer (c) as a documented helper for runtimes that cannot wait for (d)
   (the cljs page now), or drop it once (d) lands. Christian-site can switch
   its moment-matching restart to the resampling one either way.

## Tests

- Exactness: per-step filtering means bit-identical across `:all`,
  `:filtering`, `{:last k}` under one seed (done).
- Window contents: the trace keys after t pushes are exactly the last k
  steps (done). `:history` has k entries (done).
- Refusals for `:anchors` and `:retained`, and an invalid `:retain` (done).
- `:fork` of a windowed population (done).
- Bounded state, deterministic rather than heap-based, for 10 000 pushes:
  - the first particle's trace size ≤ k × sites per step;
  - with (d), the depth of the `:parent-ctx` chain of every live world ≤ a
    constant.
- Heap after GC over 10 000 pushes as an `experiments/` benchmark, not a
  unit test: GC-dependent numbers are flaky in CI.
- ClojureScript: the window and refusal tests in a portable test.

## Stream values read as observed

On the branch this was first written on, stream values were recorded through
a `:constraints` policy, so their note said `:constrained? true` but not
`:observed? true`, and `retained-choices` listed `[:y t]` as a latent
choice. `main` had already fixed this differently, and better: `supply!`
decides a pushed value as an observation under the particle's own policy,
so it is tempered or intervened on like an observe. The test
`stream-values-read-as-observed` checks the observed note and
`retained-choices`.
