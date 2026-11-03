# The programming model

Foerster's inference is assembled from a few kinds of pieces. Each kind has
one interface, plugs into a fixed set of places, and is checked where it is
taken. This guide names the pieces, says where each one goes, and lists
what is not unified yet. [Concepts](concepts.md) explains the objects they
act on (worlds, traces, measures, populations, blocks);
[Composing Inference](../notebooks/foerster/composing.clj) runs every piece
on a model with a known answer.

## The pieces

| Piece | Namespace | Interface | Made by |
|---|---|---|---|
| **Model** | `foerster.effects` | a `spin` whose sites are `sample`, `observe`, `factor`, `deterministic`; site options `:id :observe :proposal :init :stream` (others refused) | your code; `keyed` regions give it structure for incremental moves |
| **Proposal** | `foerster.proposal` | `(propose p site)` → `{:value :log-q}` or `{:value :symmetric? true}` or nil; `(log-q p site v)`; `(verified? p)`. A `site` is `{:address :dist :options :old :trace}` | `prior`, `distribution`, `random-walk`, `only`, `from-fn` |
| **Kernel** | `foerster.moves.kernel` | a Markov move that leaves the target invariant; `:only` selects its sites | `single-site-mh-kernel`, `finite-gibbs-kernel`, `block-gibbs-kernel`, `random-walk-mh-kernel`, `hmc-kernel`, `nuts-kernel`, `within-gibbs`, `delayed-acceptance-kernel`, `involutive-kernel`, `proposal-mh-kernel`, `kernel` (your own), composed by `cycle` and `mixture` |
| **Pair** | `foerster.moves.pair` | a move with auxiliary randomness: forward `(fn [choices trace])` → `{:updates :log-q :reverse :log-jacobian}`, backward `(fn [choices' u'])` → log density | `pair`, `from-involution` |
| **Population program** | `foerster.program` | `run` → `initialize` → `advance` / `resample` / `move` / `transport` / `fork` → `finish`, composed with `spin` and `await` | your code |
| **Result** | `foerster.measure`, `foerster.contract` | a measure with an `:inference-contract`: its semantics and how to read its evidence (`:exact`, `:unbiased`, `:estimate`, `:unverified`, `:unavailable`) | every entry point |

Two more choices apply to every run: the **world policy** (`:fresh`, or
`:fork` to start from the caller's state, [worlds](worlds.md)) and the
**seed**.

## Where each piece plugs in

| Piece | Takes it |
|---|---|
| Proposal | the particle methods' `:proposal` (SMC, importance sampling, PIMH, particle Gibbs, PGAS, iPMCMC); `mh-step`'s and resample-move's `:propose`; a policy's `:draw`; PMMH's `:propose` (θ as the old value) |
| Kernel | chains (`kernel-infer`, `infer :mh :rmh :nuts :kernel`); SMC's resample-move (`:rejuvenate {:kernel k}`); tempered SMC (`:kernel`); population programs (`program/move`); other kernels (`cycle`, `mixture`) |
| Pair | SMCP3 (`:smcp3`); population transport (`program/transport-kernel`); an involutive move is one (`pair/from-involution`) and is also a kernel (`involutive-kernel`) |
| Inference | a model, as a distribution (`infer/conditional`, carrying the inner contract); PMMH's filter |

## What is checked, and when

Every piece is refused where it is taken, before inference runs, when it
cannot be used correctly there:

- a misspelt site option, an option a method does not take;
- a `:proposal` together with a policy that already proposes;
- a kernel that leaves a latent site unmoved (a chain names the sites);
- a kernel that adapts where tuning must be fixed (rejuvenation, tempering,
  population moves), that cannot target a temperature below one (tempered
  SMC), or that cannot share the states it moves from;
- a rejuvenation kernel with conditional SMC (`:retained`), which needs a
  reversible move;
- a malformed pair, or a reversible jump used as an SMCP3 step;
- a PMMH filter whose evidence is not `:exact` or `:unbiased` (unless
  `:allow-unverified-evidence? true`).

What cannot be checked is stated instead. A proposal from `from-fn`, a kernel
of your own, an involution or a proposal program is its author's
obligation: results whose weights or moves depend on one carry
`:unverified` evidence (`foerster.contract`), and the docs of each say what
the obligation is (a density that matches its draws, invariance, a stated
Jacobian).

## Not unified yet

These are known seams; each is a separate interface today.

- **Three population engines.** Savepoint SMC (`foerster.inference.smc`)
  runs particles as suspended continuations of one program run; population
  programs (`foerster.program`) hold complete traces or values; θ-vector
  populations (`foerster.population.theta`) serve one-block models.
  `foerster.population.path` is the shared interpreter of tempering and θ
  populations, but savepoint SMC, the cascade and SMC² do not use it, and a
  population program cannot express barrier-wise SMC. Merging them — one
  population abstraction over three representations — is the next
  structural step.
- **Streaming has four interfaces:** `smc/stream`, SMC², SCMC
  (`start`/`append-batch`/`result`) and `program/fork`.
- **Tempering has three paths:** `tempered-infer` (traces, or θ vectors for
  one-block models), `program/temperature` with `advance`, and the θ path.
- **SMC² takes no user proposal or kernel** for its parameter moves.
- **Multi-site proposals are not proposals:** a proposal program
  (`proposal-mh-kernel`) and a pair propose several sites at once and have
  their own interfaces; a `foerster.proposal` proposes one site.
