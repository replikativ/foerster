# Extending

The ready-made algorithms are handlers over the same traces and policies;
you can write your own at each level. From least to most involved:

## A distribution

Implement `dist/Distribution` — see
[distributions](distributions.md#your-own-distribution).

## A block

A group of latents with a log density and its gradient, for HMC — see the
[blocks notebook](https://replikativ.github.io/foerster/foerster.blocks.html)
and, for compiled densities,
[foerster-raster](https://github.com/replikativ/foerster-raster).

## A policy

A **policy** decides every site of a run. `foerster.trace/policy` builds one
from options, and the particle methods take it as `:policy` (all but BBVI,
whose guide decides the latent sites):

| Option | Effect |
|---|---|
| `:constraints` `{address value}` | fix those sites; their density enters the weight (conditioning) |
| `:interventions` `{selector-or-address transform}` | replace a site's mechanism: `{:do v}` and `{:policy (fn [choices])}` fix the value and score nothing; `{:dist d}` and `{:shift δ}` replace the law, and the site is then decided and scored under it like any other |
| `:draw` `(fn [savepoint old-entry])` | a custom proposal for latent sites: return `nil` (not my site), `{:value v :log-proposal lq}`, or `{:value v :symmetric? true}`; the weight gets `log p(v) − lq` |
| `:keep?` | reuse the values of the replayed trace, rescored |
| `:noise` `{address u}` | counterfactual noise for mechanisms |
| `:init?` | start sites at their `:init` option (a chain's first state) |

```clojure
;; a proposal that knows where the data is
(infer/importance-sampling (model) 1000
  {:policy (trace/policy
            {:draw (fn [sp _]
                     (when (= :mu (:savepoint/address sp))
                       (let [q (dist/normal 7.0 1.0)
                             v (dist/draw q)]
                         {:value v :log-proposal (dist/logpdf q v)})))})})
```

A policy's `:draw` is a proposal chosen from outside the program, per
inference call. A proposal the program itself computes — from its data, or
an amortized guide — is a sample site's `:proposal` option instead
([language](language.md#sites)); it is weighed the same way, and a `:draw`
that answers for a site takes precedence over it.

## An MH move

`(trace/mh-step trace opts)` is one Metropolis–Hastings move over a trace;
`(trace/mh-chain trace n opts)` runs `n` of them. Both are CPS operations.

| Option | |
|---|---|
| `:select` `(fn [trace iteration])` | `{:targets #{address} :log-selection (fn [trace])}` — which sites to move and the log probability of selecting them (default: one latent uniformly) |
| `:propose` `(fn [savepoint old-entry])` | the proposal at a target (default: the prior; `trace/random-walk-proposal` for a random walk) — the reverse move is scored under the prior, so a custom proposal must be the prior or symmetric |
| `:step` | for `mh-chain`: a step function replacing `mh-step` (e.g. `hmc/within-gibbs`) |
| `:on-step` | for `mh-chain`: `(fn [{:keys [trace accepted?]}])` called after every step |
| `:first-iteration` | for `mh-chain`: the number of its first move (default 0); a move's randomness is keyed by its number, so a chain continued by another `mh-chain` call starts where the first stopped |

**A proposal program** — `(gfi/mh-proposal trace proposal & args)` is
Gen's `metropolis_hastings(trace, proposal, args)`: `proposal` is
`(fn [choices & args])` returning a spin whose sample sites are named by
the model addresses it proposes, any number of them, jointly and correlated
as the program likes. The move runs the proposal forward, updates the model
with what it drew, and scores the reverse move by assessing the proposal on
the replaced values under the new choices; accepted on
w(update) + log q(old | new) − log q(new | old). The proposal must be able
to propose the reverse move.

```clojure
(defn along-the-ridge [{:keys [x y]}]          ; x and y move together
  (spin (let [x' (sample (dist/normal x 0.6) :id :x)]
          (sample (dist/normal (- y (- x' x)) 0.15) :id :y))))

((gfi/mh-proposal trace along-the-ridge) on-step on-error)  ; {:trace :accepted?}
```

For moves that are not "redraw these sites" — scaling, swapping, splitting —
use involutive MCMC (`foerster.involutive/step`) and the generative function
interface (`foerster.gfi`), shown in the
[programmable inference notebook](https://replikativ.github.io/foerster/foerster.programmable.html).
An involution may change the model's dimension (reversible jump): it sets
the new sites through `:choices` and names under `:removed` the sites whose
values it moved into the auxiliary variables, as in a split of one
component into two and the matching merge
(`test/…/reversible_jump_test.clj`).

## A kernel

A `PInferenceKernel` (`foerster.kernel`) decides latent sites of SMC:

```clojure
(def fixed
  (reify k/PInferenceKernel
    (kernel-id [_] :fixed)
    (step [_ world checkpoint trace]
      ;; checkpoint: {:source distribution :options site-options :address a}
      {:value 42.0 :log-weight-delta -1.0})))

(infer/kernel-infer (model) fixed 100 {:barrier-policy :none})
```

`step` returns the site's `:value` and optionally `:log-weight-delta`, what
the value adds to the particle's weight (default 0: a draw from the site's
distribution, which cancels against its density).

A custom `PInferenceKernel` is an SMC kernel. The Markov-chain kernels of
`foerster.kernel` are combined with `k/cycle` and `k/mixture`
([algorithms](algorithms.md#kernels)); a move of your own over traces is an
`mh-chain` `:step`.

## Learned twists and proposals

A steered program (`foerster.steer/model`) records each step's state and its
reward in the trace, and `foerster.learn` reads them back as training data:
`(learn/trajectories measure)` gives every particle's
`{:states :reward :weight :log-weight}`, and `(learn/draws measure n)`
resamples `n` of them by weight into unweighted draws from the target. A
value estimate trained on them is the next run's `:value`; a proposal
trained on them returns its steps as `(steer/weighted state log-w)`, with
log-w = log p − log q of the step, so the target does not change
([algorithms](algorithms.md#steering-a-process-scored-steps)). Training
itself is outside foerster.

## An algorithm

Every algorithm here is a handler of spindel savepoints: open a session on a
world, install handlers for `:inference/choose` and `:inference/factor`, and
resume, fork, copy or abandon the savepoints they receive.
`foerster.smc` is the reference; the [design](design.md) guide explains the
structure and spindel's [savepoints](https://github.com/replikativ/spindel/blob/main/docs/savepoints.md)
the primitive.
