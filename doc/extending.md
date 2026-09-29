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
from options, and every particle method takes it as `:policy`:

| Option | Effect |
|---|---|
| `:constraints` `{address value}` | fix those sites; their density enters the weight (conditioning) |
| `:interventions` `{selector-or-address transform}` | replace a site's mechanism: `{:do v}`, `{:dist d}`, `{:shift δ}`, or `{:policy (fn [choices])}`; nothing is scored |
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

## An MH move

`(trace/mh-step trace opts)` is one Metropolis–Hastings move over a trace;
`(trace/mh-chain trace n opts)` runs `n` of them. Both are CPS operations.

| Option | |
|---|---|
| `:select` `(fn [trace iteration])` | `{:targets #{address} :log-selection (fn [trace])}` — which sites to move and the log probability of selecting them (default: one latent uniformly) |
| `:propose` `(fn [savepoint old-entry])` | the proposal at a target (default: the prior; `trace/random-walk-proposal` for a random walk) — the reverse move is scored under the prior, so a custom proposal must be the prior or symmetric |
| `:step` | for `mh-chain`: a step function replacing `mh-step` (e.g. `hmc/within-gibbs`) |
| `:on-step` | for `mh-chain`: `(fn [{:keys [trace accepted?]}])` called after every step |

For moves that are not "redraw these sites" — scaling, swapping, splitting —
use involutive MCMC (`foerster.involutive/step`) and the generative function
interface (`foerster.gfi`), shown in the
[programmable inference notebook](https://replikativ.github.io/foerster/foerster.programmable.html).

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

## An algorithm

Every algorithm here is a handler of spindel savepoints: open a session on a
world, install handlers for `:inference/choose` and `:inference/factor`, and
resume, fork, copy or abandon the savepoints they receive.
`foerster.smc` is the reference; the [design](design.md) guide explains the
structure and spindel's [savepoints](https://github.com/replikativ/spindel/blob/main/docs/savepoints.md)
the primitive.
