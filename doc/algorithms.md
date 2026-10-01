# Algorithms

A reference for every inference entry point in `org.replikativ.foerster.core`
(`infer` below). Each returns a spin resolving an `EmpiricalMeasure` (see
[posteriors](posteriors.md)); at the REPL deref it inside
`sp/with-context`, inside a spin `await` it. For which to use when, see the
[algorithms notebook](https://replikativ.github.io/foerster/foerster.algorithms.html).

## Options every particle method takes

`smc-infer`, `importance-sampling`, `pimh-infer`, `pgibbs-infer`,
`pgas-infer`, `ipmcmc-infer`, `bbvi-infer`, and `kernel-infer` with a
non-Markov-chain kernel:

| Option | Default | Meaning |
|---|---|---|
| `:world-policy` | `:fresh` | `:fork` runs the particles in canonical forks of the caller's world ([worlds](worlds.md)) |
| `:world-opts` | `{}` | fork options for `:fork`: `:systems` (`:all`, `:none` or a set of system ids), `:rights`, `:snapshots` |
| `:authority`, `:grant` | — | budgets under `:fork` ([worlds](worlds.md#budgets)) |
| `:executor` | spindel's default | the executor the worlds run on |
| `:resample-threshold` | `0.5` | SMC resamples when the effective sample size is below this fraction of the particles |
| `:policy` | the prior | a `foerster.trace/policy` deciding the sites: constraints, interventions, custom proposals ([extending](extending.md)) |
| `:anchors`, `:rejuvenate` | — | resample-move (below); `smc-infer` and `pimh-infer`, fresh worlds |
| `:resampling` | `:systematic` | `:stratified`, `:residual` or `:multinomial` (Douc, Cappé & Moulines 2005); `smc-infer`, `pimh-infer` |
| `:genealogy?` | `false` | record every resampling's ancestor indices in the measure's `:history` |
| `:smcp3` | — | `{:forward K :backward L}`: SMCP3 move-reweight steps at each observation (below); needs `:anchors` |

An option a method does not take is refused (`::infer/unknown-options`), and
so are `:world-opts`, `:authority` and `:grant` under `:world-policy :fresh`
(`::infer/fork-only-options`): an option that would be ignored is a mistake.
`importance-sampling` takes no `:resample-threshold`, BBVI no `:policy` or
`:resample-threshold` (it decides both), and the Markov-chain kernels only
`:executor` and `:world-policy :fresh`.

## Particle methods

| Function | What it does |
|---|---|
| `(smc-infer model n opts)` | Sequential Monte Carlo with `n` particles, resampling at observations. The measure's `m/log-marginal` estimates the evidence. |
| `(importance-sampling model n opts)` | `n` weighted runs, never resampled. |
| `(pimh-infer model n iterations opts)` | Particle independent MH: `iterations` SMC sweeps of `n` particles, each accepted on its evidence estimate; every sweep's particles pooled. |
| `(pgibbs-infer model n iterations opts)` | Particle Gibbs: iterated conditional SMC, each sweep keeping a trajectory drawn from the previous one; sweeps pooled. |
| `(pgas-infer model n iterations opts)` | PGAS: particle Gibbs whose retained particle redraws its ancestor at every observation. Costs a replay of the retained future per particle per observation. |
| `(ipmcmc-infer model n iterations opts)` | Interacting particle MCMC: `:num-nodes` (default 8) SMC nodes, `:num-csmc-nodes` (default half) of them conditional, exchanged by Gibbs updates on their evidence; `:all-particles?` (default `true`) emits every node's particles, `false` one per node. |
| `(bbvi-infer model n iterations opts)` | Black-box variational inference: learns a mean-field q in `iterations` updates from `n` samples each. Each site's q starts as its prior when first reached, then moves by gradient only (it does not follow a prior that depends on other latents). `:base-lr` (1.0), `:robbins-monro` (0.0; step size `base-lr / (t+1)^robbins-monro`), `:adagrad` (`true`: AdaGrad; γ ∈ (0,1): RMSprop; `false`: plain steps). The measure holds `n` importance samples from the final q; `(infer/get-variational-dists measure)` returns q, `{address distribution}`. |

The pooled particle-MCMC measures are MCMC estimates: their weights are
normalized per sweep, and `m/log-marginal` of the pool is not an evidence
estimate.

## Tempered SMC

`(tempered-infer model n opts)` (`org.replikativ.foerster.tempering`) moves a
population of complete program runs from the prior to the posterior through
the targets p(x)·L(x)^β, L the observations' and factors' likelihood, with β
rising from 0 to 1 (an SMC sampler; Del Moral, Doucet & Jasra 2006). Each
step chooses the next β so that the conditional ESS of the reweighting stays
at `:ess-target`·N (Zhou, Johansen & Aston 2016), resamples, and moves every
particle by Metropolis-Hastings at the new temperature — random walks scaled
by the population's spread on continuous sites, prior proposals elsewhere.

Use it for posteriors with separated modes, which a Markov chain does not
cross and SMC in program order only reaches if its early particles happen to:

```clojure
(infer/tempered-infer (model) 1000)                    ; β schedule in :temperatures
(infer/tempered-infer (model) 1000 {:waste-free 10})   ; Dau & Chopin 2022
```

| Option | Default | |
|---|---|---|
| `:ess-target` | 0.5 | higher: more, smaller temperature steps |
| `:moves` | one sweep | single-site MH moves per particle per step |
| `:scale` | 2.38 | random-walk scale in population standard deviations |
| `:waste-free` | — | P: resample N/P particles and keep every state of their P-step chains |
| `:max-steps` | 1000 | |

`m/log-marginal` estimates the evidence. Every move re-runs the program from
the moved site, so a step costs N sweeps of the model; fresh worlds only.

For data that arrive one observation at a time — IBIS (Chopin 2002) — use
SMC with resample-move on the static parameters, below.

## Resample-move

SMC resamples by copying good particles and dropping bad ones, so after a few
observations most particles share their early choices (a static parameter
sampled once is the extreme case). **Resample-move** (Gilks & Berzuini 2001)
gives every particle Metropolis-Hastings moves after each resampling, which
restores diversity without changing the target or the weights:

```clojure
(infer/smc-infer (model) 1000
  {:anchors #{:mu}                          ; where moves may start
   :rejuvenate {:moves 2                    ; per particle, per resampling
                :propose (trace/random-walk-proposal 0.2)}})
```

A move replays the program from an **anchor** — a kept fork of the world at a
latent site — up to the observation the particle is parked at, and accepts on
the MH ratio of the two partial traces. Which sites keep anchors is the
trade-off:

| `:anchors` | Moves may target | Cost |
|---|---|---|
| a set of addresses | those sites (static parameters) | one anchor per particle per site; a move replays from the site to the current observation, O(t) |
| `{:lag L}` | latent sites of the last L observations | anchors of older sites are released; a move replays O(L) steps — fixed-lag rejuvenation, and the choice for `smc/stream` |
| `:all` | every latent site | an anchor per latent site, shared between particles of one ancestor |
| `(fn [sp])` | sites the predicate names | |

Moves happen only when the population is resampled, so with the default
`:resample-threshold` they become rare as a static posterior concentrates
(Chopin 2002). The measure's `:rejuvenation` reports `:moves`, `:accepted`
and `:max-anchors`. Resample-move runs in fresh worlds only, and not in
conditional SMC (particle Gibbs, PGAS). A model whose moves reach effects
(a model call) runs them again on every replay.

## SMCP3: move-reweight steps

SMCP3 (Lew et al. 2023) generalizes resample-move: at each observation a
particle takes a step of a forward kernel K and is reweighted by a backward
kernel L instead of accepting or rejecting. K may change any latent site the
particle has passed — revise the past, propose from the data just seen — and
the weight keeps the particle properly weighted:

    w ← w · p(x')·L(u' | x') / (p(x)·K(u | x)) · |J|

```clojure
(def random-walk-kernel
  {:forward  (fn [{mu :mu} trace]            ; the particle's choices and trace
               (let [u (dist/draw (dist/normal 0.0 0.05))]
                 {:updates {:mu (+ mu u)}    ; new values
                  :log-q (dist/logpdf (dist/normal 0.0 0.05) u)
                  :reverse (- u)}))          ; what L would draw to go back
   :backward (fn [choices' u'] (dist/logpdf (dist/normal 0.0 0.05) u'))})

(infer/smc-infer (model) 1000 {:anchors #{:mu} :smcp3 random-walk-kernel})
```

K draws from `foerster.random` (it runs in the particle's stream), may
return `:log-jacobian` for a deterministic continuous map, and needs an
anchor at the earliest site it updates. Sites it does not update are kept,
and sites the replay reaches afresh are drawn from their prior. The weight's
variance depends on how well L reverses K: a symmetric random walk as L
multiplies the weights by π(x')/π(x), which is fine for steps small against
the posterior's spread and heavy-tailed for large ones; an L close to the
reversal of K under the target keeps them flat. Fresh worlds, not with
`:retained`.

## Steering a process: scored steps

A program whose proposals come from an outside process — a language model's
turn, a simulator — with no density to report, and whose data is a *score*
(a verifier, a judge, a process reward model), is steered by factors that
are barriers: `(factor w :barrier true)` makes SMC park and resample there,
as at an observation. `foerster.steer/model` builds the step loop:

```clojure
(infer/smc-infer
  (steer/model {:init s0
                :step (fn [state] (spin …))      ; the next state, from the process
                :value (fn [state] (spin …))     ; log ψ: the reward to come
                :reward (fn [state] (spin …))    ; the final log potential
                :done? (fn [state] …)})
  16 {:resampling :stratified})
```

The target is p(trajectory)·exp(reward): the process's own law tilted by the
reward (Korbak et al. 2022). The process's randomness has no sample site, so
it acts as the proposal and its density cancels; the weights are the
factors. A value estimate twists the intermediate targets — each step adds
log ψ_t − log ψ_{t−1} and the end reward − log ψ_T — which resamples early
without changing the final target (twisted SMC); without one, SMC is
best-of-N weighted by the reward. The model starts at `smc/start-site`, so
every particle takes every step itself.

## Static parameters: PMMH and SMC²

A state-space program with parameters θ — named sample sites — is filtered
best by SMC once θ is known; `org.replikativ.foerster.smc2` infers θ by
running SMC conditioned on it. With the parameter sites constrained, SMC's
evidence estimate is p(θ)·p̂(y | θ), so two parameter values compare by the
difference of their SMC log-evidences.

```clojure
;; a Markov chain over θ, each proposal scored by an SMC of 100 particles
(smc2/pmmh (model) 100 2000 {:params #{:sigma} :scale 0.2 :burn 200})

;; online, for a program whose data arrive at stream sites
(smc2/smc2 (model) {:params #{:sigma} :n-theta 50 :n-x 100})
;; → a step like smc/stream's: :push the next observation, :measure over θ
```

- **PMMH** (Andrieu, Doucet & Holenstein 2010) is exact for any number of
  particles; its acceptance falls as the SMC's evidence estimates get
  noisier, so give the inner SMC enough particles for a log-evidence
  standard deviation around 1. Its measure holds one particle per iteration,
  drawn from that iteration's SMC (θ and a state trajectory), and
  `:thetas` the chain.
- **SMC²** (Chopin, Jacob & Papaspiliopoulos 2013) carries θ-particles, each
  with an inner streaming SMC. A pushed observation reweights each by its
  inner evidence increment; when the θ-particles' ESS falls below
  `:ess-target`, they are resampled — a duplicate *forks* its inner
  population's worlds (`smc/stream`'s `:fork`), so copies evolve
  independently — and moved by PMMH steps that run a fresh inner filter on
  the data so far. `m/log-marginal` of its measure is the evidence of the
  data so far.

θ's prior is the program's own: a θ-particle starts from a simulation of the
program. A proposal that leaves the support — a distribution refusing its
parameters — has density zero and is rejected. Correlated PMMH, which needs
SMC driven by explicit noise, is not provided.

## Kernels

`(kernel-infer model kernel n opts)` runs a kernel from
`org.replikativ.foerster.kernel` (`k`):

| Kernel | Runs |
|---|---|
| `(k/prior-kernel)` | SMC from the prior (`:barrier-policy :none`: importance sampling) |
| a custom `PInferenceKernel` | SMC whose latent sites take the kernel's `step` value ([extending](extending.md)) |
| `(k/single-site-mh-kernel iterations opts)` | `n` independent Markov chains of single-site MH, proposals from the prior |
| `(k/random-walk-mh-kernel iterations {:step-size 0.1})` | random-walk MH on continuous sites |
| `(k/block-gibbs-kernel iterations selector block-kernels classifier opts)` | block Gibbs: `classifier` `(fn [address entry])` assigns sites to block ids, `selector` (`k/round-robin-selector`, `k/random-selector`) picks a block per iteration, `block-kernels` `{block-id kernel}` moves it (`k/prior-block-kernel`, `k/random-walk-block-kernel`) |
| `(k/hmc-kernel iterations {:step-size 0.1 :steps 10})` | HMC-within-Gibbs on block sites ([blocks](https://replikativ.github.io/foerster/foerster.blocks.html)) |

The Markov-chain kernels run in fresh worlds only (they refuse `:fork`) and
take output options: `:samples :final` (default) emits each chain's last
state; `:samples :all` emits every state after the first `:burn` moves,
equally weighted.

## Streaming

`(smc/stream model n opts)` (`org.replikativ.foerster.smc`) is a CPS
operation resolving a step `{:measure :push :fork :done? :close}` once every
particle waits at its next `:stream` site; `((:push step) value)` scores the
value and runs on ([streaming notebook](https://replikativ.github.io/foerster/foerster.streaming.html)).
`((:fork step))` resolves an independent copy of the population — every
waiting particle's world forked — to be pushed and closed on its own.

## Programmable inference and counterfactuals

- `org.replikativ.foerster.gfi`: `simulate`, `generate`, `assess`, `update`,
  `regenerate`, `mh`, `close!` — CPS operations on traces of a model spin.
- `org.replikativ.foerster.involutive/step`: an involutive MCMC move.
- `(counterfactual/counterfactual model {:evidence :interventions
  :particles})`: twin-world counterfactuals of the model spin.

See the [programmable](https://replikativ.github.io/foerster/foerster.programmable.html)
and [counterfactuals](https://replikativ.github.io/foerster/foerster.counterfactuals.html)
notebooks.
