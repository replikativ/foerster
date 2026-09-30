# Language

A foerster model is a [spindel](https://github.com/replikativ/spindel) spin
that calls three effects: `sample`, `observe` and `factor` (and, to record
what it computes, `deterministic`). Everything else is ordinary Clojure.

```clojure
(require '[org.replikativ.foerster.effects :refer [sample observe factor]]
         '[org.replikativ.foerster.dist :as dist]
         '[org.replikativ.spindel.spin.cps :refer [spin]])

(defn model []
  (spin
   (let [mu (sample (dist/normal 0.0 10.0) :id :mu)]
     (observe (dist/normal mu 1.0) 1.3)
     (factor (if (pos? mu) 0.0 -1.0))
     mu)))
```

The value of the spin — here `mu` — is what the posterior is over.

## Sites

| Effect | Meaning |
|---|---|
| `(sample d & opts)` | a latent value drawn from distribution `d` |
| `(observe d v & opts)` | the data `v`, scored under `d` (`log d(v)` enters the weight); evaluates to `v` |
| `(factor w)` | multiplies the weight by `exp(w)`: a score that is not the density of a value (a soft constraint, a reward, a likelihood computed elsewhere) |
| `(factor w :barrier true)` | the same, and SMC resamples there as at an observation (scored steps; see [algorithms](algorithms.md#steering-a-process-scored-steps)) |
| `(deterministic v :id a)` | records `v`, a value computed from the choices, in the trace under `a`; no randomness, no weight; evaluates to `v` |

A `deterministic` site makes a derived quantity part of every particle's
trace, so its posterior can be read without returning it from the model:
`(m/measure-stats measure #(m/site-value % :bmi))` (see
[posteriors](posteriors.md)). Traces that carry what a model computed are also
the training data of learned proposals.

Options of `sample` and `observe`:

| Option | Meaning |
|---|---|
| `:id` | the site's name, its address (see below) |
| `:init v` | the first state of a Markov chain starts the site at `v` (a block that cannot be drawn from needs it) |
| `:stream true` | the site's value arrives from outside, pushed by `smc/stream` |

To draw from the prior, run a model under `gfi/simulate` (see the
[programmable inference notebook](https://replikativ.github.io/foerster/foerster.programmable.html))
or importance sampling, which also record the trace. A model run with no
inference at all — deref'd in a bare world — simulates forward too: every
site draws afresh, observations add to the world's `[:inference :log-weight]`,
and nothing is recorded.

Distributions are in [`foerster.dist`](distributions.md).

## Addresses

Every site has an **address**, the name under which its value is recorded in
the trace, constrained (`gfi/generate`), intervened on or proposed. The rules
below hold under inference (every algorithm, `gfi`, `counterfactual`) and in
a bare forward run alike.

- A site with `:id` is addressed by it. A vector id is a hierarchical name:
  `:id [:step 3 :x]`.
- Without `:id`, the address is **structural**: the site's source location,
  how often it was reached in the current run of its spin, and the
  surrounding scope. It does not depend on which *other* sites ran before, so
  a site keeps its address when control flow upstream changes — which MCMC
  over programs with branches needs.
- `with-scope` (spindel `engine.addressing`) prefixes the names of the sites
  inside it: in a loop, `(with-scope [:step i] (sample d :id :x))` addresses
  `[:step 0 :x]`, `[:step 1 :x]`, … — Gen's `:step => i => :x`.

An `:id` must be unique within a run: a site reached twice under one name
is an error ("Duplicate site address"). In a loop, use a vector id that
includes the loop index, `:id [:y i]`, or `with-scope`.

Name the sites you will refer to — to constrain, intervene on or read from a
trace. Naming matters for reproducibility too: an unnamed site's address
includes the identity of its spin, and a spin created again in the same world
gets a new one, so its random stream changes (see
[reproducibility](distributions.md#reproducibility)).

## Selectors

Operations that act on a set of sites — `gfi/regenerate`, `gfi/mh`,
interventions — take a set of addresses or a spindel **selector**
(`org.replikativ.spindel.select`): `(select/id :x)`, `(select/path [:step :* :x])`,
`(select/prefix [:step])`, `(select/site :inference/choose)`, combined with
`union`, `intersection` and `complement*`.

## The rules of the spin macro

A spin body is transformed into continuation-passing style, so that an
inference algorithm can suspend it at a site, fork the world it runs in, and
resume it — any number of times. The transformation only sees code that is
lexically inside the `spin`:

- **Call sites directly in the spin body, not inside functions passed to
  `map`, `filter`, `reduce`, or in `for`/`doseq`.** The macro cannot see into
  them. Loop with `loop`/`recur`:

  ```clojure
  ;; wrong: the observes are inside a closure
  (spin (doseq [y ys] (observe (dist/normal mu 1.0) y)))

  ;; right
  (spin (loop [[y & more] ys]
          (when y (observe (dist/normal mu 1.0) y) (recur more))))
  ```

- **Use `await` for other spins and `track` for signals, never `@`** inside a
  spin body (spindel's rules apply unchanged).
- A model's randomness must come from its sites (or `foerster.random`), not
  from `rand`: only site draws are recorded, keyed and reproducible.

A model can call functions that build spins — a sub-model is a function
returning a `spin`, awaited from the parent:

```clojure
(defn noisy [mu] (spin (sample (dist/normal mu 1.0))))
(spin (let [x (await (noisy 0.0))] …))
```

## Interventions from inside

`(intervene! address value)` sets a site's value for the rest of the run,
Pearl's do-operator applied by the program itself; the site draws and scores
nothing. From outside a model, pass interventions to a trace policy instead
(see the [counterfactuals notebook](https://replikativ.github.io/foerster/foerster.counterfactuals.html)).
