# Language

A foerster model is a [spindel](https://github.com/replikativ/spindel) spin
that calls three effects: `sample`, `observe` and `factor`. Everything else is
ordinary Clojure.

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

Options of `sample` and `observe`:

| Option | Meaning |
|---|---|
| `:id` | the site's name, its address (see below) |
| `:init v` | the first state of a Markov chain starts the site at `v` (a block that cannot be drawn from needs it) |
| `:stream true` | the site's value arrives from outside, pushed by `smc/stream` |

`sample` also works outside inference: a spin run without an inference
handler simulates forward, drawing every latent from its distribution.

Distributions are in [`foerster.dist`](distributions.md).

## Addresses

Every site has an **address**, the name under which its value is recorded in
the trace, constrained (`gfi/generate`), intervened on or proposed.

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

Name the sites you will refer to — to constrain, intervene on or read from a
trace; structural addresses are enough for the rest.

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
