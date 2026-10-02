# Distributions and Reproducibility

## Distributions

`org.replikativ.foerster.dist` has the distributions models sample from and
observe under. They are portable Clojure — the same on the JVM and in
JavaScript — and named and parameterized like
[raster](https://github.com/replikativ/raster)'s, so a model's site
distributions and a compiled block's densities mean the same thing. (Close to
[Distributions.jl](https://juliastats.org/Distributions.jl/) too, except that
the exponential takes a rate and `discrete` counts from 0.)

| Constructor | Support | Parameters |
|---|---|---|
| `(normal mu sigma)` | reals | mean, **standard deviation** |
| `(uniform a b)` | [a, b] | bounds |
| `(exponential lambda)` | x ≥ 0 | **rate** (mean 1/λ) |
| `(gamma alpha beta)` | x > 0 | shape, **scale** (mean αβ) |
| `(beta alpha beta)` | [0, 1] | shapes |
| `(poisson lambda)` | 0, 1, 2, … | mean |
| `(negative-binomial r p)` | 0, 1, 2, … | failures before the r-th success, success probability p |
| `(bernoulli p)` | 0 or 1 | P(1) |
| `(flip p)` | `true`/`false` | P(true) |
| `(discrete weights)` | indices 0 … n−1 | weights, normalized |
| `(categorical outcomes)` | any values | `{value weight}` or `[[value weight] …]` |
| `(dirichlet alpha)` | the simplex | concentrations |
| `(mvn mean cov)` | vectors | mean vector, covariance matrix |
| `(student-t nu)`, `(student-t nu mu sigma)` | reals | degrees of freedom, location, scale |
| `(chi-squared k)` | x > 0 | degrees of freedom |
| `(binomial n p)` | 0 … n | trials, success probability |
| `(uniform-discrete a b)` | a, a+1, …, b−1 | bounds (upper exclusive) |
| `(log-normal mu sigma)` | x > 0 | mean and sd of log x |
| `(half-normal sigma)` | x ≥ 0 | scale of the normal it folds |
| `(cauchy x0 gamma)` | reals | location, scale |
| `(half-cauchy gamma)` | x ≥ 0 | scale |

Note the gamma's second parameter: it is the **scale**, as in raster,
Distributions.jl, NumPy and SciPy — not the rate, as in Anglican and PyTorch.

Every distribution supports

```clojure
(dist/draw d)        ; a sample, from the current generator (below)
(dist/logpdf d x)    ; log density — log mass for a discrete law; ##-Inf outside the support
```

and some of

```clojure
(dist/cdf d x) (dist/quantile d p) (dist/mean d) (dist/variance d)
```

| Distribution | `cdf` | `quantile` | `mean`, `variance` |
|---|---|---|---|
| normal, uniform, exponential, Poisson | ✓ | ✓ | ✓ |
| log-normal, half-normal, uniform-discrete | ✓ | ✓ | ✓ |
| Cauchy, half-Cauchy | ✓ | ✓ | — (no moments) |
| binomial | — | — | ✓ |
| gamma, χ² | ✓ | — (throws) | ✓ |
| discrete | ✓ | ✓ | — |
| beta, Bernoulli, Dirichlet, Student-t | — | — | ✓ |
| mvn | — | — | ✓ (`variance`: the diagonal) |
| flip, categorical | — | — | — |

An operation a distribution does not have throws. A count may be passed as
`3` or `3.0` alike.

The special functions behind them — `lgamma` (Lanczos), the normal CDF
(through the incomplete gamma function, precise far into the tail),
`normal-quantile` (Wichura's AS241), `regularized-gamma-p` and `-q` — are
public in the same namespace. They are checked against Apache Commons Math.

### Your own distribution

A distribution is anything implementing the `dist/Distribution` protocol:

```clojure
(defrecord Laplace [mu b]
  dist/Distribution
  (-draw [_]
    (let [u (- (random/uniform01) 0.5)]
      (- mu (* b (if (neg? u) -1.0 1.0) (Math/log (- 1.0 (* 2.0 (Math/abs u))))))))
  (-logpdf [_ x]
    (- (/ (- (Math/abs (- x mu))) b) (Math/log (* 2.0 b)))))
```

Draw from `random/uniform01` (or another distribution's `draw`), never from
`rand`, so that seeded runs stay reproducible. `dist/Univariate` adds `-cdf`
and `-quantile`, `dist/Moments` `-mean` and `-variance`.

For counterfactuals a distribution must also be readable as a *mechanism*
(`foerster.mechanism/PMechanism`: `noise`, `push`, `abduct`): the normal,
uniform, exponential, flip, Bernoulli and discrete are. For BBVI, a site's
distribution needs gradients (`foerster.gradient/PDistGradient`): the
normal, gamma, beta, flip, exponential, Dirichlet, discrete and uniform have
them.

## Reproducibility

Every draw comes from `org.replikativ.foerster.random`: a xoshiro128**
generator on 32-bit words, so **the same seed draws the same numbers on the
JVM and in JavaScript**.

```clojure
(random/set-seed! 42)   ; seeds the process generator
```

A seeded inference run is reproducible even though its particles and chains
run concurrently, on several threads, in whatever order they arrive. Every
draw made while deciding something in a world — a sample site, the site an
MH move picks, its acceptance — reads a **stream keyed by that world's seed
and what is being decided**:

    stream(world, key) = generator seeded by hash(seed(world), key)

World seeds are derived per fork from the parent's seed, the site's address
and the fork's index, so the same run draws the same numbers however it is
scheduled, and a proposal draws from a fresh stream. SMC's resampling at a
barrier draws from a stream of its run's seed and the barrier's index; the
seeds themselves (one per SMC run, BBVI iteration or chain) come from the
process generator in program order — IPMCMC draws its nodes' seeds before
running them in parallel.

Streams are keyed by **address**. A named site (`:id`) is addressed by its
name, so a seeded run draws the same numbers wherever and however often the
model is created. An unnamed site's address includes the identity of its
spin, and a spin created again in the same world gets a new identity: run
such a model twice in one world with the same seed and it draws differently.
Name the sites of a model you want to reproduce, or create it in a fresh
world (`sp/create-execution-context`) for each run.

The moves draw from streams too: the accept steps of MH, `gfi/mh`,
`involutive/step` (and its auxiliary draw) from the worlds they decide, PIMH's
from a stream per iteration, and a counterfactual's abducted noise from the
factual world's. What breaks reproducibility: randomness from `rand` or
`Math/random`, effects whose results vary (a language model's answer), and
iteration over unordered collections that feeds sites.

## Validation

A constructor refuses parameters outside their domain (a negative standard
deviation, a probability above 1, weights that are all zero) with
`::dist/invalid-parameters`; `quantile` refuses a probability outside
[0, 1]. `logpdf` is `##-Inf` outside the support, which for the Dirichlet is
the simplex. A categorical outcome listed twice carries both weights.
