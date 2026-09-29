# Distributions and Reproducibility

## Distributions

`org.replikativ.foerster.dist` has the distributions models sample from and
observe under. They are portable Clojure — the same on the JVM and in
JavaScript — and named and parameterized like
[raster](https://github.com/replikativ/raster)'s (and
[Distributions.jl](https://juliastats.org/Distributions.jl/)), so a model's
site distributions and a compiled block's densities mean the same thing.

| Constructor | Support | Parameters |
|---|---|---|
| `(normal mu sigma)` | reals | mean, **standard deviation** |
| `(uniform a b)` | [a, b] | bounds |
| `(exponential lambda)` | x ≥ 0 | **rate** (mean 1/λ) |
| `(gamma alpha beta)` | x > 0 | shape, **scale** (mean αβ) |
| `(beta alpha beta)` | [0, 1] | shapes |
| `(poisson lambda)` | 0, 1, 2, … | mean |
| `(bernoulli p)` | 0 or 1 | P(1) |
| `(flip p)` | `true`/`false` | P(true) |
| `(discrete weights)` | indices 0 … n−1 | weights, normalized |
| `(categorical outcomes)` | any values | `{value weight}` or `[[value weight] …]` |
| `(dirichlet alpha)` | the simplex | concentrations |
| `(mvn mean cov)` | vectors | mean vector, covariance matrix |
| `(student-t nu)`, `(student-t nu mu sigma)` | reals | degrees of freedom, location, scale |
| `(chi-squared k)` | x > 0 | degrees of freedom |

Note the gamma's second parameter: it is the **scale**, as in raster,
Distributions.jl, NumPy and SciPy — not the rate, as in Anglican and PyTorch.

Every distribution supports

```clojure
(dist/draw d)        ; a sample, from the current generator (below)
(dist/logpdf d x)    ; log density — log mass for a discrete law; ##-Inf outside the support
```

and where they are defined `(dist/cdf d x)`, `(dist/quantile d p)`,
`(dist/mean d)` and `(dist/variance d)` (the normal, uniform and
exponential have all four; the gamma, Poisson, χ² and discrete laws have a
CDF). A count may be passed as `3` or `3.0` alike.

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
      (- mu (* b (Math/signum u) (Math/log (- 1.0 (* 2.0 (Math/abs u))))))))
  (-logpdf [_ x]
    (- (/ (- (Math/abs (- x mu))) b) (Math/log (* 2.0 b)))))
```

Draw from `random/uniform01` (or another distribution's `draw`), never from
`rand`, so that seeded runs stay reproducible. `dist/Univariate` adds `-cdf`
and `-quantile`, `dist/Moments` `-mean` and `-variance`. For counterfactuals
a distribution can also be read as a mechanism (`foerster.mechanism`), and
for BBVI it can provide gradients (`foerster.gradient`).

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
scheduled, and a proposal draws from a fresh stream. Draws outside any world
— the seeds of an inference's sessions, SMC's resampling at a barrier — come
from the process generator in program order.

What breaks reproducibility: randomness from `rand` or `Math/random`,
effects whose results vary (a language model's answer), and iteration over
unordered collections that feeds sites.
