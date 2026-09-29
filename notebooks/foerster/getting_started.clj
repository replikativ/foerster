;; # Getting Started

;; foerster runs probabilistic programs: ordinary Clojure code with
;; **sample** and **observe** sites, whose posterior an inference algorithm
;; approximates with weighted samples. Programs are
;; [spindel](https://github.com/replikativ/spindel) spins, and every site is a
;; spindel *savepoint*, so inference forks, copies and abandons the worlds the
;; program runs in.

(ns foerster.getting-started
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]))

;; ## A world to run in
;;
;; Spins run in an *execution context*, a spindel world. At the REPL one is
;; enough; `sp/with-context` binds it for the code inside.

(def world (sp/create-execution-context))

;; Seeding the generator makes every draw below reproducible
;; (see [Distributions and Reproducibility](https://github.com/replikativ/foerster/blob/main/doc/distributions.md)).

(random/set-seed! 42)

;; ## A first model
;;
;; A coin of unknown bias `p`, a uniform prior, and seven tosses of which six
;; came up heads. `sample` draws a latent value from a distribution; `observe`
;; scores the data under one. The value of the program is what we want to
;; know about.

(def tosses [true true false true true true true])

(defn coin []
  (spin
   (let [p (sample (dist/beta 1.0 1.0) :id :p)]
     (loop [[t & more] tosses]
       (when (some? t)
         (observe (dist/flip p) t)
         (recur more)))
     p)))

;; A few rules come from spindel: the model is a `spin`, and inside it the
;; sites are called directly — not from inside a function passed to `map`
;; or `for`, which the spin macro cannot see into; loop with `loop`/`recur`
;; instead (see the [Language](https://github.com/replikativ/foerster/blob/main/doc/language.md) guide).

;; ## Inference
;;
;; `smc-infer` runs Sequential Monte Carlo with 2000 particles. It returns a
;; spin; outside a spin (at the REPL) we deref it.

(def posterior
  (sp/with-context world
    @(infer/smc-infer (coin) 2000)))

;; The result is an *empirical measure*: weighted samples of the program's
;; value. `query` summarizes it:

(select-keys (infer/query posterior identity) [:mean :std-dev :quantiles])

;; `query` also returns the samples and their weights (`:samples`,
;; `:weights`). The posterior of `p` is Beta(7, 2): mean 7/9 ≈ 0.778,
;; standard deviation ≈ 0.131, median ≈ 0.80.

;; ## Looking at the posterior
;;
;; `predict` resamples the weighted particles into plain draws, which any
;; plotting library can show.

(def draws (infer/predict posterior m/get-value 4000))

(-> (tc/dataset {:p draws})
    (plotly/layer-histogram {:=x :p :=histogram-nbins 40}))

;; ## The evidence
;;
;; SMC also estimates the model's marginal likelihood p(data), the
;; normalizer of the posterior. Here it has a closed form: the integral of
;; p⁶(1 − p) over [0, 1] is B(7, 2) = 1/56.

{:estimate (m/log-marginal posterior)
 :exact (Math/log (/ 1.0 56))}

;; ## Continuous data
;;
;; The same pattern for a normal mean with known noise. Under the N(0, 10²)
;; prior the posterior of `mu` given the four measurements is again normal:
;; precision 1/100 + 4, so N(1.222, 0.499²).

(def measurements [1.3 0.7 1.9 1.0])

(defn normal-mean []
  (spin
   (let [mu (sample (dist/normal 0.0 10.0) :id :mu)]
     (loop [[y & more] measurements]
       (when y
         (observe (dist/normal mu 1.0) y)
         (recur more)))
     mu)))

(def mu-posterior
  (sp/with-context world
    @(infer/smc-infer (normal-mean) 2000)))

(select-keys (infer/query mu-posterior identity) [:mean :std-dev])

;; ## Where to go next
;;
;; - [Choosing an Inference Algorithm](foerster.algorithms.html)
;; - [Models in Worlds](foerster.worlds.html): models that read and write
;;   the systems of the world they run in
;; - the guides in [doc/](https://github.com/replikativ/foerster/blob/main/doc/README.md)
