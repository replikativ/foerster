(ns org.replikativ.foerster.conjugate
  "Conjugate parameters, carried as posteriors instead of sampled.

  A static parameter of a conjugate pair need never be drawn: the program
  holds its posterior as a value, scores each datum under the posterior
  predictive and updates the posterior in closed form —

    (loop [c (conjugate/beta-bernoulli 1 1) [y & more] ys]
      (if y
        (do (observe (conjugate/predictive c) y)
            (recur (conjugate/update c y) more))
        (sample (conjugate/posterior c) :id :p)))   ; the parameter, if needed

  — which is exact marginalization (Rao-Blackwellization; the conjugate case
  of delayed sampling, Murray et al. 2018). Under SMC the parameter cannot
  degenerate, since it is never resampled, and a streaming program pays the
  same for it at every step. Every operation is a plain function of plain
  values, so it composes with any foerster program and inference method.

  Families: `normal-mean` (Normal mean with known observation sd),
  `beta-bernoulli`, `gamma-poisson` (Gamma shape and SCALE, as
  `foerster.dist/gamma`) and `dirichlet-discrete`."
  (:refer-clojure :exclude [update])
  (:require [org.replikativ.foerster.dist :as dist]))

(defn normal-mean
  "The mean μ ~ Normal(mean, sd) of observations Normal(μ, obs-sd)."
  [mean sd obs-sd]
  {:family :normal-mean :mean (double mean) :sd (double sd) :obs-sd (double obs-sd)})

(defn beta-bernoulli
  "p ~ Beta(α, β) of observations Bernoulli(p) (0 or 1)."
  [alpha beta]
  {:family :beta-bernoulli :alpha (double alpha) :beta (double beta)})

(defn gamma-poisson
  "λ ~ Gamma(shape, scale) of observations Poisson(λ)."
  [shape scale]
  {:family :gamma-poisson :shape (double shape) :scale (double scale)})

(defn dirichlet-discrete
  "w ~ Dirichlet(α) of observations discrete(w) (indices 0 … n−1)."
  [alpha]
  {:family :dirichlet-discrete :alpha (mapv double alpha)})

(defn predictive
  "The posterior predictive distribution of the next observation."
  [{:keys [family] :as c}]
  (case family
    :normal-mean (let [{:keys [mean sd obs-sd]} c]
                   (dist/normal mean (Math/sqrt (+ (* sd sd) (* obs-sd obs-sd)))))
    :beta-bernoulli (let [{:keys [alpha beta]} c] (dist/bernoulli (/ alpha (+ alpha beta))))
    :gamma-poisson (let [{:keys [shape scale]} c] (dist/negative-binomial shape (/ 1.0 (+ 1.0 scale))))
    :dirichlet-discrete (dist/discrete (:alpha c))))

(defn update
  "The posterior after observing `y`."
  [{:keys [family] :as c} y]
  (case family
    :normal-mean (let [{:keys [mean sd obs-sd]} c
                       prec (+ (/ 1.0 (* sd sd)) (/ 1.0 (* obs-sd obs-sd)))
                       mean' (/ (+ (/ mean (* sd sd)) (/ y (* obs-sd obs-sd))) prec)]
                   (assoc c :mean mean' :sd (Math/sqrt (/ 1.0 prec))))
    :beta-bernoulli (if (== 1 y) (clojure.core/update c :alpha inc) (clojure.core/update c :beta inc))
    :gamma-poisson (-> c
                       (clojure.core/update :shape + y)
                       (clojure.core/update :scale #(/ % (+ 1.0 %))))
    :dirichlet-discrete (clojure.core/update-in c [:alpha (long y)] inc)))

(defn posterior
  "The parameter's posterior distribution."
  [{:keys [family] :as c}]
  (case family
    :normal-mean (dist/normal (:mean c) (:sd c))
    :beta-bernoulli (dist/beta (:alpha c) (:beta c))
    :gamma-poisson (dist/gamma (:shape c) (:scale c))
    :dirichlet-discrete (dist/dirichlet (:alpha c))))
