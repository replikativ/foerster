(ns org.replikativ.foerster.mechanism
  "Sample sites as structural equations x = f(u), u exogenous noise.

  A distribution is read as a mechanism: `noise` draws u, `push` computes the
  value from it, `abduct` goes back from a value to its noise. For an
  invertible mechanism (location-scale, inverse CDF of a continuous law)
  abduction is exact. For a discrete one, many u give the same value, and
  `abduct` draws u from its posterior given the value — here under the
  inverse-CDF convention x = F⁻¹(u), u ~ U(0,1), under which that posterior is
  uniform on the value's CDF interval. Which convention a discrete site uses
  changes counterfactual answers and cannot be told from data, so it is fixed
  here and documented rather than guessed.

  Parameters come from the distribution at the site, so replaying a site with
  its old u under new parents is the counterfactual value."
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]))

(defprotocol PMechanism
  (noise [d] "A fresh draw of the site's exogenous noise.")
  (push [d u] "The value the mechanism gives noise `u`.")
  (abduct [d x] "Noise that gives value `x`: exact when the mechanism is
    invertible, a draw from the posterior over u otherwise."))

(defn- uniform-between [a b] (+ a (* (- b a) (m/uniform01))))

(extend-type org.replikativ.foerster.dist.Normal
  PMechanism
  (noise [_] (dist/draw (dist/normal 0.0 1.0)))
  (push [d u] (+ (:mu d) (* (:sigma d) u)))
  (abduct [d x] (/ (- x (:mu d)) (:sigma d))))

(extend-type org.replikativ.foerster.dist.Uniform
  PMechanism
  (noise [_] (m/uniform01))
  (push [d u] (+ (:a d) (* (- (:b d) (:a d)) u)))
  (abduct [d x] (/ (- x (:a d)) (- (:b d) (:a d)))))

(extend-type org.replikativ.foerster.dist.Exponential
  PMechanism
  (noise [_] (m/uniform01))
  (push [d u] (/ (- (Math/log (- 1.0 u))) (:lambda d)))
  (abduct [d x] (- 1.0 (Math/exp (- (* (:lambda d) x))))))

;; flip(p) = [u < p]: true takes u ∈ [0, p), false u ∈ [p, 1)
(extend-type org.replikativ.foerster.dist.Flip
  PMechanism
  (noise [_] (m/uniform01))
  (push [d u] (< u (:p d)))
  (abduct [d x] (if x (uniform-between 0.0 (:p d)) (uniform-between (:p d) 1.0))))

(extend-type org.replikativ.foerster.dist.Bernoulli
  PMechanism
  (noise [_] (m/uniform01))
  (push [d u] (if (< u (:p d)) 1 0))
  (abduct [d x] (if (== 1 x) (uniform-between 0.0 (:p d)) (uniform-between (:p d) 1.0))))

;; discrete(w) = the first index whose cumulative weight exceeds u·Σw
(extend-type org.replikativ.foerster.dist.Discrete
  PMechanism
  (noise [_] (m/uniform01))
  (push [d u]
    (let [target (* u (:total d))]
      (loop [i 0 acc 0.0 [w & more] (:weights d)]
        (let [acc' (+ acc w)]
          (if (or (< target acc') (empty? more)) i (recur (inc i) acc' more))))))
  (abduct [d x]
    (let [total (:total d)
          x (long x)
          lo (reduce + 0.0 (take x (:weights d)))]
      (uniform-between (/ lo total) (/ (+ lo (nth (:weights d) x)) total)))))

(defn mechanism?
  "Whether distribution `d` can be read as a mechanism."
  [d]
  (satisfies? PMechanism d))

(defn mechanism
  "Give distribution `d` an explicit structural interpretation. `ops` has
  zero-argument :noise, one-argument :push and one-argument :abduct functions.
  :abduct samples the conditional noise, not an arbitrary inverse. These
  functions declare the coupling across interventions; the density alone
  does not determine it."
  [d {:keys [noise push abduct]}]
  (when-not (and (fn? noise) (fn? push) (fn? abduct))
    (throw (ex-info "A mechanism needs :noise, :push and :abduct functions"
                    {:type ::invalid-mechanism})))
  (reify
    dist/Distribution
    (-draw [_] (push (noise)))
    (-logpdf [_ x] (dist/logpdf d x))
    dist/Finite
    (-support [_] (dist/support d))
    dist/Continuous
    (-continuous? [_] (dist/continuous? d))
    PMechanism
    (noise [_] (noise))
    (push [_ u] (push u))
    (abduct [_ x] (abduct x))))

(defn shifted
  "Translate a law, preserving its noise coordinate when it is a mechanism."
  [d delta]
  (let [law (reify dist/Distribution
              (-draw [_] (+ (dist/draw d) delta))
              (-logpdf [_ x] (dist/logpdf d (- x delta)))
              dist/Finite
              (-support [_] (when-let [support (dist/support d)] (mapv #(+ % delta) support)))
              dist/Continuous
              (-continuous? [_] (dist/continuous? d)))]
    (if (mechanism? d)
      (mechanism law {:noise #(noise d)
                      :push #(+ (push d %) delta)
                      :abduct #(abduct d (- % delta))})
      law)))

(defn inverse-cdf
  "Explicit inverse-CDF mechanism for a law with working cdf/quantile.
  `lower-cdf`, when supplied, returns P(X < x) for a discrete law. Without
  it the law is interpreted as continuous. Category ordering is part of
  this structural assumption. Unsupported quantiles fail explicitly."
  ([d] (inverse-cdf d nil))
  ([d lower-cdf]
   (mechanism d {:noise m/uniform01
                 :push #(dist/quantile d %)
                 :abduct (fn [x]
                           (let [hi (dist/cdf d x)]
                             (if lower-cdf
                               (let [lo (lower-cdf x)]
                                 (when-not (< lo hi)
                                   (throw (ex-info "Cannot abduct a zero-mass value"
                                                   {:type ::impossible-value :value x})))
                                 (uniform-between lo hi))
                               hi)))})))

;; These laws have implemented continuous quantiles. Do not extend every
;; Univariate: e.g. Gamma declares the protocol but its quantile is refused.
(extend-protocol PMechanism
  org.replikativ.foerster.dist.LogNormal
  (noise [_] (m/uniform01)) (push [d u] (dist/quantile d u)) (abduct [d x] (dist/cdf d x))
  org.replikativ.foerster.dist.HalfNormal
  (noise [_] (m/uniform01)) (push [d u] (dist/quantile d u)) (abduct [d x] (dist/cdf d x))
  org.replikativ.foerster.dist.Cauchy
  (noise [_] (m/uniform01)) (push [d u] (dist/quantile d u)) (abduct [d x] (dist/cdf d x))
  org.replikativ.foerster.dist.HalfCauchy
  (noise [_] (m/uniform01)) (push [d u] (dist/quantile d u)) (abduct [d x] (dist/cdf d x))
  org.replikativ.foerster.dist.Weibull
  (noise [_] (m/uniform01)) (push [d u] (dist/quantile d u)) (abduct [d x] (dist/cdf d x))
  org.replikativ.foerster.dist.Laplace
  (noise [_] (m/uniform01)) (push [d u] (dist/quantile d u)) (abduct [d x] (dist/cdf d x))
  org.replikativ.foerster.dist.Gumbel
  (noise [_] (m/uniform01)) (push [d u] (dist/quantile d u)) (abduct [d x] (dist/cdf d x)))

(extend-type org.replikativ.foerster.dist.Categorical
  PMechanism
  (noise [_] (m/uniform01))
  (push [d u]
    (nth (:values d) (push (dist/->Discrete (:weights d) (:total d)) u)))
  (abduct [d x]
    (let [i (first (keep-indexed #(when (= %2 x) %1) (:values d)))]
      (when (nil? i)
        (throw (ex-info "Value outside categorical support" {:type ::impossible-value :value x})))
      (abduct (dist/->Discrete (:weights d) (:total d)) i))))
