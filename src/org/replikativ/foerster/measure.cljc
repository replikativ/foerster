(ns org.replikativ.foerster.measure
  "Measure abstraction for probabilistic programming.

  Measures represent probability distributions over execution traces.
  This is the foundation for compositional inference algorithms."
  (:require [replikativ.logging :as log]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.foerster.random :as random]))

;; =============================================================================
;; Inference randomness
;; =============================================================================

(defn uniform01
  "One uniform draw from the current generator (`foerster.random`), so
   resampling, ancestor choice, the accept step and the site choice of every
   kernel obey `random/set-seed!` exactly like the program's samples do.
   `clojure.core/rand` did not: it reads `Math/random`'s own unseedable
   generator, which made a seeded run reproducible in its prior draws and
   random in its resampling and moves."
  []
  (random/uniform01))

(defn pick-uniformly
  "A uniformly chosen element of a vector, drawn through `uniform01`."
  [v]
  (nth v (int (* (uniform01) (count v)))))

;; =============================================================================
;; PMeasure Protocol
;; =============================================================================

(defprotocol PMeasure
  "Protocol for probability measures over execution traces.

  A measure represents a distribution over execution contexts (traces);
  inference resolves an EmpiricalMeasure, a weighted particle set."

  (measure-type [this]
    "Returns the type of measure: :empirical.")

  (sample-measure [this n]
    "Sample n execution contexts from this measure.

    Returns vector of [context log-weight] pairs, drawn according to the
    weights.")

  (log-marginal [this]
    "Estimate of log p(observations): the log mean particle weight plus the
    normalizer of earlier resampling steps.")

  (effective-sample-size [this]
    "Effective sample size (ESS) for particle degeneracy detection.

    ESS = (sum w_i)^2 / sum(w_i^2) where w_i are normalized weights.
    Low ESS indicates degeneracy -> trigger resampling.")

  (measure-stats [this query-fn]
    "Compute statistics over the measure using query-fn.

    query-fn: (fn [context] -> value)

    Returns map with :mean, :variance, :quantiles, etc.
    Useful for extracting posterior statistics."))

;; =============================================================================
;; Utility Functions
;; =============================================================================

(defn log-sum-exp
  "Numerically stable log-sum-exp.

  log(sum(exp(x_i))) = log-max + log(sum(exp(x_i - log-max)))"
  [log-weights]
  (when (empty? log-weights)
    (throw (ex-info "log-sum-exp requires non-empty sequence" {})))
  (let [log-max (apply max log-weights)]
    (if (= log-max ##-Inf)
      ##-Inf
      (+ log-max (Math/log (reduce + (map #(Math/exp (- % log-max)) log-weights)))))))

(defn log-mean-exp
  "log of the mean of exp(x_i)."
  [log-weights]
  (- (log-sum-exp log-weights) (Math/log (count log-weights))))

(defn sample-categorical
  "One index drawn from normalized weights, through `uniform01`."
  [weights]
  (let [u (uniform01)
        n (count weights)]
    (loop [i 0 cumsum 0.0]
      (if (>= i (dec n))
        (dec n)
        (let [cumsum' (+ cumsum (nth weights i))]
          (if (< u cumsum') i (recur (inc i) cumsum')))))))

(defn normalize-log-weights
  "Convert log-weights to normalized linear weights.

  Returns vector of weights summing to 1.0"
  [log-weights]
  (let [log-max (apply max log-weights)]
    (if (= log-max ##-Inf)
      (vec (repeat (count log-weights) (/ 1.0 (count log-weights))))
      (let [weights (mapv #(Math/exp (- % log-max)) log-weights)
            total (reduce + weights)]
        (mapv #(/ % total) weights)))))

(defn compute-ess
  "Compute effective sample size from normalized weights.

  ESS = 1 / sum(w_i^2) for normalized weights"
  [weights]
  (let [sum-squares (reduce + (map #(* % %) weights))]
    (/ 1.0 sum-squares)))

(defn weighted-quantiles
  "A function of p in [0, 1]: the p-quantile of `values` weighted by the
  normalized `weights`, the smallest value whose cumulative weight reaches p."
  [values weights]
  (let [pairs (sort-by first (map vector values weights))
        sorted (mapv first pairs)
        cumulative (vec (reductions + (map second pairs)))
        last-index (dec (count sorted))]
    (fn [p]
      (loop [i 0]
        (if (or (>= i last-index) (>= (nth cumulative i) p))
          (nth sorted i)
          (recur (inc i)))))))

(defn systematic-resample
  "Systematic resampling algorithm from Anglican.

  Low-variance resampling for particle filters.
  weights: normalized weights (sum to 1)
  n: number of particles to sample

  Returns vector of indices into original particle vector."
  [weights n]
  {:pre [(every? #(and (>= % 0) (<= % 1)) weights)
         (< (Math/abs (- (reduce + weights) 1.0)) 1e-6)]}
  (let [u (/ (uniform01) n)  ; Random offset, from the seeded generator
        cumsum (reductions + 0 weights)]
    (loop [i 0
           j 0
           indices []]
      (if (>= i n)
        indices
        (let [threshold (+ u (/ i n))]
          (if (< threshold (nth cumsum (inc j)))
            (recur (inc i) j (conj indices j))
            (recur i (inc j) indices)))))))

;; -----------------------------------------------------------------------------
;; Other resampling schemes (Douc, Cappé & Moulines 2005)
;; -----------------------------------------------------------------------------

(defn multinomial-resample
  "`n` indices drawn independently from the normalized `weights`."
  [weights n]
  (vec (repeatedly n #(sample-categorical weights))))

(defn stratified-resample
  "Stratified resampling: one uniform draw in each of the n strata [i/n,
  (i+1)/n). Lower variance than multinomial, never more than systematic's
  failure modes."
  [weights n]
  (let [cumsum (vec (reductions + weights))
        last-index (dec (count weights))]
    (loop [i 0 j 0 out []]
      (if (= i n)
        out
        (let [u (/ (+ i (uniform01)) n)
              j (loop [j j] (if (and (< j last-index) (< (nth cumsum j) u)) (recur (inc j)) j))]
          (recur (inc i) j (conj out j)))))))

(defn residual-resample
  "Residual resampling: ⌊n·wᵢ⌋ copies of each particle, the rest drawn
  multinomially from the residual weights."
  [weights n]
  (let [copies (mapv #(long (Math/floor (* n %))) weights)
        fixed (into [] (mapcat (fn [i c] (repeat c i)) (range) copies))
        left (- n (count fixed))]
    (if (zero? left)
      fixed
      (let [residual (mapv (fn [w c] (- (* n w) c)) weights copies)
            total (reduce + residual)]
        (into fixed (multinomial-resample (mapv #(/ % total) residual) left))))))

(defn resample
  "`n` ancestor indices from the normalized `weights` by `scheme`:
  `:systematic` (default), `:stratified`, `:residual` or `:multinomial`."
  [scheme weights n]
  (case (or scheme :systematic)
    :systematic (systematic-resample weights n)
    :stratified (stratified-resample weights n)
    :residual (residual-resample weights n)
    :multinomial (multinomial-resample weights n)
    (throw (ex-info "Unknown resampling scheme" {:type ::unknown-resampling :scheme scheme}))))

;; =============================================================================
;; EmpiricalMeasure - Weighted Particle Set
;; =============================================================================

(defrecord EmpiricalMeasure [particles]
  ;; particles: vector of [context log-weight] pairs

  PMeasure

  (measure-type [_]
    :empirical)

  (sample-measure [_ n]
    (let [log-weights (mapv second particles)
          weights (normalize-log-weights log-weights)
          ;; Systematic resampling (low-variance)
          indices (systematic-resample weights n)]
      (mapv #(nth particles %) indices)))

  (log-marginal [this]
    ;; the normalizer accumulated at earlier resampling steps (SMC attaches
    ;; it as :log-normalizer) plus the log MEAN current weight
    (+ (or (:log-normalizer this) 0.0)
       (log-mean-exp (mapv second particles))))

  (effective-sample-size [_]
    (let [log-weights (mapv second particles)
          weights (normalize-log-weights log-weights)]
      (compute-ess weights)))

  (measure-stats [_ query-fn]
    (let [log-weights (mapv second particles)
          weights (normalize-log-weights log-weights)
          values (mapv (fn [[ctx _]] (query-fn ctx)) particles)
          mean (reduce + (map * weights values))
          variance (reduce + (map (fn [w v] (* w (Math/pow (- v mean) 2)))
                                  weights values))
          quantile (weighted-quantiles values weights)]
      {:mean mean
       :variance variance
       :std-dev (Math/sqrt variance)
       :samples values
       :weights weights
       :quantiles {:p025 (quantile 0.025)
                   :p50 (quantile 0.5)
                   :p975 (quantile 0.975)}
       :type :empirical})))

(defn empirical
  "Create an empirical measure from weighted particles.

  particles: vector of [context log-weight] pairs"
  [particles]
  {:pre [(vector? particles)
         (every? (fn [[ctx lw]] (and (map? ctx) (number? lw))) particles)]}
  (->EmpiricalMeasure particles))

;; =============================================================================
;; Helper Functions for Particle Manipulation
;; =============================================================================

(defn get-contexts
  "Extract execution contexts from measure."
  [measure]
  (mapv first (:particles measure)))

(defn get-log-weights
  "Extract log-weights from measure."
  [measure]
  (mapv second (:particles measure)))

(defn get-particles
  "The particles of `measure`: a vector of [context log-weight] pairs."
  [measure]
  (:particles measure))

;; =============================================================================
;; Context Value Extraction
;; =============================================================================

(defrecord Sample [result trace])

(defn sample-particle
  "A particle that carries only a program result and its trace — what a
   pooled MCMC estimate keeps per draw, instead of pinning a context."
  [result trace]
  (->Sample result trace))

(defn get-value
  "Extract the program result from an execution context (or a Sample).

  The result is stored in [:inference :result] after execution completes.

  Args:
    context - ExecutionContext from inference

  Returns: The value returned by the probabilistic program"
  [context]
  (if (instance? Sample context)
    (:result context)
    (rtp/get-state context [:inference :result])))

(defn world-descriptors
  "The world descriptors of `measure`'s particles that ran in canonical worlds
  (`:world-policy :fork`), in particle order. Particles of pure inference ran
  in no such world, and contribute none."
  [measure]
  (into [] (keep #(if (instance? Sample %)
                    (:world-descriptor %)
                    (rtp/get-state % [:inference :world-descriptor])))
        (get-contexts measure)))

(defn get-trace
  "The trace {address -> entry} of a particle (context or Sample)."
  [context]
  (if (instance? Sample context)
    (:trace context)
    (rtp/get-state context [:inference :trace])))

(defn site-value
  "The value a particle's site `address` took: a sample, an observation or a
  `deterministic` value."
  [particle address]
  (:value (get (get-trace particle) address)))

;; =============================================================================
;; Print Methods (avoid StackOverflow from circular refs)
;; =============================================================================

#?(:clj
   (defmethod print-method EmpiricalMeasure [m ^java.io.Writer w]
     (.write w "#EmpiricalMeasure{")
     (.write w (str ":n " (count (:particles m))))
     (.write w (str ", :ess " (format "%.2f" (double (effective-sample-size m)))))
     (.write w (str ", :log-marginal " (format "%.4f" (double (log-marginal m)))))
     (.write w "}")))
