(ns org.replikativ.foerster.measure
  "Measure abstraction for probabilistic programming.

  Measures represent probability distributions over execution traces.
  This is the foundation for compositional inference algorithms."
  (:require [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.contract :as contract]
            [org.replikativ.spindel.engine.protocols :as rtp]))

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
    "Log population mass: log mean weight plus earlier normalizers. Returns nil
    for explicitly marked chain outputs, which do not estimate evidence. A
    numerical value alone does not certify its scale or unbiasedness.")

  (effective-sample-size [this]
    "Effective sample size (ESS) for particle degeneracy detection.

    ESS = (sum w_i)^2 / sum(w_i^2) where w_i are normalized weights.
    Low ESS indicates degeneracy -> trigger resampling. For Markov-chain
    draws (equal weights) this is the number of draws, not the chain ESS:
    use `foerster.diagnostics/summary`.")

  (measure-stats [this query-fn]
    "Compute statistics over the measure using query-fn.

    query-fn: (fn [context] -> value)

    Returns map with :mean, :variance, :quantiles, etc.
    Useful for extracting posterior statistics."))

;; =============================================================================
;; Utility Functions
;; =============================================================================

(defn- possible!
  "Throw unless some weight is positive: a measure whose particles all have
  probability zero (evidence zero) has no posterior to report or draw from."
  [log-weights]
  (when (every? #(= ##-Inf %) log-weights)
    (throw (ex-info "Every particle has probability zero: the measure is impossible"
                    {:type ::impossible}))))

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
          _ (possible! log-weights)
          weights (weights/normalize-log-weights log-weights)
          ;; Systematic resampling (low-variance)
          indices (weights/systematic-resample weights n)]
      (mapv #(nth particles %) indices)))

  (log-marginal [this]
    ;; the normalizer accumulated at earlier resampling steps (SMC attaches
    ;; it as :log-normalizer) plus the log MEAN current weight
    (case (contract/normalizer-status this)
      :unavailable nil
      ;; a chain's separate evidence estimate: the draws' weights combine
      ;; samples and carry no evidence
      :estimate (:log-normalizer this)
      (+ (or (:log-normalizer this) 0.0)
         (weights/log-mean-exp (mapv second particles)))))

  (effective-sample-size [_]
    (let [log-weights (mapv second particles)
          weights (weights/normalize-log-weights log-weights)]
      (weights/ess weights)))

  (measure-stats [_ query-fn]
    (let [log-weights (mapv second particles)
          _ (possible! log-weights)
          weights (weights/normalize-log-weights log-weights)
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
         (seq particles)
         (every? (fn [[ctx lw]] (and (map? ctx) (number? lw) (not (NaN? lw)))) particles)]}
  (assoc (->EmpiricalMeasure particles)
         :inference-contract (contract/weighted-population)))

(defn markov-measure
  "Settled chain draws. Their weights combine samples, not evidence estimates,
  so `log-marginal` is nil. A chain method that computes a separate evidence
  estimate (parallel tempering's stepping stones) passes it as `normalizer`
  {:estimator e :log-normalizer l}: the contract then names that estimator,
  status `:estimate`, and `log-marginal` returns it. Either way the draws are
  autocorrelated: their weight-based `effective-sample-size` is not the chain
  ESS (see `foerster.diagnostics/summary`, which uses `:chain-lengths`)."
  ([particles] (markov-measure particles nil))
  ([particles {:keys [estimator log-normalizer]}]
   (cond-> (assoc (empirical particles) :inference-contract
                  (contract/markov-chain (if estimator {:estimator estimator} {})))
     estimator (assoc :log-normalizer log-normalizer))))

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
     (.write w (str ", :log-marginal " (if-some [z (log-marginal m)]
                                         (format "%.4f" (double z)) "unavailable")))
     (.write w "}")))
