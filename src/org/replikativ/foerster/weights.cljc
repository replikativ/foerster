(ns org.replikativ.foerster.weights
  "Shared population mass accounting. Normalized selection weights and total
  mass are separate; neither asserts posterior accuracy or unbiased evidence.
  A zero-mass population has no posterior and cannot be drawn or resampled.

  Two drivers keep the ledger: the population-path interpreter
  (`population/path`: tempered SMC, θ-vector populations, population
  programs) holds a mass state (`from-log`, `reweight`, `selected`,
  `resampled`); the savepoint particle drivers (`foerster.inference.smc`: SMC,
  importance sampling, streaming, batched and conditional SMC, PGAS, and
  through them PIMH, particle Gibbs and IPMCMC), whose raw weights live in
  their particles' worlds, account each barrier with `barrier`. Both fold
  the same log mass: the normalizer before plus the log mean weight.

  Drawing from weights is here too: `sample-categorical` and the resampling
  schemes (`resample`), which read the current generator (`foerster.random`)."
  (:require [org.replikativ.foerster.random :as random]))

(defn log-sum-exp [xs]
  (when-not (seq xs)
    (throw (ex-info "log-sum-exp requires non-empty sequence" {:type ::empty-population})))
  (when (some #(or (not (number? %)) (#?(:clj Double/isNaN :cljs js/isNaN) %)) xs)
    (throw (ex-info "A log weight is NaN or not numeric" {:type :org.replikativ.foerster.measure/nan-weight})))
  (let [top (reduce max ##-Inf xs)]
    (if (or (= top ##-Inf) (= top ##Inf))
      top
      (+ top (Math/log (reduce + (map #(Math/exp (- % top)) xs)))))))

(defn log-mean-exp [xs]
  (- (log-sum-exp xs) (Math/log (count xs))))

(defn normalize
  "Linear selection weights. Zero or infinite total mass requires an explicit
  :zero/:infinite :uniform policy (legacy selection inside existing engines)."
  ([xs] (normalize xs {}))
  ([xs {:keys [zero infinite]}]
   (let [z (log-sum-exp xs)]
     (cond
       (= z ##-Inf)
       (if (= :uniform zero)
         (vec (repeat (count xs) (/ 1.0 (count xs))))
         (throw (ex-info "A zero-mass population has no selection distribution" {:type ::zero-mass})))
       (= z ##Inf)
       (if (= :uniform infinite)
         (let [n (count (filter #(= ##Inf %) xs))]
           (mapv #(if (= % ##Inf) (/ 1.0 n) 0.0) xs))
         (throw (ex-info "Infinite population mass has no defined normalization" {:type ::infinite-mass})))
       :else
       ;; Subtract the maximum, then divide: preserve legacy seeded selection
       ;; rounding as well as stability for large finite scores.
       (let [top (reduce max xs)
             ws (mapv #(Math/exp (- % top)) xs)
             total (reduce + ws)]
         (mapv #(/ % total) ws))))))

(defn ess
  "The effective sample size 1/Σw² of normalized selection weights `ws`."
  [ws]
  (/ 1.0 (reduce + (map #(* % %) ws))))

(defn barrier
  "The accounting of a savepoint particle population at a barrier: `xs` its
  raw log weights since its last resampling, `log-normalizer` the log mass
  folded in before them. Returns

    :selection  normalized selection weights (`normalize`; uniform among the
                impossible or the infinite, so that a degenerate population
                can still be resampled)
    :ess        `ess` of the selection
    :log-mean   the barrier's factor of the evidence, log mean exp of `xs`
    :log-mass   the population's log mass, `log-normalizer` + `:log-mean`
                (`from-log`'s), which a resampling folds into the
                normalizer as its children restart at equal weight
    :reset      the log weight the children restart from: 0, or −∞ when the
                whole population is impossible (it stays impossible)"
  [xs log-normalizer]
  (let [selection (normalize xs {:zero :uniform :infinite :uniform})
        log-mean (log-mean-exp xs)]
    {:selection selection
     :ess (ess selection)
     :log-mean log-mean
     :log-mass (+ log-normalizer log-mean)
     :reset (if (= ##-Inf (reduce max xs)) ##-Inf 0.0)}))

(defn from-log
  "Canonical accounting from raw log weights and an accumulated log normalizer.
  log-mass is log-normalizer + logmeanexp(raw). A zero-mass state stays explicit."
  ([xs] (from-log xs 0.0))
  ([xs log-normalizer]
   (let [xs (vec xs)
         z (log-sum-exp xs)]
     (when (or (= ##Inf z) (not (number? log-normalizer))
               (#?(:clj Double/isNaN :cljs js/isNaN) log-normalizer)
               (= ##Inf log-normalizer))
       (throw (ex-info "Population mass must be finite or zero" {:type ::invalid-mass})))
     (if (or (= ##-Inf z) (= ##-Inf log-normalizer))
       {:log-weights (vec (repeat (count xs) ##-Inf)) :log-mass ##-Inf :status :zero-mass}
       {:log-weights (mapv #(- % z) xs)
        :log-mass (+ log-normalizer (- z (Math/log (count xs))))
        :status :weighted}))))

(defn uniform
  "Equal selection weights at the given total log mass."
  ([n] (uniform n 0.0))
  ([n log-mass]
   (when-not (pos-int? n)
     (throw (ex-info "A population needs positive size" {:type ::empty-population})))
   (from-log (vec (repeat n 0.0)) log-mass)))

(defn reweight [state increments]
  (when-not (= (count increments) (count (:log-weights state)))
    (throw (ex-info "One correction is required per particle" {:type ::weight-count-mismatch})))
  (let [next (mapv + (:log-weights state) increments)
        z (log-sum-exp next)]
    (when (= ##Inf z)
      (throw (ex-info "Infinite incremental population mass" {:type ::infinite-mass})))
    (if (= ##-Inf z)
      (uniform (count next) ##-Inf)
      {:log-weights (mapv #(- % z) next)
       :log-mass (+ (:log-mass state) z)
       :status :weighted})))

(defn selection [state]
  (normalize (:log-weights state)))

(defn resampled [state n]
  (when (= :zero-mass (:status state))
    (throw (ex-info "Cannot resample a zero-mass population" {:type ::zero-mass})))
  (uniform n (:log-mass state)))

;; =============================================================================
;; Drawing from weights
;; =============================================================================

(defn- cumulative
  "The running sums of `weights`, as a vector."
  [weights]
  (vec (reductions + weights)))

(defn- search-cumulative
  "The first index whose cumulative weight exceeds `u`, by bisection; the
  last index when rounding leaves `u` above every sum."
  [cumsum u]
  (loop [lo 0 hi (dec (count cumsum))]
    (if (>= lo hi)
      lo
      (let [mid (quot (+ lo hi) 2)]
        (if (< u (nth cumsum mid)) (recur lo mid) (recur (inc mid) hi))))))

(defn sample-categorical
  "One index drawn from normalized weights, through `random/uniform01`."
  [weights]
  (random/decide (constantly weights) #(search-cumulative (cumulative weights) (random/uniform01))))

(defn normalize-log-weights
  "Legacy engine selection: equal mass among impossible candidates, or among
  infinite-weight candidates. This does not make a zero-mass posterior valid."
  [log-weights]
  (normalize log-weights {:zero :uniform :infinite :uniform}))

(defn systematic-resample
  "Systematic resampling algorithm from Anglican.

  Low-variance resampling for particle filters.
  weights: normalized weights (sum to 1)
  n: number of particles to sample

  Returns vector of indices into original particle vector."
  [weights n]
  {:pre [(every? #(and (>= % 0) (<= % 1)) weights)
         (< (Math/abs (- (reduce + weights) 1.0)) 1e-6)]}
  (let [u (/ (random/uniform01) n)  ; Random offset, from the seeded generator
        cumsum (cumulative weights)
        last-index (dec (count weights))]
    (loop [i 0
           j 0
           indices []]
      (if (>= i n)
        indices
        (let [threshold (+ u (/ i n))]
          ;; the last index takes whatever rounding leaves above the sum
          (if (or (= j last-index) (< threshold (nth cumsum j)))
            (recur (inc i) j (conj indices j))
            (recur i (inc j) indices)))))))

;; -----------------------------------------------------------------------------
;; Other resampling schemes (Douc, Cappé & Moulines 2005)
;; -----------------------------------------------------------------------------

(defn multinomial-resample
  "`n` indices drawn independently from the normalized `weights`."
  [weights n]
  (let [cumsum (cumulative weights)]
    (vec (repeatedly n (fn [] (random/decide (constantly weights)
                                             #(search-cumulative cumsum (random/uniform01))))))))

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
        (let [u (/ (+ i (random/uniform01)) n)
              j (loop [j j] (if (and (< j last-index) (< (nth cumsum j) u)) (recur (inc j)) j))]
          (recur (inc i) j (conj out j)))))))

(defn residual-resample
  "Residual resampling: ⌊n·wᵢ⌋ copies of each particle, the rest drawn
  multinomially from the residual weights."
  [weights n]
  ;; n·(1/n) may round to just below 1: the whole part is taken with a
  ;; margin, or equal weights would fall back to multinomial draws
  (let [copies (mapv #(long (Math/floor (+ (* n %) 1e-9))) weights)
        fixed (into [] (mapcat (fn [i c] (repeat c i)) (range) copies))
        left (- n (count fixed))]
    (if (zero? left)
      fixed
      (let [residual (mapv (fn [w c] (max 0.0 (- (* n w) c))) weights copies)
            total (reduce + residual)]
        (into fixed (if (pos? total)
                      (multinomial-resample (mapv #(/ % total) residual) left)
                      (systematic-resample weights left)))))))

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
