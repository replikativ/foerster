(ns org.replikativ.foerster.diagnostics
  "Checking an inference result.

  Markov chains: `kernel-infer` keeps each chain's draws together (the
  measure's `:chain-lengths`), so the draws of a quantity split back into
  chains (`chains`) and the convergence diagnostics of Vehtari, Gelman,
  Simpson, Carpenter & Bürkner (2021) apply — rank-normalized split R-hat,
  bulk and tail effective sample sizes, and the Monte Carlo standard error of
  the mean, computed as Stan and ArviZ compute them:

    (diagnostics/summary measure :mu)
    ;; => {:mean … :sd … :quantiles … :rhat 1.002 :ess-bulk 1830.0
    ;;     :ess-tail 1590.0 :mcse 0.004 :chains 4 :draws 4000}

  An R-hat above 1.01 or an effective sample size below about 100 per chain
  says the chains have not mixed: run longer or change the kernel.

  Particle methods: weights carry the information, so `summary` reports the
  weighted moments with the weight-based ESS; after resampling that counts
  particles, not distinct histories — `distinct-count` of an early site
  says how many histories survive.

  Model comparison: `pointwise-log-likelihood` gives each draw's log density
  of each observation, the input of PSIS-LOO and WAIC; particle methods also
  estimate the evidence (`measure/log-marginal`)."
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]))

;; =============================================================================
;; Draws by chain
;; =============================================================================

(defn- value-fn [f]
  (cond (keyword? f) (fn [p] (let [v (m/get-value p)] (if (map? v) (get v f) (m/site-value p f))))
        (vector? f) (fn [p] (m/site-value p f))
        :else (comp f m/get-value)))

(defn chains
  "The draws of `f` in `measure`, one vector per Markov chain, in draw order.
  `f` is a function of the program's value, a keyword (a field of a map value,
  else a site address) or a vector (a site address). A measure without chain
  structure (particle methods) is one chain."
  [measure f]
  (let [g (value-fn f)
        xs (mapv (comp double g first) (m/get-particles measure))
        lengths (or (:chain-lengths measure) [(count xs)])]
    (loop [xs xs [n & more] lengths out []]
      (if n (recur (subvec xs n) more (conj out (subvec xs 0 n))) out))))

;; =============================================================================
;; Vehtari et al. (2021)
;; =============================================================================

(defn- mean [xs] (/ (reduce + xs) (count xs)))

(defn- variance
  "The sample variance, n − 1 in the denominator."
  [xs]
  (let [mu (mean xs)]
    (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) (dec (count xs)))))

(defn- split-chains
  "Every chain cut in two halves of equal length (an odd middle draw
  dropped), all truncated to the shortest."
  [cs]
  (let [halves (mapcat (fn [c] (let [h (quot (count c) 2)]
                                 [(subvec c 0 h) (subvec c (- (count c) h))]))
                       cs)
        n (apply min (map count halves))]
    (mapv #(subvec % 0 n) halves)))

(defn- ranks
  "Average ranks (1-based) of the pooled draws, ties shared."
  [xs]
  (let [order (vec (sort-by #(nth xs %) (range (count xs))))
        r (double-array (count xs))]
    (loop [i 0]
      (when (< i (count order))
        (let [v (nth xs (nth order i))
              j (loop [j i] (if (and (< (inc j) (count order)) (= v (nth xs (nth order (inc j))))) (recur (inc j)) j))
              avg (/ (+ i j 2) 2.0)]
          (doseq [k (range i (inc j))] (aset r (int (nth order k)) avg))
          (recur (inc j)))))
    (vec r)))

(defn- rank-normalize
  "z-scores of the pooled ranks (Blom's offset), chain shape kept."
  [cs]
  (let [pooled (vec (apply concat cs))
        s (count pooled)
        z (mapv #(dist/normal-quantile (/ (- % 0.375) (+ s 0.25))) (ranks pooled))]
    (loop [z z [c & more] cs out []]
      (if c (recur (subvec z (count c)) more (conj out (subvec z 0 (count c)))) out))))

(defn- median [xs]
  (let [s (vec (sort xs)) n (count s)]
    (if (odd? n) (nth s (quot n 2)) (* 0.5 (+ (nth s (dec (quot n 2))) (nth s (quot n 2)))))))

(defn- quantile
  "The `p` quantile of `xs`, interpolated linearly between order statistics
  (numpy's default)."
  [xs p]
  (let [s (vec (sort xs)) h (* p (dec (count s))) lo (long (Math/floor h))]
    (if (>= lo (dec (count s)))
      (peek s)
      (+ (nth s lo) (* (- h lo) (- (nth s (inc lo)) (nth s lo)))))))

(defn- rhat* [cs]
  (let [n (count (first cs))
        w (mean (map variance cs))
        b (* n (variance (map mean cs)))
        var+ (+ (* (/ (dec n) n) w) (/ b n))]
    (if (pos? w) (Math/sqrt (/ var+ w)) ##NaN)))

(defn- autocovariance
  "(fn [t]) → the autocovariance of `xs` at lag t, normalized by n: computed
  per lag when asked, as Geyer's sum stops long before the last lag."
  [xs]
  (let [n (count xs)
        mu (mean xs)
        d (double-array (map #(- % mu) xs))]
    (memoize
     (fn [t]
       (/ (loop [i 0 acc 0.0]
            (if (< (+ i t) n) (recur (inc i) (+ acc (* (aget d i) (aget d (+ i t))))) acc))
          n)))))

(defn- ess*
  "Effective sample size of chains of equal length (ArviZ's `_ess`): the
  combined autocorrelation summed by Geyer's initial positive, then initial
  monotone sequence."
  [cs]
  (let [m (count cs) n (count (first cs))]
    (if (< n 4)
      ##NaN
      (let [acovs (mapv autocovariance cs)
            mean-acov (fn [t] (mean (map #(% t) acovs)))
            mean-var (* (mean-acov 0) (/ n (dec n)))
            var+ (+ (* mean-var (/ (dec n) n))
                    (if (> m 1) (variance (map mean cs)) 0.0))
            rho (fn [t] (- 1.0 (/ (- mean-var (mean-acov t)) var+)))]
        (if-not (pos? var+)
          ##NaN                              ; constant draws: no information
          (let [r (double-array n)
                _ (aset r 0 1.0)
                _ (aset r 1 (rho 1))
            ;; initial positive sequence
                [t even] (loop [t 1 even 1.0 odd (aget r 1)]
                           (if (and (< t (- n 3)) (> (+ even odd) 0.0))
                             (let [even' (rho (inc t)) odd' (rho (+ t 2))]
                               (when (>= (+ even' odd') 0.0)
                                 (aset r (inc t) even')
                                 (aset r (+ t 2) odd'))
                               (recur (+ t 2) even' odd'))
                             [t even]))
                max-t (- t 2)
                _ (when (> even 0.0) (aset r (inc max-t) even))
            ;; initial monotone sequence
                _ (loop [t 1]
                    (when (<= t (- max-t 2))
                      (when (> (+ (aget r (inc t)) (aget r (+ t 2)))
                               (+ (aget r (dec t)) (aget r t)))
                        (let [v (/ (+ (aget r (dec t)) (aget r t)) 2.0)]
                          (aset r (inc t) v)
                          (aset r (+ t 2) v)))
                      (recur (+ t 2))))
                tau (+ -1.0
                       (* 2.0 (reduce + (map #(aget r %) (range (inc max-t)))))
                       (aget r (inc max-t)))
                tau (max tau (/ 1.0 (Math/log10 (* m n))))]
            (/ (* m n) tau)))))))

(defn rhat
  "Rank-normalized split R-hat of `cs` (draws by chain): the larger of the
  bulk and the folded (tail) statistic. Near 1 when the chains agree; above
  1.01 they have not mixed."
  [cs]
  (let [split (split-chains cs)
        z (rank-normalize split)
        med (median (apply concat split))
        folded (rank-normalize (mapv (fn [c] (mapv #(Math/abs (- % med)) c)) split))]
    (max (rhat* z) (rhat* folded))))

(defn ess-bulk
  "The bulk effective sample size of `cs` (draws by chain): of the
  rank-normalized split chains."
  [cs]
  (ess* (rank-normalize (split-chains cs))))

(defn ess-tail
  "The tail effective sample size of `cs`: the smaller of the effective sample
  sizes of the indicators of the 5% and the 95% quantile."
  [cs]
  (let [pooled (apply concat cs)
        ess-of (fn [c] (ess* (split-chains (mapv (fn [chain] (mapv #(if (<= % c) 1.0 0.0) chain)) cs))))]
    (min (ess-of (quantile pooled 0.05)) (ess-of (quantile pooled 0.95)))))

(defn mcse
  "Monte Carlo standard error of the mean of `cs`: the sd of all draws over
  √ESS, the ESS of the split chains themselves."
  [cs]
  (/ (Math/sqrt (variance (vec (apply concat cs)))) (Math/sqrt (ess* (split-chains cs)))))

(defn summary
  "Summary of the quantity `f` (see `chains`) in `measure`: weighted mean, sd
  and quantiles, and — for Markov chains — `:rhat`, `:ess-bulk`, `:ess-tail`
  and `:mcse`; for weighted particles `:ess`, the weight-based effective
  sample size."
  [measure f]
  (let [g (value-fn f)
        {:keys [mean std-dev quantiles]} (m/measure-stats measure g)
        cs (chains measure f)
        base {:mean mean :sd std-dev :quantiles quantiles
              :chains (count cs) :draws (reduce + (map count cs))}]
    (if (:chain-lengths measure)
      (assoc base :rhat (rhat cs) :ess-bulk (ess-bulk cs) :ess-tail (ess-tail cs) :mcse (mcse cs))
      (assoc base :ess (m/effective-sample-size measure)))))

(defn distinct-count
  "How many distinct values `f` (see `chains`) takes over the particles of
  `measure`: for an early site after resampling, the number of histories
  that survive."
  [measure f]
  (let [g (value-fn f)]
    (count (into #{} (map (comp g first)) (m/get-particles measure)))))

;; =============================================================================
;; Pointwise log-likelihood
;; =============================================================================

(defn pointwise-log-likelihood
  "Each draw's log density of each observation: a vector, one map
  {address log-p} per particle of `measure`, in particle order — the input of
  PSIS-LOO and WAIC (with every draw equally weighted, as Markov chains give;
  resample a weighted measure first)."
  [measure]
  (mapv (fn [[particle _]]
          (into {} (keep (fn [[address {:keys [observed? log-prob]}]]
                           (when observed? [address log-prob])))
                (m/get-trace particle)))
        (m/get-particles measure)))
