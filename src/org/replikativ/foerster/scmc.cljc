(ns org.replikativ.foerster.scmc
  "Sequential core-set Monte Carlo (Beronov, Weilbach, Wood & Campbell, UAI
  2021): SMC over data arriving in batches whose rejuvenation moves cost a
  bounded amount however much data has arrived.

  The target after t batches is p_t(θ) ∝ p₀(θ) · Π_{s≤t} Π_b p(x_b⁽ˢ⁾ | θ).
  Plain resample-move SMC (IBIS; Chopin 2002) must score every past datum in
  each move, so step t costs O(t). SCMC keeps at most `:memory` M past data
  points with weights — a core-set — whose weighted log-likelihood
  approximates that of all data so far, and moves the particles against

    log p₀(θ) + Σ_j w_j log p(u_j | θ).

  After each step's moves, when the core-set and the new batch exceed M, it
  is recompressed: a sparse nonnegative least-squares fit (GIGA; Campbell &
  Broderick 2018) of the core-set's log-likelihood functions, centred and
  evaluated at the current particles, to their weighted sum. The fit is best
  where the population has its mass.

  The reweighting by each new batch is exact. Only the moves target the
  core-set approximation, so the particles carry its error: small when the
  log-likelihoods compress well (exchangeable data, few parameters), larger
  for a small `:memory`. With M at least the amount of data, SCMC is
  resample-move SMC.

  The engine is `foerster.population/theta-smc` on θ vectors: a step
  resamples, then moves every particle by population-scaled random-walk
  Metropolis."
  (:require [org.replikativ.foerster.population :as pop]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.weights :as weights]))

;; =============================================================================
;; GIGA
;; =============================================================================

(defn- dot ^double [^doubles a ^doubles b]
  (let [n (alength a)]
    (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (* (aget a i) (aget b i)))) s))))

(defn- norm ^double [a] (Math/sqrt (dot a a)))

(defn- lin
  "a·x + b·y, a fresh array."
  ^doubles [a ^doubles x b ^doubles y]
  (let [n (alength x) out (double-array n)]
    (dotimes [i n] (aset out i (+ (* (double a) (aget x i)) (* (double b) (aget y i)))))
    out))

(defn giga
  "Greedy iterative geodesic ascent (Campbell & Broderick 2018, Alg. 1):
  nonnegative weights w over the vectors `ls` (double arrays of one length),
  at most `m` of them nonzero, with Σ w_n·ls_n close to `target`.

  It works on the unit sphere: each iteration adds the vector whose geodesic
  from the current direction best aligns with the geodesic to the target's
  direction, with the step that maximizes the alignment; the weights are
  scaled optimally at the end. A vector of norm 0 gets weight 0. Returns the
  weights as a vector."
  [ls target m]
  (let [n (count ls)
        norms (mapv norm ls)
        tn (norm target)]
    (if (or (zero? n) (zero? tn))
      (vec (repeat n 0.0))
      (let [ell (lin (/ 1.0 tn) target 0.0 target)
            ells (mapv (fn [l nl] (when (pos? nl) (lin (/ 1.0 nl) l 0.0 l))) ls norms)
            ;; the unit vector along the geodesic from u to v, nil if none
            towards (fn [^doubles v ^doubles u]
                      (let [d (lin 1.0 v (- (dot v u)) u) dn (norm d)]
                        (when (> dn 1e-12) (lin (/ 1.0 dn) d 0.0 d))))
            finish (fn [w ^doubles lw]
                     (let [align (max 0.0 (dot lw ell))]
                       (mapv (fn [wj nj] (if (pos? nj) (* wj (/ tn nj) align) 0.0)) w norms)))]
        (loop [t 0 w (vec (repeat n 0.0)) lw (double-array (alength ^doubles ell))]
          (let [d (when (< t m) (towards ell lw))
                [j score] (when d
                            (reduce (fn [[_ best :as acc] j]
                                      (let [dj (some-> (ells j) (towards lw))
                                            s (if dj (dot d dj) ##-Inf)]
                                        (if (> s best) [j s] acc)))
                                    [nil ##-Inf] (range n)))]
            (if (or (nil? j) (= ##-Inf score))
              (finish w lw)
              (let [ej (ells j)
                    z0 (dot ell ej) z1 (dot ell lw) z2 (dot ej lw)
                    num (- z0 (* z1 z2))
                    den (+ num (- z1 (* z0 z2)))
                    gamma (if (pos? den) (min 1.0 (max 0.0 (/ num den))) 0.0)
                    lw' (lin (- 1.0 gamma) lw gamma ej)
                    nn (norm lw')]
                (if (or (zero? gamma) (zero? nn))
                  (finish w lw)
                  (recur (inc t)
                         (update (mapv #(/ (* (- 1.0 gamma) %) nn) w) j + (/ gamma nn))
                         (lin (/ 1.0 nn) lw' 0.0 lw')))))))))))

;; =============================================================================
;; The core-set projection
;; =============================================================================

(defn- recompress
  "The core-set [[w datum] …] compressed to at most `m` points: GIGA on the
  data's log-likelihoods at the particles `qs` (normalized log weights
  `lw`), each centred by its population mean and scaled by √π_k, towards
  the core-set's own weighted sum."
  [qs lw coreset log-lik m]
  (let [pis (mapv #(Math/exp %) lw)
        roots (mapv #(Math/sqrt %) pis)
        cols (mapv (fn [[_ u]]
                     (let [f (mapv #(double (log-lik % u)) qs)
                           mean (reduce + (map * pis f))]
                       (double-array (map (fn [fk r] (* r (- fk mean))) f roots))))
                   coreset)
        target (reduce (fn [^doubles acc [^doubles col [w _]]] (lin 1.0 acc (double w) col))
                       (double-array (count qs))
                       (map vector cols coreset))
        w (giga cols target m)]
    (into [] (keep (fn [[wj [_ u]]] (when (pos? wj) [wj u]))) (map vector w coreset))))

(defn start
  "Initialize resumable numeric SCMC. model provides :prior-sample,
  :prior-log-density and :log-lik. State is a persistent in-memory value,
  including model functions; it is not a serialized inference artifact.
  opts: :particles (1000), :memory (100), :scale (2.38), :seed."
  [{:keys [prior-sample] :as model}
   & [{:keys [particles memory scale seed]
       :or {particles 1000 memory 100 scale 2.38}}]]
  (when-not (pos-int? particles)
    (throw (ex-info ":particles is a positive count" {:type ::invalid-particles :particles particles})))
  (when-not (pos-int? memory)
    (throw (ex-info ":memory is a positive count" {:type ::invalid-memory :memory memory})))
  (let [seed (or seed (random/fresh-seed))]
    {:model (select-keys model [:prior-sample :prior-log-density :log-lik])
     :particles (mapv (fn [i] (random/with-stream* seed [::init i] #(vec (prior-sample))))
                      (range particles))
     :memory memory :scale scale :seed seed :coreset [] :steps 0
     :compressed? false :approximate? false :log-normalizer 0.0
     :max-coreset-size 0 :moves 0 :accepted 0}))

(defn append-batch
  "Advance a persistent SCMC state by one non-empty batch. The old state
  remains reusable. Retains only the population and a bounded coreset; the
  caller owns any observation history needed for replay or correction.
  :last-update reports ESS before resampling, not convergence or accuracy.
  Historical replacement/retraction is not an append operation."
  [{:keys [particles memory scale seed coreset steps model compressed?] :as state} batch]
  (let [batch (vec batch)]
    (when (empty? batch)
      (throw (ex-info "An SCMC update requires a non-empty batch" {:type ::empty-batch})))
    (let [{:keys [prior-log-density log-lik]} model
          n (count particles)
          expanded (into coreset (map (fn [x] [1.0 x])) batch)
          increments (mapv (fn [q] (reduce + 0.0 (map #(log-lik q %) batch))) particles)
          mass (weights/from-log increments)
          _ (when (= :zero-mass (:status mass))
              (throw (ex-info "Every particle has zero likelihood for a batch: no posterior population"
                              {:type ::zero-evidence :batch steps})))
          ess (m/compute-ess (weights/selection mass))
          stats (atom {:moves 0 :accepted 0})
          path {:start 0 :done? pos?
                :advance (fn [_ _ _] {:pos 1 :log-incr increments})
                :density (fn [_]
                           (fn [q]
                             (let [lp (prior-log-density q)]
                               (if (= ##-Inf lp) lp
                                   (+ lp (reduce + 0.0
                                                 (map (fn [[w u]] (* w (log-lik q u))) expanded)))))))
                :exhausted (fn [_] (ex-info "SCMC single update exhausted" {:type ::exhausted}))}
          {:keys [qs log-z]} (pop/theta-smc particles path
                                            {:n n :scale scale :seed seed :max-steps 1 :step-offset steps
                                             :positions? false :stats stats
                                             :keys {:resample ::resample :move ::move}})
          compress? (> (count expanded) memory)
          coreset (if compress?
                    (recompress qs (vec (repeat n (- (Math/log n)))) expanded log-lik memory)
                    expanded)]
      (-> state
          (assoc :particles qs :coreset coreset :steps (inc steps)
                 :compressed? (or compressed? compress?)
                 :approximate? (or (:approximate? state) compressed?)
                 :last-update {:ess ess :ess-fraction (/ ess n) :batch-size (count batch)
                               :compressed? compress?})
          (update :log-normalizer + log-z)
          (update :max-coreset-size max (count expanded))
          (update :moves + (:moves @stats))
          (update :accepted + (:accepted @stats))))))

(defn result
  "Project an SCMC snapshot to the common settled measure API. Projection
  does not consume the snapshot. Coreset moves and population adaptation do
  not certify an unbiased normalizer or posterior convergence."
  [{:keys [particles approximate?] :as state}]
  (merge (m/empirical (mapv (fn [q] [(m/sample-particle q {}) 0.0]) particles))
         (select-keys state [:coreset :steps :max-coreset-size :moves :accepted
                             :log-normalizer :last-update])
         {:inference-contract {:semantics (if approximate? :approximate-population :weighted-population)
                               :normalizer {:status :unverified}
                               :adaptation :population
                               :approximation (when approximate? :coreset-rejuvenation)}}))

(defn scmc
  "Run numeric SCMC over :batches using start, append-batch and result.
  model: :prior-sample (fn [] theta-vector), :prior-log-density (fn [theta]),
  :log-lik (fn [theta datum]), :batches (lazy sequence of non-empty batches).
  opts: :particles (1000), :memory (100), :scale (2.38), :seed.
  Returns an EmpiricalMeasure with :coreset, :steps, :max-coreset-size,
  :moves, :accepted, :last-update and :log-normalizer. Streaming memory is
  bounded by population, coreset and batch; no history of updates is retained.
  To pause between batches, retain the state returned by append-batch."
  [{:keys [batches] :as model} & [opts]]
  (result (reduce append-batch (start model opts) batches)))
