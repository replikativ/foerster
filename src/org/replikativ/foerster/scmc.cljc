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
            [org.replikativ.foerster.random :as random]))

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

(defn scmc
  "SCMC of the model given by
    :prior-sample        (fn []) θ ~ p₀, a vector
    :prior-log-density   (fn [θ]) log p₀(θ)
    :log-lik             (fn [θ datum]) log p(datum | θ)
    :batches             the data, a sequence of batches (sequences of data),
                         consumed lazily: a stream of batches runs in memory
                         bounded by the core-set and the population
  with options `:particles` (1000), `:memory` M (100), `:scale` (2.38, the
  random walk in population standard deviations), `:seed`.

  Returns {:particles [θ …] (equally weighted) :log-z (the evidence estimate,
  its increments exact, its moves approximate) :coreset [[w datum] …] (at
  most M) :steps :max-coreset-size (the largest core-set the moves used,
  before recompression: at most M plus a batch) :moves
  :accepted}. Throws `::zero-evidence` when a batch has zero likelihood
  under every particle."
  [{:keys [prior-sample prior-log-density log-lik batches]}
   & [{:keys [particles memory scale seed] :or {particles 1000 memory 100 scale 2.38}}]]
  (when-not (pos-int? particles)
    (throw (ex-info ":particles is a positive count" {:type ::invalid-particles :particles particles})))
  (when-not (pos-int? memory)
    (throw (ex-info ":memory is a positive count" {:type ::invalid-memory :memory memory})))
  (let [seed (or seed (random/fresh-seed))
        ;; consumed one batch at a time, never holding the sequence's head:
        ;; a stream of batches runs in bounded memory
        remaining (atom (seq batches))
        compressed? (atom false)
        stats (atom {:moves 0 :accepted 0})
        ;; [steps, the largest expanded core-set]: bounded, unlike a history
        sizes (atom [0 0])
        qs0 (mapv (fn [i] (random/with-stream* seed [::init i] #(vec (prior-sample)))) (range particles))
        path {:start {:t 0 :coreset []}
              :done? (fn [_] (nil? @remaining))
              :advance (fn [qs lw {:keys [t coreset]}]
                         ;; lw: the population's normalized log weights
                         (let [coreset (if (> (count coreset) memory)
                                         (do (reset! compressed? true)
                                             (recompress qs lw coreset log-lik memory))
                                         coreset)
                               batch (first @remaining)
                               _ (swap! remaining next)
                               expanded (into coreset (map (fn [x] [1.0 x])) batch)]
                           (swap! sizes (fn [[k top]] [(inc k) (max top (count expanded))]))
                           (let [incr (mapv (fn [q] (reduce + 0.0 (map #(log-lik q %) batch))) qs)]
                             (when (every? #(or (= ##-Inf %) (NaN? %)) incr)
                               (throw (ex-info "Every particle has zero likelihood for a batch: no posterior population"
                                               {:type ::zero-evidence :batch t})))
                             {:pos {:t (inc t) :coreset expanded} :log-incr incr})))
              :density (fn [{:keys [coreset]}]
                         (fn [q]
                           (let [lp (prior-log-density q)]
                             (if (= ##-Inf lp)
                               lp
                               (+ lp (reduce + 0.0 (map (fn [[w u]] (* w (log-lik q u))) coreset)))))))
              :exhausted (fn [pos] (ex-info "SCMC ran out of steps" {:type ::exhausted :t (:t pos)}))}
        {:keys [qs log-z positions]} (pop/theta-smc qs0 path {:n particles :scale scale :seed seed
                                                              :max-steps #?(:clj Long/MAX_VALUE :cljs js/Number.MAX_SAFE_INTEGER)
                                                              :positions? false
                                                              :stats stats
                                                              :keys {:resample ::resample :move ::move}})
        final (:coreset (peek positions))
        uniform (vec (repeat (count qs) (- (Math/log (count qs)))))]
    {:inference-contract {:semantics (if @compressed? :approximate-population :weighted-population)
                          :normalizer {:status :unverified}
                          :adaptation :population
                          :approximation (when @compressed? :coreset-rejuvenation)}
     :particles qs
     :log-z log-z
     :coreset (if (> (count final) memory) (recompress qs uniform final log-lik memory) final)
     :steps (first @sizes)
     :max-coreset-size (second @sizes)
     :moves (:moves @stats)
     :accepted (:accepted @stats)}))
