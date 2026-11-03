(ns org.replikativ.foerster.inference.scmc
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
  evaluated at the current particles, to their weighted sum. GIGA fixes the
  weights' proportions greedily and fits only their common scale; its
  weights are then refitted by nonnegative least squares on the points it
  selected, the fully corrective update of Campbell & Beronov (NeurIPS
  2019) (`:refit :nnls`, the default; `:refit :none` keeps GIGA's weights,
  as in the UAI 2021 algorithm). The fit is best where the population has
  its mass.

  `:projection-inflation k` is a research option and a deviation from the
  UAI 2021 algorithm, off by default: the projection is fitted at the
  particles spread by the factor k about their mean, μ + k·(θ − μ), rather
  than at the particles themselves, so the core-set's log-likelihood is
  matched over a wider region than the current population (the posterior
  can move outside it as data arrive). It costs N·|E| likelihood
  evaluations per compression, and the likelihood must be finite at the
  spread points. k = 1, or no factor, is the UAI 2021 projection.

  Each new batch reweights the particles by its own exact likelihood, but
  the particles it reweights are not distributed as the previous target:
  they were moved against the previous step's core-set before that step
  compressed it (the core-set plus that step's batch), while the posterior
  they stand for from then on uses the compressed core-set. No importance
  correction exp(Σ_C' w'·ℓ − Σ_C w·ℓ) is applied for that switch, as in
  the paper's Algorithm 1; the next moves target the compressed core-set.
  So the particles carry the core-set's approximation error: small when the
  log-likelihoods compress well (exchangeable data, few parameters), larger
  for a small `:memory`. With M at least the amount of data, SCMC is
  resample-move SMC and nothing is approximated.

  The engine is `foerster.population.theta/theta-smc` on θ vectors: a step
  resamples, then moves every particle by population-scaled random-walk
  Metropolis."
  (:require [org.replikativ.foerster.population.theta :as pop]
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

(defn- lin!
  "a·x + b·y into `out` (which may be x or y), computed as `lin` does."
  ^doubles [^doubles out a ^doubles x b ^doubles y]
  (let [a (double a) b (double b)]
    (dotimes [i (alength out)] (aset out i (+ (* a (aget x i)) (* b (aget y i)))))
    out))

(defn- towards!
  "The unit vector along the geodesic from the unit vector u to v, into
  `out`; nil if there is none."
  [^doubles out ^doubles v ^doubles u]
  (let [d (lin! out 1.0 v (- (dot v u)) u) dn (norm d)]
    (when (> dn 1e-12) (lin! out (/ 1.0 dn) d 0.0 d))))

(defn- towards-score
  "⟨d, towards(v, u)⟩ without building towards(v, u): ##-Inf if there is no
  geodesic from u to v. Computes the same arithmetic as `towards!`."
  ^double [^doubles d ^doubles v ^doubles u]
  (let [n (alength v)
        c (- (dot v u))
        dd (fn ^double [^long i] (+ (* 1.0 (aget v i)) (* c (aget u i))))
        dn (Math/sqrt (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (let [x (dd i)] (+ s (* x x)))) s)))]
    (if (> dn 1e-12)
      (let [inv (/ 1.0 dn)]
        (loop [i 0 s 0.0]
          (if (< i n)
            (recur (inc i) (+ s (* (aget d i) (let [x (dd i)] (+ (* inv x) (* 0.0 x))))))
            s)))
      ##-Inf)))

(defn giga
  "Greedy iterative geodesic ascent (Campbell & Broderick 2018, Alg. 1):
  nonnegative weights w over the vectors `ls` (double arrays of one length),
  at most `m` of them nonzero, with Σ w_n·ls_n close to `target`.

  It works on the unit sphere: each iteration adds the vector whose geodesic
  from the current direction best aligns with the geodesic to the target's
  direction, with the step that maximizes the alignment; the weights are
  scaled optimally at the end. A vector of norm 0 gets weight 0. Returns the
  weights as a vector. Working storage is allocated once per call."
  [ls target m]
  (let [n (count ls)
        norms (mapv norm ls)
        tn (norm target)]
    (if (or (zero? n) (zero? tn))
      (vec (repeat n 0.0))
      (let [ell (lin (/ 1.0 tn) target 0.0 target)
            ells (mapv (fn [l nl] (when (pos? nl) (lin (/ 1.0 nl) l 0.0 l))) ls norms)
            k (alength ^doubles ell)
            d (double-array k)
            lw (double-array k)
            lw' (double-array k)
            w (double-array n)
            finish (fn []
                     (let [align (max 0.0 (dot lw ell))]
                       (mapv (fn [wj nj] (if (pos? nj) (* wj (/ tn nj) align) 0.0)) w norms)))]
        (loop [t 0]
          (let [d (when (< t m) (towards! d ell lw))
                [j score] (when d
                            (loop [j 0 best nil score ##-Inf]
                              (if (< j n)
                                (let [ej (ells j)
                                      s (if ej (towards-score d ej lw) ##-Inf)]
                                  (if (> s score) (recur (inc j) j s) (recur (inc j) best score)))
                                [best score])))]
            (if (or (nil? j) (= ##-Inf score))
              (finish)
              (let [^doubles ej (ells j)
                    z0 (dot ell ej) z1 (dot ell lw) z2 (dot ej lw)
                    num (- z0 (* z1 z2))
                    den (+ num (- z1 (* z0 z2)))
                    gamma (if (pos? den) (min 1.0 (max 0.0 (/ num den))) 0.0)
                    _ (lin! lw' (- 1.0 gamma) lw gamma ej)
                    nn (norm lw')]
                (if (or (zero? gamma) (zero? nn))
                  (finish)
                  (do (dotimes [i n] (aset w i (/ (* (- 1.0 gamma) (aget w i)) nn)))
                      (aset w j (+ (aget w j) (/ gamma nn)))
                      (lin! lw (/ 1.0 nn) lw' 0.0 lw')
                      (recur (inc t))))))))))))

;; =============================================================================
;; NNLS refit on GIGA's support
;; =============================================================================

(def ^:private eps 2.220446049250313e-16)

(defn- residual-norm
  "‖Σ_j w_j·ls_j − target‖."
  ^double [ls w ^doubles target]
  (let [r (aclone target)]
    (doseq [[wj ^doubles l] (map vector w ls)
            :when (not (zero? wj))]
      (dotimes [i (alength r)] (aset r i (- (aget r i) (* (double wj) (aget l i))))))
    (norm r)))

(defn- qr-add
  "The Householder QR of the passive columns `qr` (from `qr-empty`) with the
  column `col` appended, or nil when `col` is numerically dependent on
  them. The column is reflected by the existing reflectors in order and
  then by its own, exactly as a factorization from scratch would; the
  argument is not modified. A QR holds per column its reflector v (from
  its diagonal down, with R's entries above it), vᵀv and R's diagonal, and
  Qᵀb."
  [{:keys [vs vvs diag ^doubles y] :as qr} ^doubles col]
  (let [r (alength y)
        k (count vs)
        sum-from (fn ^double [^long k ^doubles x ^doubles z]
                   (loop [i k s 0.0] (if (< i r) (recur (inc i) (+ s (* (aget x i) (aget z i)))) s)))
        reflect! (fn [^long k ^doubles v ^double vv ^doubles x]
                   (let [f (/ (* 2.0 (sum-from k v x)) vv)]
                     (loop [i k] (when (< i r) (aset x i (- (aget x i) (* f (aget v i)))) (recur (inc i))))))
        x (aclone col)]
    (dotimes [j k] (reflect! j (nth vs j) (nth vvs j) x))
    (let [sigma (Math/sqrt (sum-from k x x))]
      (when-not (or (>= k r) (<= sigma (* 100.0 eps (norm col))))
        (let [alpha (if (pos? (aget x k)) (- sigma) sigma)
              _ (aset x k (- (aget x k) alpha))
              vv (sum-from k x x)
              y (aclone y)]
          (reflect! k x vv y)
          (assoc qr :vs (conj vs x) :vvs (conj vvs vv) :diag (conj diag alpha) :y y))))))

(defn- qr-empty [^doubles b] {:vs [] :vvs [] :diag [] :y (aclone b)})

(defn- qr-solve
  "The least-squares coefficients of b on the QR's columns, in order."
  [{:keys [vs diag ^doubles y]}]
  (let [p (count vs) s (double-array p)]
    (loop [k (dec p)]
      (when (>= k 0)
        (let [acc (loop [j (inc k) acc (aget y k)]
                    (if (< j p) (recur (inc j) (- acc (* (aget ^doubles (nth vs j) k) (aget s j)))) acc))]
          (aset s k (/ acc (double (nth diag k))))
          (recur (dec k)))))
    s))

(defn- qr-of
  "The QR of the columns `ls` at indices `ps`, in order, or nil when one is
  numerically dependent on those before it."
  [ls ps b]
  (reduce (fn [qr j] (or (qr-add qr (nth ls j)) (reduced nil))) (qr-empty b) ps))

(defn nnls
  "Nonnegative least squares (Lawson & Hanson 1974, ch. 23): weights x ≥ 0
  over the vectors `ls` (double arrays of one length) minimizing
  ‖Σ_j x_j·ls_j − target‖, by the active-set method SciPy's `nnls`
  implements. A vector numerically dependent on the passive ones, or one
  that would enter with a nonpositive coefficient, is not added. The
  passive set's Householder QR is extended by one reflection per entering
  vector and refactorized only when vectors leave. Returns the weights as a
  vector."
  [ls target]
  (let [n (count ls)
        ^doubles target target
        tol (* 10.0 eps (max 1 (alength target))
               (reduce max 0.0 (map (fn [^doubles l] (areduce l i s 0.0 (+ s (Math/abs (aget l i))))) ls)))
        dual (fn [x]
               (let [r (aclone target)]
                 (doseq [j (range n) :let [xj (double (nth x j))] :when (pos? xj)]
                   (let [^doubles l (nth ls j)]
                     (dotimes [i (alength r)] (aset r i (- (aget r i) (* xj (aget l i)))))))
                 (mapv #(dot (nth ls %) r) (range n))))
        max-iterations (+ 100 (* 30 n))]
    (loop [x (vec (repeat n 0.0)) passive [] qr (qr-empty target) iterations 0]
      (let [w (dual x)
            in-passive (set passive)
            ;; the most improving vector that keeps the passive columns
            ;; independent and enters with a positive coefficient
            entering (->> (range n)
                          (remove in-passive)
                          (filter #(> (nth w %) tol))
                          (sort-by #(- (nth w %)))
                          (some (fn [t]
                                  (when-let [qr (when qr (qr-add qr (nth ls t)))]
                                    (let [s (qr-solve qr)]
                                      (when (pos? (aget s (dec (alength s))))
                                        [(conj passive t) s qr]))))))]
        (if (or (nil? entering) (>= iterations max-iterations))
          x
          (let [[ps s qr] entering
                [x ps qr]
                (loop [x x ps ps ^doubles s s qr qr inner 0]
                  (if (or (every? pos? s) (>= inner max-iterations))
                    [(reduce (fn [x [j sj]] (assoc x j sj)) x (map vector ps s)) ps qr]
                    ;; step from x towards s until a passive weight reaches 0,
                    ;; and drop the weights that did
                    (let [alpha (reduce min 1.0 (keep (fn [[j sj]]
                                                        (when (<= sj 0.0)
                                                          (let [xj (nth x j)] (/ xj (- xj sj)))))
                                                      (map vector ps s)))
                          x (reduce (fn [x [j sj]] (let [xj (nth x j)] (assoc x j (+ xj (* alpha (- sj xj))))))
                                    x (map vector ps s))
                          floor (* 1e3 eps (reduce max 0.0 x))
                          leaving (set (filter #(<= (nth x %) floor) ps))
                          x (reduce #(assoc %1 %2 0.0) x leaving)
                          ps (filterv (complement leaving) ps)]
                      (if (empty? ps)
                        [x ps (qr-empty target)]
                        (let [qr (qr-of ls ps target)]
                          (recur x ps (some-> qr qr-solve) qr (inc inner)))))))]
            (recur x ps qr (inc iterations))))))))

(defn nnls-refit
  "GIGA's weights `w` refitted by nonnegative least squares on their
  support, the fully corrective update of Campbell & Beronov (NeurIPS 2019;
  `build` then `optimize` in Campbell's bayesian-coresets library): the
  support stays, its weights become the NNLS solution for
  the same `target`, and the refit is kept only if it does not increase the
  residual by more than a relative 1e-12. Weights NNLS sets to zero leave
  the support."
  [ls target w]
  (let [support (vec (keep-indexed (fn [j wj] (when (pos? wj) j)) w))]
    (if (empty? support)
      (vec w)
      (let [x (nnls (mapv #(nth ls %) support) target)
            w' (reduce (fn [w [j xj]] (assoc w j xj)) (vec w) (map vector support x))]
        (if (> (residual-norm ls w' target) (* (residual-norm ls w target) (+ 1.0 1e-12)))
          (vec w)
          w')))))

;; =============================================================================
;; The core-set projection
;; =============================================================================

(defn- recompress
  "The indices of the core-set [[w datum] …] compressed to at most `m`
  points, with their new weights, as [[j w'] …]: GIGA on the data's
  log-likelihoods at the particles (`lls`: per particle, the log-likelihood
  of each core-set datum; normalized log weights `lw`), each centred by its
  population mean and scaled by √π_k, towards the core-set's own weighted
  sum; then, unless `refit` is :none, the weights refitted by NNLS on GIGA's
  support (`nnls-refit`)."
  [lls lw coreset m refit]
  (let [pis (mapv #(Math/exp %) lw)
        roots (mapv #(Math/sqrt %) pis)
        cols (mapv (fn [j]
                     (let [f (mapv #(aget ^doubles % j) lls)
                           mean (reduce + (map * pis f))]
                       (double-array (map (fn [fk r] (* r (- fk mean))) f roots))))
                   (range (count coreset)))
        target (reduce (fn [^doubles acc [^doubles col [w _]]] (lin 1.0 acc (double w) col))
                       (double-array (count lls))
                       (map vector cols coreset))
        w (cond->> (giga cols target m)
            (not= :none refit) (nnls-refit cols target))]
    (into [] (keep-indexed (fn [j wj] (when (pos? wj) [j wj]))) w)))

(defn- weighted-sum
  "Σ_j w_j·ls_j over the core-set entries [[w datum] …], from 0.0 in order."
  [coreset ls]
  (reduce + 0.0 (map (fn [[w _] l] (* w l)) coreset ls)))

(defn- row
  "A particle's cached log-likelihoods: a compact persistent vector."
  [xs]
  #?(:clj (into (vector-of :double) xs) :cljs (vec xs)))

(defn- inflated?
  "Whether a :projection-inflation factor moves the projection's points."
  [k]
  (and (some? k) (not (== 1 k))))

(defn- inflate
  "The particles `qs` spread by the factor `k` about their (uniformly
  weighted) mean: μ + k·(q − μ)."
  [qs k]
  (let [n (count qs)
        mean (apply mapv (fn [& xs] (/ (reduce + xs) n)) qs)]
    (mapv (fn [q] (mapv (fn [x mu] (+ mu (* k (- x mu)))) q mean)) qs)))

(defn start
  "Initialize resumable numeric SCMC. model provides :prior-sample,
  :prior-log-density and :log-lik. State is a persistent in-memory value,
  including model functions; it is not a serialized inference artifact.
  opts: :particles (1000), :memory (100), :scale (2.38), :seed, :refit
  (:nnls, the default, or :none) and :projection-inflation (none): see the
  namespace. Without a :seed, start draws one and records it in the state
  (:seed): replay from `(initial state)` or `(start model (options
  state))`, never from the original opts, to reproduce the run."
  [{:keys [prior-sample] :as model}
   & [{:keys [particles memory scale seed refit projection-inflation]
       :or {particles 1000 memory 100 scale 2.38 refit :nnls}}]]
  (when-not (pos-int? particles)
    (throw (ex-info ":particles is a positive count" {:type ::invalid-particles :particles particles})))
  (when-not (pos-int? memory)
    (throw (ex-info ":memory is a positive count" {:type ::invalid-memory :memory memory})))
  (when-not (#{:nnls :none} refit)
    (throw (ex-info ":refit is :nnls or :none" {:type ::invalid-refit :refit refit})))
  (when-not (or (nil? projection-inflation)
                (and (number? projection-inflation) (pos? projection-inflation)
                     (< projection-inflation ##Inf)))
    (throw (ex-info ":projection-inflation is a positive factor"
                    {:type ::invalid-projection-inflation :projection-inflation projection-inflation})))
  (let [seed (or seed (random/fresh-seed))
        qs (mapv (fn [i] (random/with-stream* seed [::init i] #(vec (prior-sample))))
                 (range particles))]
    {:model (select-keys model [:prior-sample :prior-log-density :log-lik])
     :particles qs
     :memory memory :scale scale :seed seed :refit refit
     :projection-inflation projection-inflation :coreset [] :steps 0
     ;; per particle, the log-likelihood of each core-set datum, for this
     ;; log-likelihood, core-set and population only
     :coreset-log-likelihoods {:log-lik (:log-lik model) :coreset [] :particles qs
                               :rows (vec (repeat particles (row [])))}
     :compressed? false :approximate? false :log-normalizer 0.0
     :max-retained-coreset-size 0 :max-transient-coreset-size 0 :max-coreset-size 0
     :moves 0 :accepted 0}))

(defn options
  "The `start` options that reproduce `state`'s run: with the same batches,
  `(append-batch …)` from `(start model (options state))` gives `state`
  again. They include the seed `start` drew when none was given, so a
  replay (after a historical correction, say) is reproducible. A state
  without :refit was made before the refit existed and resumes without it."
  [{:keys [particles memory scale seed refit projection-inflation]}]
  (cond-> {:particles (count particles) :memory memory :scale scale :seed seed
           ;; a state from before :refit existed was made without the refit
           :refit (or refit :none)}
    projection-inflation (assoc :projection-inflation projection-inflation)))

(defn initial
  "The state `state`'s run started from: `start` with its model and
  `options`, the seed included."
  [state]
  (start (:model state) (options state)))

(defn append-batch
  "Advance a persistent SCMC state by one non-empty batch. The old state
  remains reusable. Retains only the population and a bounded coreset; the
  caller owns any observation history needed for replay or correction.
  :last-update reports ESS before resampling, not convergence or accuracy.
  Historical replacement/retraction is not an append operation."
  [{:keys [particles memory scale seed coreset steps model compressed? refit projection-inflation] :as state} batch]
  (let [batch (vec batch)]
    (when (empty? batch)
      (throw (ex-info "An SCMC update requires a non-empty batch" {:type ::empty-batch})))
    (let [{:keys [prior-log-density log-lik]} model
          n (count particles)
          expanded (into coreset (map (fn [x] [1.0 x])) batch)
          ;; every particle's log-likelihood of each new datum; with the
          ;; cached ones of the core-set, the density of the moves at a
          ;; start state needs no new likelihood evaluation
          batch-lls (mapv (fn [q] (mapv #(log-lik q %) batch)) particles)
          increments (mapv #(reduce + 0.0 %) batch-lls)
          mass (weights/from-log increments)
          _ (when (= :zero-mass (:status mass))
              (throw (ex-info "Every particle has zero likelihood for a batch: no posterior population"
                              {:type ::zero-evidence :batch steps})))
          ess (m/compute-ess (weights/selection mass))
          ;; the cache holds the log-likelihoods of this log-likelihood
          ;; function, core-set and population; any other is re-evaluated
          cache (:coreset-log-likelihoods state)
          cached (when (and (map? cache)
                            (identical? log-lik (:log-lik cache))
                            (= coreset (:coreset cache))
                            (= particles (:particles cache)))
                   (:rows cache))
          start-index (when cached (zipmap particles (range)))
          start-row (fn [i]
                      (let [c (nth cached i) b (nth batch-lls i)
                            k (count c) out (double-array (+ k (count b)))]
                        (dotimes [j k] (aset out j (double (nth c j))))
                        (dotimes [j (count b)] (aset out (+ k j) (double (nth b j))))
                        out))
          ;; the likelihood rows of the states the current chain has
          ;; evaluated, and of the states the chains kept: only the latter
          ;; outlive a chain
          pending (volatile! {})
          survivors (volatile! {})
          counts (volatile! {:move 0 :projection 0})
          evaluate (fn [phase q]
                     (vswap! counts update phase + (count expanded))
                     (double-array (map (fn [[_ u]] (log-lik q u)) expanded)))
          target (fn [q lp ls]
                   (vswap! pending assoc q ls)
                   (+ lp (weighted-sum expanded ls)))
          stats (atom {:moves 0 :accepted 0})
          path {:start 0 :done? pos?
                :advance (fn [_ _ _] {:pos 1 :log-incr increments})
                :density (fn [_]
                           (fn [q]
                             (let [lp (prior-log-density q)]
                               (if (= ##-Inf lp) lp (target q lp (evaluate :move q))))))
                :start-log-density (fn [_]
                                     (fn [q]
                                       (when-let [i (get start-index q)]
                                         (let [lp (prior-log-density q)]
                                           (if (= ##-Inf lp) lp (target q lp (start-row i)))))))
                :kept (fn [_ q]
                        (when-let [ls (get @pending q)] (vswap! survivors assoc q ls))
                        (vreset! pending {}))
                :exhausted (fn [_] (ex-info "SCMC single update exhausted" {:type ::exhausted}))}
          {:keys [qs log-z]} (pop/theta-smc particles path
                                            {:n n :scale scale :seed seed :max-steps 1 :step-offset steps
                                             :positions? false :stats stats
                                             :keys {:resample ::resample :move ::move}})
          lls (mapv (fn [q] (or (get @survivors q) (evaluate :projection q))) qs)
          compress? (> (count expanded) memory)
          kept (when compress?
                 (recompress (if (inflated? projection-inflation)
                               (let [ls (mapv #(evaluate :projection %) (inflate qs projection-inflation))]
                                 (when-not (every? (fn [row] (every? #(< ##-Inf % ##Inf) row)) ls)
                                   (throw (ex-info "A likelihood is not finite at an inflated projection point"
                                                   {:type ::non-finite-projection :batch steps
                                                    :projection-inflation projection-inflation})))
                                 ls)
                               lls)
                             (vec (repeat n (- (Math/log n)))) expanded memory (or refit :none)))
          coreset (if compress?
                    (mapv (fn [[j w]] [w (second (expanded j))]) kept)
                    expanded)]
      (-> state
          (assoc :particles qs :coreset coreset :steps (inc steps)
                 :coreset-log-likelihoods {:log-lik log-lik :coreset coreset :particles qs
                                           :rows (if compress?
                                                   (mapv (fn [^doubles ls] (row (map #(aget ls (first %)) kept))) lls)
                                                   (mapv row lls))}
                 :compressed? (or compressed? compress?)
                 :approximate? (or (:approximate? state) compressed?)
                 :last-update {:ess ess :ess-fraction (/ ess n) :batch-size (count batch)
                               :compressed? compress?
                               :likelihood-evaluations (assoc @counts :reweight (* n (count batch)))})
          (update :log-normalizer + log-z)
          (update :max-retained-coreset-size (fnil max 0) (count coreset))
          (update :max-transient-coreset-size (fnil max 0) (count expanded))
          ;; the transient size, under its pre-split name
          (update :max-coreset-size max (count expanded))
          (update :moves + (:moves @stats))
          (update :accepted + (:accepted @stats))))))

(defn result
  "Project an SCMC snapshot to the common settled measure API. Projection
  does not consume the snapshot. Coreset moves and population adaptation do
  not certify an unbiased normalizer or posterior convergence."
  [{:keys [particles approximate?] :as state}]
  (merge (m/empirical (mapv (fn [q] [(m/sample-particle q {}) 0.0]) particles))
         (select-keys state [:coreset :steps :max-retained-coreset-size :max-transient-coreset-size
                             :max-coreset-size :moves :accepted :log-normalizer :last-update])
         {:inference-contract {:semantics (if approximate? :approximate-population :weighted-population)
                               :normalizer {:status :unverified}
                               :adaptation :population
                               :approximation (when approximate? :coreset-rejuvenation)}}))

(defn scmc
  "Run numeric SCMC over :batches using start, append-batch and result.
  model: :prior-sample (fn [] theta-vector), :prior-log-density (fn [theta]),
  :log-lik (fn [theta datum]), :batches (lazy sequence of non-empty batches).
  opts: :particles (1000), :memory (100), :scale (2.38), :seed, :refit
  (:nnls or :none), :projection-inflation.
  Returns an EmpiricalMeasure with :coreset, :steps,
  :max-retained-coreset-size (the largest core-set kept between steps, at
  most :memory), :max-transient-coreset-size (the largest core-set plus
  batch a step moved against; :max-coreset-size is its old name), :moves,
  :accepted, :last-update and :log-normalizer. Streaming memory is
  bounded by population, coreset and batch; no history of updates is retained.
  To pause between batches, retain the state returned by append-batch."
  [{:keys [batches] :as model} & [opts]]
  (result (reduce append-batch (start model opts) batches)))
