(ns org.replikativ.foerster.causal-model
  "Fit and check declared, causally sufficient DAGs from complete numeric rows.
  The default mechanisms are linear Gaussian with independent standardized
  noise. Fitting is not causal discovery, and diagnostics do not establish
  the graph or its counterfactual coupling. No external numerical runtime."
  (:require [org.replikativ.foerster.causal :as causal]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.mechanism :as mechanism]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.spin.cps :refer [spin]])
  #?(:cljs (:require-macros [org.replikativ.spindel.spin.cps :refer [spin]])))

(defn- finite? [x] (and (number? x) (< ##-Inf x ##Inf)))
(defn- mean [xs] (/ (reduce + 0.0 xs) (count xs)))
(defn- dot [xs ys] (reduce + 0.0 (map * xs ys)))
(defn- centered [xs] (let [mu (mean xs)] (mapv #(- % mu) xs)))

(defn- rows! [nodes rows]
  (when-not (and (seq rows) (every? #(and (map? %) (every? (fn [v] (finite? (get % v))) nodes)) rows))
    (throw (ex-info "Supply nonempty complete rows with finite numeric node values"
                    {:type ::invalid-data})))
  (vec rows))

(defn- least-squares
  "Reorthogonalized modified Gram-Schmidt on centered, scaled columns.
  Avoid normal equations and refuse rank deficiency rather than regularize."
  [columns y]
  (let [{:keys [q r]}
        (reduce (fn [{:keys [q r]} column]
                  (let [[v projections]
                        (reduce (fn [[v totals] _]
                                  (reduce (fn [[v totals] [i qi]]
                                            (let [a (dot qi v)]
                                              [(mapv - v (mapv #(* a %) qi)) (update totals i + a)]))
                                          [v totals] (map-indexed vector q)))
                                [column (vec (repeat (count q) 0.0))] (range 2))
                        norm (Math/sqrt (dot v v))]
                    (when-not (> norm 1e-10)
                      (throw (ex-info "Parent design is rank deficient"
                                      {:type ::rank-deficient :column (count q)})))
                    {:q (conj q (mapv #(/ % norm) v))
                     :r (conj r (conj projections norm))}))
                {:q [] :r []} columns)
        rhs (mapv #(dot % y) q)]
    (loop [j (dec (count q)) beta (vec (repeat (count q) 0.0))]
      (if (neg? j) beta
          (let [tail (reduce + 0.0 (for [k (range (inc j) (count q))]
                                     (* (get-in r [k j]) (beta k))))]
            (recur (dec j) (assoc beta j (/ (- (rhs j) tail) (get-in r [j j])))))))))

(defn linear-gaussian
  "Fitter contract: (fitter node ordered-parents training-rows) -> mechanism.
  OLS mean, maximum-likelihood residual scale; refuses insufficient rows,
  constant/collinear parents and effectively deterministic residuals."
  [node parents rows]
  (let [n (count rows) p (count parents)]
    (when-not (> n (inc p))
      (throw (ex-info "A mechanism needs more rows than mean parameters"
                      {:type ::insufficient-data :node node})))
    (let [y (mapv #(get % node) rows) yc (centered y)
          columns (mapv (fn [v] (mapv #(get % v) rows)) parents)
          centers (mapv mean columns)
          centered-columns (mapv centered columns)
          scales (mapv #(Math/sqrt (dot % %)) centered-columns)]
      (when-not (every? pos? scales)
        (throw (ex-info "Constant parents do not identify coefficients"
                        {:type ::rank-deficient :node node})))
      (let [beta (mapv / (least-squares (mapv (fn [xs s] (mapv #(/ % s) xs)) centered-columns scales) yc) scales)
            y-mean (mean y)
            intercept (- y-mean (dot beta centers))
            coefficients (zipmap parents beta)
            predict (fn [row] (+ y-mean (dot beta (mapv - (mapv #(get row %) parents) centers))))
            residuals (mapv #(- (get % node) (predict %)) rows)
            sigma (Math/sqrt (/ (dot residuals residuals) n))]
        (when-not (and (finite? sigma) (> sigma (* 1e-10 (Math/sqrt (/ (dot yc yc) n)))))
          (throw (ex-info "Residual scale is zero or numerically degenerate; declare a deterministic equation instead"
                          {:type ::degenerate-noise :node node})))
        {:family :linear-gaussian :intercept intercept :coefficients coefficients :sigma sigma
         :distribution (fn [row] (dist/normal (predict row) sigma))}))))

(defn fit
  "Fit a declared DAG, defaulting to linear Gaussian mechanisms. `rows` are
  IID observational records, with all graph nodes observed. Hidden common
  causes (:bidirected edges) are refused. :fitters maps node -> custom fitter
  with the `linear-gaussian` signature, returning {:distribution (fn [parents]
  law)}. That law must implement Distribution and PMechanism. A custom
  :noise-score maps noise to a finite scalar for diagnostics (default identity).
  Independent noise and the graph's causal meaning are author assumptions."
  ([g rows] (fit g rows nil))
  ([g rows {:keys [fitters] :as options}]
   (let [g (causal/graph g) order (causal/topological-order g) rows (rows! order rows)]
     (when (or (empty? order) (seq (:bidirected g)) (some #(not (contains? (:nodes g) %)) (keys fitters)))
       (throw (ex-info "Fitting needs a nonempty causally sufficient DAG and declared fitter nodes"
                       {:type ::unsupported-graph})))
     {:graph g :order order :training-count (count rows) :fit-options options
      :mechanisms
      (into {} (for [v order
                     :let [parents (vec (filter (set (map first (filter #(= v (second %)) (:directed g)))) order))
                           fitted ((get fitters v linear-gaussian) v parents rows)
                           law (when (fn? (:distribution fitted)) ((:distribution fitted) (select-keys (first rows) parents)))]]
                 (do (when-not (and (dist/distribution? law) (mechanism/mechanism? law))
                       (throw (ex-info "A fitter must return a structural distribution"
                                       {:type ::invalid-fitter :node v})))
                     [v (assoc fitted :parents parents)])))})))

(defn distribution
  "Conditional structural law at `node`. Only declared parents reach its callback."
  [fitted node values]
  (let [m (get-in fitted [:mechanisms node])]
    (when-not (and m (every? #(contains? values %) (:parents m)))
      (throw (ex-info "Unknown node or missing parents" {:type ::missing-parents :node node})))
    ((:distribution m) (select-keys values (:parents m)))))

(defn model
  "An ordinary foerster model spin: one sample site per node, node IDs as
  addresses. Use with inference, cf/query, attribution and trace policies."
  [fitted]
  (spin
   (loop [remaining (:order fitted) values {}]
     (if (seq remaining)
       (let [v (first remaining)
             x (sample (distribution fitted v values) :id v)]
         (recur (rest remaining) (assoc values v x)))
       values))))

(defn- abduct-noise [fitted node row]
  (let [law (distribution fitted node row) x (get row node)]
    (when-not (> (dist/logpdf law x) ##-Inf)
      (throw (ex-info "Observation has zero density/mass or an undefined score under its mechanism"
                      {:type ::impossible-observation :node node :value x})))
    (mechanism/abduct law x)))

(defn abduct
  "Reusable cf abduction for one COMPLETE observation at fixed fitted
  parameters. Continuous invertible noise is exact; custom discrete
  mechanisms sample conditional noise. No latent/parameter posterior is
  inferred. Use cf/query with evidence for incomplete observations instead."
  [fitted row]
  (rows! (:order fitted) [row])
  {:factual (select-keys row (:order fitted)) :weight 0.0 :unsupported []
   :noise (into {} (for [v (:order fitted)]
                     [v (abduct-noise fitted v row)]))})

(defn replay
  "Synchronous structural evaluation at supplied noise. `interventions` maps
  nodes to hard values (not trace policy maps). For richer interventions and
  compositional worlds use `model` with the existing cf APIs."
  ([fitted noise] (replay fitted noise {}))
  ([fitted noise interventions]
   (when (some #(not (contains? (:mechanisms fitted) %)) (keys interventions))
     (throw (ex-info "Intervention names an unknown node" {:type ::unknown-intervention})))
   (reduce (fn [values v]
             (assoc values v
                    (if (contains? interventions v) (get interventions v)
                        (if (contains? noise v)
                          (mechanism/push (distribution fitted v values) (get noise v))
                          (throw (ex-info "Replay needs noise for every unintervened node"
                                          {:type ::missing-noise :node v}))))))
           {} (:order fitted))))

(defn simulate
  "IID draws from the fitted model, optionally under hard interventions.
  :seed gives a reproducible stream without changing the process RNG."
  ([fitted n] (simulate fitted n nil))
  ([fitted n {:keys [seed interventions]}]
   (when-not (and (int? n) (not (neg? n)))
     (throw (ex-info "Sample count must be a nonnegative integer" {:type ::invalid-options})))
   (when (some #(not (contains? (:mechanisms fitted) %)) (keys interventions))
     (throw (ex-info "Intervention names an unknown node" {:type ::unknown-intervention})))
   (random/with-stream* (or seed (random/fresh-seed)) ::simulate
     #(mapv (fn [_]
              (reduce (fn [values v]
                        (assoc values v (if (contains? interventions v) (get interventions v)
                                            (dist/draw (distribution fitted v values)))))
                      {} (:order fitted))) (range n)))))

(defn- correlation [xs ys]
  (let [x (centered xs) y (centered ys) denominator (Math/sqrt (* (dot x x) (dot y y)))]
    (if (pos? denominator) (/ (dot x y) denominator) 0.0)))

(defn- features [xs]
  (let [x (centered xs) scale (Math/sqrt (/ (dot x x) (count x)))
        z (if (pos? scale) (mapv #(/ % scale) x) x)]
    [z (mapv #(* % %) z)]))

(defn- dependence [x y]
  (apply max (for [a (features x) b (features y)] (Math/abs (correlation a b)))))

(defn- shuffled [xs]
  (loop [v (vec xs) i (dec (count xs))]
    (if (pos? i)
      (let [j (int (* (random/uniform01) (inc i)))]
        (recur (assoc v i (v j) j (v i)) (dec i))) v)))

(defn- independence-test [x y permutations]
  (let [observed (dependence x y)
        exceedances (count (filter #(>= % observed) (repeatedly permutations #(dependence x (shuffled y)))))]
    {:statistic observed :p-value (/ (inc exceedances) (double (inc permutations)))}))

(defn evaluate
  "Evaluate on separate held-out IID complete rows. Reports node log scores,
  scalar noise moments and, for built-in Gaussian fits, RMSE and 90% coverage.
  Seeded permutation tests use the maximum absolute correlation between
  centered/scaled x,x² and y,y²: residual-vs-parent and residual-vs-residual.
  Holm correction controls family-wise error at :alpha (default .05).
  Defaults: 199 permutations. This test detects some nonlinear dependence,
  not all; no rejection is not evidence of causal correctness or Gaussianity.
  Report :rejected-checks, never a 'causally valid' flag."
  ([fitted held-out] (evaluate fitted held-out nil))
  ([fitted held-out {:keys [seed permutations alpha] :or {permutations 199 alpha 0.05}}]
   (let [order (:order fitted) rows (rows! order held-out)]
     (when-not (and (>= (count rows) 4) (pos-int? permutations) (< 0 alpha 1))
       (throw (ex-info "Evaluation needs at least four rows, positive permutations and 0 < alpha < 1"
                       {:type ::invalid-options})))
     (random/with-stream* (or seed (random/fresh-seed)) ::evaluate
       (fn []
         (let [noise (mapv (fn [row]
                             (into {} (for [v order
                                            :let [u (abduct-noise fitted v row)
                                                  score ((get-in fitted [:mechanisms v :noise-score] identity) u)]]
                                        (do (when-not (finite? score)
                                              (throw (ex-info "Diagnostics need finite scalar noise scores"
                                                              {:type ::invalid-noise-score :node v})))
                                            [v score])))) rows)
               metrics (into {} (for [v order
                                      :let [u (mapv #(get % v) noise)
                                            laws (mapv #(distribution fitted v %) rows)
                                            log-scores (mapv #(dist/logpdf %1 (get %2 v)) laws rows)
                                            gaussian? (= :linear-gaussian (get-in fitted [:mechanisms v :family]))]]
                                  [v (cond-> {:mean-log-score (mean log-scores)
                                              :noise-mean (mean u) :noise-variance (mean (mapv #(* % %) (centered u)))}
                                       gaussian? (assoc :rmse (Math/sqrt (mean (mapv #(* % %) (mapv #(- (get %1 v) (:mu %2)) rows laws))))
                                                        :coverage-90 (mean (mapv #(if (<= 0.05 (dist/cdf %1 (get %2 v)) 0.95) 1.0 0.0) laws rows))))]))
               pairs (concat (for [v order p (get-in fitted [:mechanisms v :parents])]
                               {:kind :noise-parent :nodes [v p] :x (mapv #(get % v) noise) :y (mapv #(get % p) rows)})
                             (for [i (range (count order)) j (range (inc i) (count order))]
                               {:kind :noise-noise :nodes [(order i) (order j)]
                                :x (mapv #(get % (order i)) noise) :y (mapv #(get % (order j)) noise)}))
               tests (mapv (fn [{:keys [x y] :as pair}]
                             (merge (dissoc pair :x :y) (independence-test x y permutations))) pairs)
               adjusted (loop [remaining (sort-by :p-value tests) rank 0 previous 0.0 out []]
                          (if-let [[test & more] (seq remaining)]
                            (let [p (min 1.0 (max previous (* (- (count tests) rank) (:p-value test))))]
                              (recur more (inc rank) p (conj out (assoc test :adjusted-p-value p :rejected? (<= p alpha))))) out))]
           {:evaluation-count (count rows) :mechanisms metrics :checks adjusted
            :rejected-checks (vec (filter :rejected? adjusted))
            :alpha alpha :permutations permutations}))))))

(defn bootstrap
  "IID row bootstrap: refit ALL mechanisms and compute a finite scalar
  (statistic fitted-model). Reports percentile interval, point estimate,
  and every draw. Defaults: 200 replicates, 95% interval. Failures abort
  rather than silently dropping singular refits. This is sampling uncertainty
  conditional on graph/family, not uncertainty about causal assumptions.
  Fitting callbacks must be deterministic for seeded reproducibility."
  ([fitted training statistic] (bootstrap fitted training statistic nil))
  ([fitted training statistic {:keys [seed replicates level] :or {replicates 200 level 0.95}}]
   (let [rows (rows! (:order fitted) training) n (count rows)
         score (fn [m] (let [x (statistic m)]
                         (when-not (finite? x)
                           (throw (ex-info "Bootstrap statistic must be finite" {:type ::invalid-statistic}))) x))]
     (when-not (and (pos-int? replicates) (< 0 level 1))
       (throw (ex-info "Use positive replicates and 0 < level < 1" {:type ::invalid-options})))
     (random/with-stream* (or seed (random/fresh-seed)) ::bootstrap
       (fn []
         (let [point (score fitted)
               draws (mapv (fn [_]
                             (score (fit (:graph fitted)
                                         (mapv (fn [_] (rows (int (* n (random/uniform01))))) (range n))
                                         (:fit-options fitted)))) (range replicates))
               sorted (vec (sort draws))
               quantile (fn [p]
                          (let [position (* p (dec replicates)) lo (int (Math/floor position))
                                hi (int (Math/ceil position)) t (- position lo)]
                            (+ (* (- 1 t) (sorted lo)) (* t (sorted hi)))))
               tail (/ (- 1 level) 2)]
           {:estimate point :interval [(quantile tail) (quantile (- 1 tail))]
            :level level :replicates replicates :draws draws}))))))
