(ns org.replikativ.foerster.incremental-baseline
  "What a single-site move costs today, as the data grow: the baseline for
  incremental density computation (Zaiser et al., PLDI 2026,
  doi:10.1145/3808316; doc/design.md, \"Incremental re-execution\").

  Not part of the test suite: a timing experiment, run with
  `clojure -M:incremental [sizes…]`. Machine load moves the absolute numbers;
  the scaling in N is what it measures.

  The models are the paper's case studies 1 and 2, written as foerster
  programs in two shapes:

    flat   one spin, a loop over the data with a sample and an observe per
           point
    items  the same, but each point is a sub-spin the loop awaits, keyed by
           its index: the shape Gen's Map combinator incrementalizes
    keyed  the points as one `spindel.spin.keyed` loop: a move of a point
           re-runs only its key (`update-step`); the mixture also times an
           exact Gibbs update of one assignment (`update-gibbs`)

  Robust regression is the artifact's model (m, b ~ N(0, 1); a point is an
  outlier with probability 0.1, y ~ N(0, 100), else y ~ N(m x + b, 0.5); x
  on a grid over [−5, 5]). The binary mixture keeps the artifact's mixing
  weight ~ Beta(0.5, 0.5) and means ~ N(μ₀ₖ, Σ/0.4) with μ₀ = (−2, −2),
  (2, 2), but fixes Σ = I: foerster has no Inverse-Wishart. Neither changes
  what a move has to re-execute.

  Each move is an `mh-step` on one site with a deterministic proposal (an
  indicator flipped, a parameter shifted), timed from the trace it starts
  from; a move is accepted or not as MH decides, and the trace it starts
  from is kept. Reported per N, medians in ms:

    run      a traced run of the whole program (`gfi/generate`): what full
             recomputation costs
    first    flip the first point's indicator: replays the whole program
    mid      flip point N/2's: replays half
    last     flip the last point's: replays one point
    param    shift a parameter (b, or component 0's mean), which every
             point reads"
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.effects.await :as aw]
            [org.replikativ.spindel.engine.addressing :as addressing]
            [org.replikativ.spindel.spin.keyed :refer [keyed]]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.trace :as trace]))

;; --- robust regression (case study 1) ---------------------------------------

(defn regression-data
  "N points on a grid over [−5, 5], simulated with m = −2.4, b = −1.25."
  [n seed]
  (let [r (java.util.Random. seed)
        xs (mapv #(+ -5.0 (* 10.0 (/ % (double (max 1 (dec n)))))) (range n))]
    {:xs xs
     :ys (mapv (fn [x] (if (< (.nextDouble r) 0.1)
                         (* 100.0 (.nextGaussian r))
                         (+ (* -2.4 x) -1.25 (* 0.5 (.nextGaussian r)))))
               xs)}))

(defn regression-flat [{:keys [xs ys]}]
  (spin
   (let [m (sample (dist/normal 0.0 1.0) :id :m)
         b (sample (dist/normal 0.0 1.0) :id :b)]
     (loop [i 0]
       (when (< i (count xs))
         (let [o (sample (dist/flip 0.1) :id [:o i])]
           (observe (dist/normal (if o 0.0 (+ (* m (nth xs i)) b)) (if o 100.0 0.5))
                    (nth ys i) :id [:y i])
           (recur (inc i)))))
     [m b])))

(defn- regression-item [m b x y i]
  (spin
   (let [o (sample (dist/flip 0.1) :id [:o i])]
     (observe (dist/normal (if o 0.0 (+ (* m x) b)) (if o 100.0 0.5)) y :id [:y i])
     o)))

(defn regression-keyed
  "The points as one keyed loop: a move of one point re-runs that key
  (`update-step`)."
  [{:keys [xs ys]}]
  (spin
   (let [m (sample (dist/normal 0.0 1.0) :id :m)
         b (sample (dist/normal 0.0 1.0) :id :b)
         os (aw/await (keyed :points (range (count xs)) (fn [i] (regression-item m b (xs i) (ys i) i))))]
     [m b (count (filter true? os))])))

(defn regression-items [{:keys [xs ys]}]
  (spin
   (let [m (sample (dist/normal 0.0 1.0) :id :m)
         b (sample (dist/normal 0.0 1.0) :id :b)]
     (loop [i 0]
       (when (< i (count xs))
         (aw/await (addressing/with-key i (regression-item m b (nth xs i) (nth ys i) i)))
         (recur (inc i))))
     [m b])))

;; --- binary mixture (case study 2) ------------------------------------------

(def ^:private prior-means [[-2.0 -2.0] [2.0 2.0]])
(def ^:private identity2 [[1.0 0.0] [0.0 1.0]])
(def ^:private mean-cov [[2.5 0.0] [0.0 2.5]]) ; Σ/κ₀, Σ = I, κ₀ = 0.4

(defn mixture-data
  "N points from the mixture with weight 0.4 and the prior means."
  [n seed]
  (let [r (java.util.Random. seed)]
    {:points (vec (repeatedly n #(let [[mx my] (prior-means (if (< (.nextDouble r) 0.4) 1 0))]
                                   [(+ mx (.nextGaussian r)) (+ my (.nextGaussian r))])))}))

(defn mixture-flat [{:keys [points]}]
  (spin
   (let [w (sample (dist/beta 0.5 0.5) :id :weight)
         mu0 (sample (dist/mvn (prior-means 0) mean-cov) :id [:mu 0])
         mu1 (sample (dist/mvn (prior-means 1) mean-cov) :id [:mu 1])]
     (loop [i 0]
       (when (< i (count points))
         (let [z (sample (dist/flip w) :id [:z i])]
           (observe (dist/mvn (if z mu1 mu0) identity2) (nth points i) :id [:x i])
           (recur (inc i)))))
     [w mu0 mu1])))

(defn- mixture-item [w mu0 mu1 point i]
  (spin
   (let [z (sample (dist/flip w) :id [:z i])]
     (observe (dist/mvn (if z mu1 mu0) identity2) point :id [:x i])
     z)))

(defn mixture-keyed
  "The points as one keyed loop: an assignment's Gibbs update re-runs that
  key once per value (`update-gibbs`)."
  [{:keys [points]}]
  (spin
   (let [w (sample (dist/beta 0.5 0.5) :id :weight)
         mu0 (sample (dist/mvn (prior-means 0) mean-cov) :id [:mu 0])
         mu1 (sample (dist/mvn (prior-means 1) mean-cov) :id [:mu 1])
         zs (aw/await (keyed :points (range (count points)) (fn [i] (mixture-item w mu0 mu1 (nth points i) i))))]
     [w mu0 mu1 (count (filter true? zs))])))

(defn mixture-items [{:keys [points]}]
  (spin
   (let [w (sample (dist/beta 0.5 0.5) :id :weight)
         mu0 (sample (dist/mvn (prior-means 0) mean-cov) :id [:mu 0])
         mu1 (sample (dist/mvn (prior-means 1) mean-cov) :id [:mu 1])]
     (loop [i 0]
       (when (< i (count points))
         (aw/await (addressing/with-key i (mixture-item w mu0 mu1 (nth points i) i)))
         (recur (inc i))))
     [w mu0 mu1])))

;; --- measuring ----------------------------------------------------------------

(defn- await* [op]
  (let [p (promise)]
    (op #(deliver p %) #(deliver p %))
    (let [v (deref p 600000 ::timeout)]
      (cond (= ::timeout v) (throw (ex-info "timed out" {}))
            (instance? Throwable v) (throw v)
            :else v))))

(defn- ms [f]
  (let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6)))

(defn- median [xs] (let [s (vec (sort xs))] (nth s (quot (count s) 2))))

(defn- at
  "mh-step options that move exactly `address` with `propose`."
  [address propose]
  {:select (fn [_ _] {:targets #{address} :log-selection (fn [_] 0.0)})
   :propose propose})

(defn- flip-value [_ old] {:value (not (:value old)) :symmetric? true})

(defn- shift-value [_ old]
  {:value (let [v (:value old)] (if (vector? v) (mapv #(+ % 0.05) v) (+ v 0.05)))
   :symmetric? true})

(defn- move-ms
  "Median ms of `reps` moves from `t` by `step` (`mh-step`, `update-step`);
  each move's result is released, `t` kept."
  [step t opts reps]
  (median (for [i (range reps)]
            (let [result (volatile! nil)
                  elapsed (ms #(vreset! result (await* (step t (assoc opts :iteration i :keep-old? true)))))
                  {t' :trace accepted? :accepted?} @result]
              (when accepted? (trace/release! t' t))
              elapsed))))

(def cases
  {:regression {:data regression-data :site :o :param :b
                :constraints {:m 0.0 :b 0.0}
                :shapes {:flat regression-flat :items regression-items :keyed regression-keyed}}
   :mixture {:data mixture-data :site :z :param [:mu 0]
             :constraints {:weight 0.4 [:mu 0] [-2.0 -2.0] [:mu 1] [2.0 2.0]}
             :shapes {:flat mixture-flat :items mixture-items :keyed mixture-keyed}}})

(defn measure
  "One row: `case` (a key of `cases`) in `shape` at `n` points."
  [case shape n reps]
  (let [{:keys [data site param constraints shapes]} (cases case)
        model (shapes shape)
        d (data n 7)
        point-site? (fn [address] (and (vector? address) (#{:o :y :z :x} (first address))))
        generate (fn []
                   (let [t (:trace (await* (gfi/generate (model d) constraints)))]
                     (when-let [e (:trace/error t)]
                       (throw (ex-info "the model failed" {:case case :shape shape} e)))
                     (when-not (= (* 2 n) (count (filter point-site? (keys (:trace/entries t)))))
                       (throw (ex-info "the trace misses sites" {:case case :shape shape :n n})))
                     t))
        run-ms (median (for [_ (range 3)] (ms #(await* (gfi/close! (generate))))))
        t (generate)
        ;; a keyed model moves a point by re-running its key
        step (if (= :keyed shape) itrace/update-step itrace/mh-step)]
    (try
      (cond-> {:case case :shape shape :n n
               :run run-ms
               :first (move-ms step t (at [site 0] flip-value) reps)
               :mid (move-ms step t (at [site (quot n 2)] flip-value) reps)
               :last (move-ms step t (at [site (dec n)] flip-value) reps)
               :param (move-ms step t (at param shift-value) reps)}
        (and (= :keyed shape) (= :z site))
        (assoc :gibbs (median (for [i (range reps)]
                                (ms #(await* (itrace/update-gibbs t [site (quot n 2)]
                                                                  {:iteration (+ 1000 i) :keep-old? true})))))))
      (finally (await* (gfi/close! t))))))

(defn run
  "Rows for every case and shape at `sizes`, each printed as it is measured."
  [sizes reps]
  (random/set-seed! 7)
  (let [root (ctx/create-execution-context)]
    (try
      (binding [rtc/*execution-context* root]
        ;; warm the JIT on a small instance of each
        (doseq [case (keys cases) shape [:flat :items :keyed]] (measure case shape 50 2))
        (vec (for [case (keys cases) shape [:flat :items :keyed] n sizes]
               (let [row (measure case shape n reps)]
                 (println (format "%-10s %-5s %5d  run %8.1f  first %8.1f  mid %8.1f  last %6.1f  param %8.1f%s"
                                  (name case) (name shape) n
                                  (:run row) (:first row) (:mid row) (:last row) (:param row)
                                  (if-let [g (:gibbs row)] (format "  gibbs %6.1f" g) "")))
                 (flush)
                 row))))
      (finally (ctx/stop-context! root)))))

(defn -main
  "`[sizes…]` (default 100 200 400 800)."
  [& sizes]
  (run (if (seq sizes) (mapv #(Long/parseLong %) sizes) [100 200 400 800]) 5)
  (shutdown-agents)
  (System/exit 0))
