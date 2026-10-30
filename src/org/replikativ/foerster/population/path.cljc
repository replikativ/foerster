(ns org.replikativ.foerster.population.path
  "One population-path interpreter over representation-specific batch phases.
  Value phases may complete synchronously and contain fused numeric loops;
  trace phases may suspend. Both use the same weighting/selection/retention."
  (:require [is.simm.partial-cps.async :as pcps]
            [is.simm.partial-cps.runtime :as cps-runtime]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

(defn reweight [population increments]
  (update population :mass weights/reweight increments))

(defn selection-plan
  "Statistical selection only. Representation ownership/copy follows selected.
  Normalized weights are reset while total mass is retained."
  [population n {:keys [method seed key] :or {method :systematic}}]
  (when-not (pos-int? n)
    (throw (ex-info "Selection requires positive population size" {:type ::invalid-size :size n})))
  (let [selection (weights/selection (:mass population))
        ancestors (random/with-stream* seed key #(m/resample method selection n))]
    {:ancestors ancestors :mass (weights/resampled (:mass population) n)}))

(defn selected [population plan particles]
  (when-not (= (count particles) (count (:ancestors plan)))
    (throw (ex-info "Selection returned the wrong population size" {:type ::invalid-selection-size})))
  (assoc population :particles (vec particles) :mass (:mass plan)))

(defn moved
  "A cardinality-preserving invariant population move retains its weights."
  [population particles]
  (when-not (= (count particles) (count (:particles population)))
    (throw (ex-info "Move returned the wrong population size" {:type ::invalid-move-size})))
  (assoc population :particles (vec particles)))

(defn retain
  "Retain a chain per parent, optionally including its initial state. Split
  each parent's normalized mass among its outputs; uniform equal-length chains
  retain canonical uniform weights (the waste-free SMC case)."
  [population chains retain-initial?]
  (let [parents (:particles population)
        _ (when-not (= (count parents) (count chains))
            (throw (ex-info "One chain is required per selected parent" {:type ::invalid-chain-count})))
        groups (mapv (fn [parent chain] (if retain-initial? (into [parent] chain) (vec chain))) parents chains)
        counts (mapv count groups)]
    (when (some zero? counts)
      (throw (ex-info "A retained chain must contribute a state" {:type ::empty-chain})))
    (let [particles (into [] cat groups)
          parent-logs (get-in population [:mass :log-weights])
          mass (if (and (apply = counts) (apply = parent-logs))
                 (weights/resampled (:mass population) (count particles))
                 (assoc (:mass population) :log-weights
                        (into [] (mapcat (fn [lw n] (repeat n (- lw (Math/log n)))) parent-logs counts))))]
      (assoc population :particles particles :mass mass))))

(defn run
  "CPS interpreter. initial has :particles/:mass/:position. path has :done?,
  :advance [particles normalized-log-weights position] -> {:pos :log-incr}, and
  :exhausted. backend :move [starts position step moves] returns CPS resolving
  {:chains [states-per-parent] :seen [...]}; optional :select [particles indices]
  and :reclaim [old seen live] are CPS batch operations. opts controls :n,
  :waste-free, :seed, :keys {:resample role}, :max-steps and :positions?."
  [initial path backend {:keys [n waste-free seed keys max-steps positions?]
                         :or {max-steps 1000 positions? true}}]
  (let [n (or n (count (:particles initial)))
        chains (if waste-free (quot n waste-free) n)
        moves (if waste-free (dec waste-free) 1)
        {:keys [done? advance exhausted]} path
        select (or (:select backend) (fn [particles ancestors]
                                       (fn [resolve _] (resolve (mapv particles ancestors)))))
        reclaim (or (:reclaim backend) (fn [_ _ _] (fn [resolve _] (resolve nil))))]
    (pcps/async
     (loop [state (assoc initial :positions [(:position initial)] :steps 0)]
       (cond
         (done? (:position state)) state
         (>= (:steps state) max-steps) (throw (exhausted (:position state)))
         :else
         (let [old (:particles state)
               k (:steps state)
               {next-position :pos increments :log-incr}
               (advance old (get-in state [:mass :log-weights]) (:position state))
               state (assoc (reweight state increments) :position next-position)
               plan (selection-plan state chains {:seed seed :key [(:resample keys) k]})
               starts (pcps/await (select old (:ancestors plan)))
               state (selected state plan starts)
               result (pcps/await ((:move backend) starts next-position k moves))
               state (retain state (:chains result) (some? waste-free))]
           (pcps/await (reclaim old (:seen result) (:particles state)))
           (recur (-> state (update :steps inc)
                      (assoc :positions (if positions? (conj (:positions state) next-position)
                                            [next-position]))))))))))

(defn run-sync
  "Execute a known synchronous backend without blocking. Drive a returned
  Partial-CPS Thunk even under an enclosing trampoline; reject true suspension."
  [operation]
  (let [pending #?(:clj (Object.) :cljs (js-obj))
        outcome (volatile! pending)]
    (loop [result (operation #(vreset! outcome [:ok %]) #(vreset! outcome [:error %]))]
      (when (cps-runtime/thunk? result) (recur (cps-runtime/force-thunk result))))
    (when (identical? pending @outcome)
      (throw (ex-info "A synchronous population backend suspended" {:type ::asynchronous-backend})))
    (let [[status result] @outcome]
      (if (= :error status) (throw result) result))))
