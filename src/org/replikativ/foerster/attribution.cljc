(ns org.replikativ.foerster.attribution
  "Attribution and repair search over declared counterfactual mechanisms.
  A baseline and an outcome score are part of the question. Contributions
  are allocations of that score, not probabilities that a node is a cause."
  (:require [clojure.set :as set]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.random :as random]))

(defn- subsets [nodes]
  (reduce (fn [ss v] (into ss (map #(conj % v) ss))) [#{}] nodes))

(defn- factorial [n] (reduce * 1.0 (range 1 (inc n))))

(defn- shuffle-nodes [nodes]
  (loop [v (vec nodes) i (dec (count nodes))]
    (if (pos? i)
      (let [j (int (* (random/uniform01) (inc i))) a (nth v i) b (nth v j)]
        (recur (assoc v i b j a) (dec i))) v)))

(defn shapley
  "Allocate `value` of a coalition (set of nodes). Exact or sampled
  permutation Shapley values. Memoizes coalition evaluations and enforces
  :max-evaluations (default 65536). :seed makes sampling reproducible.
  A budget failure is an exception, never a completed attribution."
  ([nodes value] (shapley nodes value nil))
  ([nodes value {:keys [method permutations seed max-evaluations]
                 :or {method :exact permutations 128 max-evaluations 65536}}]
   (let [nodes (vec nodes) n (count nodes) cache (atom {})
         v (fn [s]
             (if (contains? @cache s) (get @cache s)
                 (do (when (>= (count @cache) max-evaluations)
                       (throw (ex-info "Attribution evaluation budget exceeded" {:type ::budget})))
                     (let [x (value s)]
                       (when-not (and (number? x) (< ##-Inf x ##Inf))
                         (throw (ex-info "Coalition value must be finite" {:type ::invalid-score :coalition s})))
                       (swap! cache assoc s x) x))))]
     (when-not (= n (count (set nodes)))
       (throw (ex-info "Attribution nodes must be unique" {:type ::duplicate-nodes})))
     (when-not (and (#{:exact :permutation} method) (pos-int? permutations))
       (throw (ex-info "Use :exact or :permutation with positive :permutations" {:type ::invalid-options})))
     (when (and (= method :exact) (> n 16))
       (throw (ex-info "Exact Shapley is limited to 16 nodes; use :permutation" {:type ::budget})))
     (let [contributions
           (if (= method :exact)
             (into {} (for [node nodes]
                        [node (reduce + 0.0
                                      (for [s (subsets (remove #{node} nodes))
                                            :let [k (count s)
                                                  w (/ (* (factorial k) (factorial (- n k 1))) (factorial n))]]
                                        (* w (- (v (conj s node)) (v s)))))]))
             (random/with-stream* (or seed (random/fresh-seed)) ::shapley
               #(let [totals (reduce (fn [acc _]
                                       (loop [order (seq (shuffle-nodes nodes)) s #{} acc acc]
                                         (if-let [[node & more] order]
                                           (recur more (conj s node)
                                                  (update acc node + (- (v (conj s node)) (v s)))) acc)))
                                     (zipmap nodes (repeat 0.0)) (range permutations))]
                  (into {} (map (fn [[node total]] [node (/ total permutations)]) totals)))))]
       {:contributions contributions :baseline (v #{}) :total (v (set nodes))
        :method method :evaluations (count @cache)}))))

(defn attribute-noise
  "Exact noise-coalition attribution for ONE abducted incident.
  `baselines` is a nonempty vector of noise maps from a declared normal
  reference population. Selected `nodes` retain factual noise; the others
  receive baseline noise. Other sites keep the incident's noise. `score`
  maps a predicted result to a number. Resolves a Shapley report, averaged
  over the supplied baseline bank, with the incident's :weight retained.
  Requires complete replay coupling; at most :max-nodes (default 10)."
  [model abduction baselines {:keys [nodes score max-nodes] :or {max-nodes 10}}]
  (fn [resolve reject]
    (try
      (let [nodes (vec nodes) all (set nodes)]
        (when-not (and (ifn? score) (seq baselines) (<= (count nodes) max-nodes)
                       (= (count nodes) (count all))
                       (every? #(every? (fn [node] (contains? % node)) nodes) baselines)
                       (every? #(contains? (:noise abduction) %) nodes))
          (throw (ex-info "Noise attribution needs unique supported nodes, complete baselines and a score"
                          {:type ::invalid-attribution})))
        (letfn [(coalitions [remaining values]
                  (if-let [[s & more] (seq remaining)]
                    (letfn [(bank [remaining total n]
                              (if-let [[baseline & rest] (seq remaining)]
                                (let [noise (merge (:noise abduction)
                                                   (select-keys baseline (set/difference all s)))]
                                  ((cf/predict model (assoc abduction :noise noise) {})
                                   (fn [r]
                                     (try (bank rest (+ total (score (:counterfactual r))) (inc n))
                                          (catch #?(:clj Throwable :cljs :default) e (reject e)))) reject))
                                (coalitions more (assoc values s (/ total n)))))]
                      (bank baselines 0.0 0))
                    (resolve (assoc (shapley nodes values) :weight (:weight abduction)
                                    :baseline-count (count baselines)))))]
          (coalitions (subsets nodes) {})))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

(defn repairs
  "Evaluate finite named candidate interventions against an abducted
  posterior. `candidates` maps name -> interventions, `success?` judges a
  predicted result. Resolves reports sorted by decreasing success
  probability, optionally including :cost from a function of candidate name.
  Candidates can include joint repairs; this is enumeration, not a claim
  of minimality or a Halpern-Pearl actual-causality decision."
  [model abductions candidates success? & [{:keys [cost]}]]
  (fn [resolve reject]
    (letfn [(candidate [remaining reports]
              (if-let [[[name interventions] & more] (seq remaining)]
                (letfn [(draws [remaining results]
                          (if-let [[a & rest] (seq remaining)]
                            ((cf/predict model a interventions)
                             (fn [r] (draws rest (conj results r))) reject)
                            (try
                              (candidate more (conj reports
                                                    (cond-> {:candidate name
                                                             :success-probability
                                                             (cf/expectation results #(if (success? (:counterfactual %)) 1.0 0.0))}
                                                      cost (assoc :cost (cost name)))))
                              (catch #?(:clj Throwable :cljs :default) e (reject e)))))]
                  (draws abductions []))
                (resolve (vec (sort-by (comp - :success-probability) reports)))))]
      (candidate candidates []))))

(defn attribute-change
  "Attribute a change between two mechanism maps. `value` evaluates a mixed
  mechanism map, e.g. the mean outcome of a simulator. The coalition uses
  new mechanisms for its nodes and old mechanisms elsewhere. Both maps
  must have identical keys. Uses `shapley`'s exact/sampling options.
  Fitting mechanisms and choosing a distribution discrepancy remain explicit."
  [old new value & [opts]]
  (when-not (= (set (keys old)) (set (keys new)))
    (throw (ex-info "Old and new models need the same named mechanisms"
                    {:type ::incompatible-models})))
  (shapley (sort-by pr-str (keys old))
           #(value (merge old (select-keys new %))) opts))

(defn actual-cause
  "Modified Halpern-Pearl actual causality in a FINITE deterministic SCM.
  `evaluate` maps interventions {node value} to all endogenous values under
  ONE fixed exogenous realization. `effect?` is true on the factual result.
  `candidate` is the set of actual events under test. `domains` gives the
  alternative values allowed at candidate nodes. Witness nodes may be held
  at their factual values; by default every other endogenous node is eligible.

  Exhaustive AC2 witness search plus AC3 subset minimality. A bias toward
  small sets is not substituted for the minimality check. Returns :actual-cause,
  :not-cause, :not-minimal, or :unresolved when :max-evaluations is exhausted.
  :witnesses restricts the allowed context; :max-nodes (default 12) limits
  combinatorics. This is not a discovery algorithm or a posterior probability."
  [evaluate effect? candidate domains & [{:keys [witnesses max-evaluations max-nodes]
                                          :or {max-evaluations 65536 max-nodes 12}}]]
  (let [candidate (set candidate) factual (evaluate {})
        nodes (set (keys factual)) witnesses (set (or witnesses nodes))
        cache (atom {{} factual})
        run (fn [interventions]
              (if (contains? @cache interventions) (get @cache interventions)
                  (do (when (>= (count @cache) max-evaluations)
                        (throw (ex-info "Actual-cause search budget exhausted" {:type ::budget})))
                      (let [result (evaluate interventions)]
                        (swap! cache assoc interventions result) result))))]
    (when-not (and (seq candidate) (set/subset? candidate nodes)
                   (set/subset? witnesses nodes) (effect? factual)
                   (pos-int? max-evaluations) (pos-int? max-nodes)
                   (every? #(and (vector? (domains %)) (seq (domains %))
                                 (some (fn [x] (= x (factual %))) (domains %))) candidate))
      (throw (ex-info "Actual causality needs factual events, finite candidate domains and an actual effect"
                      {:type ::invalid-causality-query})))
    (if (> (count nodes) max-nodes)
      {:status :unresolved :reason :node-budget :evaluations 1}
      (letfn [(alternatives [remaining]
                (if-let [[v & more] (seq remaining)]
                  (for [a (alternatives more) x (domains v) :when (not= x (factual v))]
                    (assoc a v x)) [{}]))
              (witness [c]
                (first (for [held (subsets (sort-by pr-str (set/difference witnesses c)))
                             alternative (alternatives (sort-by pr-str c))
                             :let [interventions (merge (select-keys factual held) alternative)]
                             :when (not (effect? (run interventions)))]
                         {:held (select-keys factual held) :alternative alternative})))]
        (try
          (if-let [w (witness candidate)]
            (if-let [smaller (first (for [s (subsets (sort-by pr-str candidate))
                                          :when (and (seq s) (not= s candidate))
                                          :let [sw (witness s)] :when sw]
                                      {:candidate s :witness sw}))]
              {:status :not-minimal :smaller smaller :evaluations (count @cache)}
              {:status :actual-cause :witness w :evaluations (count @cache)})
            {:status :not-cause :evaluations (count @cache)})
          (catch #?(:clj clojure.lang.ExceptionInfo :cljs ExceptionInfo) e
            (if (= ::budget (:type (ex-data e)))
              {:status :unresolved :reason :evaluation-budget :evaluations (count @cache)}
              (throw e))))))))
