(ns org.replikativ.foerster.counterfactual
  "Counterfactual queries by abduction, action and prediction (Pearl).

  A program's sample and observe sites are structural equations
  (`foerster.mechanism`). Given evidence and interventions:

  1. abduction — condition the program on the evidence (importance sampling
     by `gfi/generate`) and read each weighted factual trace's exogenous
     noise off its sites;
  2. action — install the interventions;
  3. prediction — run the program again in a world of its own with every
     other site fed its factual noise, aligned by address.

  The result pairs each factual world with its counterfactual twin, under
  the factual weight: a joint measure, so quantities that need both worlds
  (probability of necessity, effect of treatment on the treated) are
  expectations over it. A site that exists only in the counterfactual world
  draws fresh noise and is listed as `:unaligned`: its answer is
  interventional, not counterfactual."
  (:require [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.mechanism :as mech]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]))

(defn- scored-entries [t]
  (if (:trace/order t)
    (itrace/entries t)
    (mapv (fn [[address e]]
            {:address address :value (:value e) :site itrace/choose-site
             :note {:dist (:distribution e)}})
          (remove (comp :deterministic? val) t))))

(defn noise-of
  "{address u} of the sample and observe sites of `trace` whose law is a
  mechanism, and the addresses of those that are not."
  [trace]
  (reduce (fn [[noise unsupported] {:keys [address value note site]}]
            (if (= itrace/choose-site site)
              (if (and (:dist note) (mech/mechanism? (:dist note)))
                [(assoc noise address
                        (if (:trace/world trace)
                          (random/in-world-stream (:trace/world trace) [::abduct address]
                                                  #(mech/abduct (:dist note) value))
                          (mech/abduct (:dist note) value)))
                 unsupported]
                [noise (conj unsupported address)])
              [noise unsupported]))
          [{} []]
          (scored-entries trace)))

(defn abduct-trace
  "Recover reusable noise from a live GFI trace. `weight` is a log weight;
  pass 0 for an already-posterior MCMC draw, not its likelihood."
  ([t] (abduct-trace t (itrace/log-weight t)))
  ([t weight]
   (let [[noise unsupported] (noise-of t)]
     {:factual (:trace/result t) :noise noise :unsupported unsupported :weight weight})))

(defn abduct-posterior
  "Abduct every draw of an EmpiricalMeasure, retaining its existing weights.
  Accepts the legacy traces kept by SMC and MCMC. :reconstruct may instead
  map a particle to {:noise … :unsupported […] :factual …}; this is the
  explicit adapter for blocks/compiled models whose mechanisms are hidden.
  :seed makes posterior discrete-noise draws reproducible. No live worlds
  are retained."
  ([posterior] (abduct-posterior posterior nil))
  ([posterior {:keys [reconstruct seed]}]
   (let [seed (or seed (random/fresh-seed))]
     (mapv (fn [i [particle weight]]
             (let [a (random/with-stream* seed [::posterior-abduct i]
                       #(if reconstruct
                          (reconstruct particle)
                          (let [[noise unsupported] (noise-of (m/get-trace particle))]
                            {:factual (m/get-value particle) :noise noise :unsupported unsupported})))]
               (when-not (and (map? (:noise a)) (sequential? (:unsupported a)))
                 (throw (ex-info "Abduction needs :noise and :unsupported"
                                 {:type ::invalid-abduction :particle i})))
               (assoc a :weight weight)))
           (range) (m/get-particles posterior)))))

(defn- then [operation f]
  (fn [resolve reject]
    (operation (fn [x] (try (f x resolve reject)
                            (catch #?(:clj Throwable :cljs :default) e (reject e))))
               reject)))

(defn- close-all
  "Close the sessions of `traces` in turn, then call `k` — also when a close
  fails, so a failure is reported with its own error."
  [traces k]
  (if-let [[t & more] (seq traces)]
    ((gfi/close! t) (fn [_] (close-all more k)) (fn [_] (close-all more k)))
    (k)))

(defn predict
  "Replay `model` under `interventions` using a reusable abducted draw.
  Returns CPS resolving {:factual :counterfactual :weight :unaligned :status}.
  :strict? defaults true: incomplete coupling is refused. With false, fresh
  noise is explicitly reported; this makes no claim of full coupling.
  :run-opts supplies GFI executor/session options. Sessions are always closed."
  ([model abduction interventions] (predict model abduction interventions nil))
  ([model {:keys [noise unsupported factual weight]} interventions
    {:keys [strict? run-opts] :or {strict? true}}]
   (fn [resolve reject]
     (if-not (map? noise)
       (reject (ex-info "Abduction needs a noise map" {:type ::invalid-abduction}))
       ((gfi/run-policy model (itrace/policy {:noise noise :interventions interventions :strict? strict?}) run-opts)
        (fn [t]
          (try
            (let [unaligned (vec (distinct (map :address (filter (comp :unaligned? :note)
                                                                 (itrace/entries t)))))
                  result {:factual factual :counterfactual (:trace/result t)
                          :weight weight :unaligned unaligned
                          :unsupported-factual (vec unsupported)
                          :status (if (seq unaligned) :partial :complete)}]
              (close-all [t] #(resolve result)))
            (catch #?(:clj Throwable :cljs :default) e (close-all [t] #(reject e)))))
        reject)))))

(defn worlds
  "Evaluate ordered [name intervention-spec] pairs against ONE abducted draw.
  A spec is an intervention map or a function of already computed worlds;
  :factual is available initially. This expresses nested counterfactuals,
  e.g. use a mediator from world :untreated in a later :direct world.
  Resolves {:worlds {name result …} :weight :unaligned :status}."
  ([model abduction specs] (worlds model abduction specs nil))
  ([model abduction specs opts]
   (fn [resolve reject]
     (letfn [(step [remaining values unaligned]
               (if-let [[[name spec] & more] (seq remaining)]
                 (try
                   (when (contains? values name)
                     (throw (ex-info "World names must be unique" {:type ::duplicate-world :name name})))
                   ((predict model abduction (if (fn? spec) (spec values) spec) opts)
                    (fn [r] (step more (assoc values name (:counterfactual r))
                                  (into unaligned (:unaligned r)))) reject)
                   (catch #?(:clj Throwable :cljs :default) e (reject e)))
                 (resolve {:worlds values :weight (:weight abduction)
                           :unaligned (vec (distinct unaligned))
                           :status (if (seq unaligned) :partial :complete)})))]
       (step specs {:factual (:factual abduction)} [])))))

(defn condition-worlds
  "Condition a computed world population on an arbitrary cross-world event.
  Retain existing log weights. A zero-mass event is an error, not a uniform
  posterior. This is Monte Carlo conditioning, not causal identification."
  [population event]
  (let [kept (filterv #(event (:worlds %)) population)]
    (when-not (some #(> (:weight %) ##-Inf) kept)
      (throw (ex-info "No posterior mass satisfies the cross-world event"
                      {:type ::impossible-evidence})))
    kept))

(defn expectation
  "Weighted expectation of numeric `f` on a computed draw (pair or worlds).
  Refuses an empty or zero-mass population."
  [population f]
  (when-not (every? #(and (number? (:weight %)) (<= ##-Inf (:weight %)) (< (:weight %) ##Inf)) population)
    (throw (ex-info "Posterior log weights must be finite or negative infinity"
                    {:type ::invalid-weight})))
  (when-not (some #(> (:weight %) ##-Inf) population)
    (throw (ex-info "A query needs positive posterior mass" {:type ::impossible-evidence})))
  (reduce + (map (fn [draw w] (* (f draw) w)) population
                 (m/normalize-log-weights (mapv :weight population)))))

(defn query
  "Compose factual inference, reusable abduction and multiple worlds.

  Supply exactly one of :posterior (an EmpiricalMeasure), :abductions
  (reusable draws), or :evidence (site values, importance inference with
  :particles, default 1). :worlds is an ordered vector of [name spec]; a
  spec may depend on previously computed worlds. Alternatively supply
  :interventions for one world named :counterfactual. :event optionally
  conditions on any predicate of the resulting world map.

  :strict? defaults true. :reconstruct and :seed go to abduct-posterior;
  `run-opts` goes to GFI. Resolves a vector of weighted world draws.
  This evaluates a specified causal model; it does not identify it from data."
  [model {:keys [posterior abductions evidence particles worlds interventions event
                 strict? reconstruct seed]
          :or {particles 1 strict? true} :as request} & [run-opts]]
  (fn [resolve reject]
    (try
      (when-not (= 1 (count (filter #(contains? request %) [:posterior :abductions :evidence])))
        (throw (ex-info "Supply exactly one factual source: :posterior, :abductions or :evidence"
                        {:type ::invalid-query})))
      (when-not (pos-int? particles)
        (throw (ex-info ":particles must be positive" {:type ::invalid-query})))
      (when (or (and (contains? request :posterior) (nil? posterior))
                (and (contains? request :abductions) (nil? abductions))
                (and (contains? request :evidence) (not (map? evidence)))
                (and worlds (not (sequential? worlds)))
                (and worlds (contains? request :interventions)))
        (throw (ex-info "Invalid factual source or ambiguous world specification" {:type ::invalid-query})))
      (let [specs (or worlds [[:counterfactual (or interventions {})]])
            options {:strict? strict? :run-opts run-opts}]
        (letfn [(finish [population]
                  (try
                    (let [population (if event (condition-worlds population event) population)]
                      ;; Enforce the same zero-mass contract even without a predicate.
                      (expectation population (constantly 0.0))
                      (resolve population))
                    (catch #?(:clj Throwable :cljs :default) e (reject e))))
                (run-draws [remaining out]
                  (if-let [[a & more] (seq remaining)]
                    (if (= ##-Inf (:weight a))
                      (run-draws more out)
                      ((org.replikativ.foerster.counterfactual/worlds model a specs options)
                       #(run-draws more (conj out %)) reject))
                    (finish out)))
                (infer-draws [i out]
                  (if (= i particles)
                    (run-draws out [])
                    ((gfi/generate model evidence run-opts)
                     (fn [{t :trace w :weight}]
                       (try
                         (let [a (when (> w ##-Inf) (abduct-trace t w))]
                           (close-all [t] #(infer-draws (inc i) (cond-> out a (conj a)))))
                         (catch #?(:clj Throwable :cljs :default) e
                           (close-all [t] #(reject e))))) reject)))]
          (cond
            posterior (run-draws (abduct-posterior posterior {:reconstruct reconstruct :seed seed}) [])
            abductions (run-draws abductions [])
            :else (infer-draws 0 []))))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

(defn- twin
  "One factual world and its counterfactual twin. Both sessions are closed
  however it ends."
  [model evidence interventions opts]
  (then (gfi/generate model evidence opts)
        (fn [{t :trace w :weight} resolve reject]
          (let [fail (fn [opened e] (close-all opened #(reject e)))]
            (try
              (let [[noise unsupported] (noise-of t)
                    factual (:trace/result t)]
                ((gfi/run-policy model (itrace/policy {:interventions interventions :noise noise}) opts)
                 (fn [t']
                   (try
                     (let [unaligned (into (vec unsupported)
                                           (comp (filter (comp :unaligned? :note)) (map :address))
                                           (itrace/entries t'))
                           out {:factual factual
                                :counterfactual (:trace/result t')
                                :weight w
                                :unaligned unaligned}]
                       (close-all [t' t] #(resolve out)))
                     (catch #?(:clj Throwable :cljs :default) e (fail [t' t] e))))
                 #(fail [t] %)))
              (catch #?(:clj Throwable :cljs :default) e (fail [t] e)))))))

(defn counterfactual
  "Pairs of factual and counterfactual results of `model` (a spin, as for
  `foerster.gfi`) given `:evidence` ({address value},
  fixed in the factual world) and `:interventions` ({selector transform}, as
  for `foerster.trace/policy`), from `:particles` importance-weighted
  factual worlds (default 1: enough when the evidence fixes every noise).

  Resolves a vector of {:factual r :counterfactual r' :weight log-w
  :unaligned [address …]}."
  [model {:keys [evidence interventions particles] :or {particles 1}} & [opts]]
  (fn [resolve reject]
    (letfn [(step [i acc]
              (if (= i particles)
                (resolve acc)
                ((twin model evidence interventions opts)
                 (fn [pair] (step (inc i) (conj acc pair)))
                 reject)))]
      (step 0 []))))
