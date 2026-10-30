(ns org.replikativ.foerster.program
  "Programmable weighted inference using ordinary Spin control flow.
  Targets share a representation and a fixed state space. Custom scores/proposals
  and invariant value transitions have explicit mathematical obligations; a
  result does not automatically certify an unbiased likelihood estimator."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.weights :as w]
            [org.replikativ.foerster.scope :as scope]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.transition :as transition]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.trace :as it]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await]]))

(defn- options! [opts allowed]
  (when-let [unknown (seq (remove allowed (keys opts)))]
    (throw (ex-info "Unsupported inference-program options"
                    {:type ::unknown-options :unknown (vec unknown)}))))

(defn- live! [scope]
  (when-not (= :open (:phase @(:resources scope)))
    (throw (ex-info "The inference scope is no longer live" {:type ::closed-scope}))))

(defn- population-size! [particles expected]
  (when-not (= (count particles) expected)
    (throw (ex-info "Representation returned the wrong population size"
                    {:type ::wrong-population-size :expected expected :actual (count particles)}))))

(defprotocol PRepresentation
  (initialize-particles [representation scope target n opts])
  (copy-particles [representation scope target particles ancestors opts])
  (move-particles [representation scope target particles kernel opts])
  (project-particle [representation target particle]))

(defn- sequential [xs f]
  (spin
   (loop [remaining (seq xs) out []]
     (if-not remaining
       out
       (let [value (await (f (first remaining)))]
         (recur (next remaining) (conj out value)))))))

(defrecord InferenceResult [measure target log-mass inference-contract history])

(defrecord InvariantValueKernel [target-id step])

(defn invariant-kernel
  "Bind a value transition to a target. step receives a value and returns
  a value. The author must establish invariance and fixed tuning independent of
  the population being moved and the offspring slot; this constructor does not prove either property."
  [target step]
  (when-not (ifn? step)
    (throw (ex-info "An invariant kernel needs a transition function" {:type ::invalid-kernel})))
  (->InvariantValueKernel (:id target) step))

(defrecord ValueRepresentation [draw log-proposal project copy]
  PRepresentation
  (initialize-particles [_ _ target n {:keys [seed]}]
    (fn [resolve reject]
      (try
        (resolve
         (mapv (fn [i]
                 (random/with-stream* seed [::initialize i]
                   #(let [x (draw)]
                      {:particle x :log-weight (- ((:score target) x) (log-proposal x))})))
               (range n)))
        (catch #?(:clj Throwable :cljs :default) e (reject e)))))
  (copy-particles [_ _ _ particles ancestors _]
    (fn [resolve _] (resolve (mapv #(copy (nth particles %)) ancestors))))
  (move-particles [_ _ target particles kernel {:keys [seed step]}]
    (fn [resolve reject]
      (try
        (when-not (and (instance? InvariantValueKernel kernel) (= (:id target) (:target-id kernel)))
          (throw (ex-info "An invariant value kernel must target this population" {:type ::wrong-kernel-target})))
        (resolve (mapv (fn [i x]
                         (random/with-stream* seed [::move step i] #((:step kernel) (copy x))))
                       (range) particles))
        (catch #?(:clj Throwable :cljs :default) e (reject e)))))
  (project-particle [_ _ particle] (m/sample-particle (project particle) {})))

(defn value-target
  "A target over persistent values. draw and log-proposal must describe the
  same normalized proposal, covering the target's support. score is the log
  unnormalized target, including any constants needed for evidence."
  [score draw log-proposal & [{:keys [label project copy] :or {project identity copy identity} :as opts}]]
  (options! opts #{:label :project :copy})
  (when-not (every? ifn? [score draw log-proposal project copy])
    (throw (ex-info "Value targets require a scored proposal, projection and copy functions"
                    {:type ::unscored-proposal})))
  (let [id (random-uuid)]
    (merge (dissoc opts :project :copy) {:id id :family id :label label :position 0 :score score
                                         :representation (->ValueRepresentation draw log-proposal project copy)})))

(defn- trace-score [beta t]
  (transduce (map (fn [{:keys [note]}]
                    (if (contains? note :log-lik)
                      (if (zero? beta) 0.0 (* beta (:log-lik note)))
                      (:log-prob note)))) + 0.0 (it/entries t)))

(defrecord TraceRepresentation [model executor]
  PRepresentation
  (initialize-particles [_ resources target n {:keys [seed]}]
    (sequential (range n)
                (fn [i]
                  (let [operation (:operation (gfi/run-policy* model (it/policy {:temperature (:position target)})
                                                               (cond-> {:inference-scope resources
                                                                        :seed (sp/derive-seed seed ::initialize i)}
                                                                 executor (assoc :executor executor))))]
                    (fn [resolve reject]
                      (operation #(resolve {:particle % :log-weight (it/log-weight %)}) reject))))))
  (copy-particles [_ _ target particles ancestors {:keys [seed step]}]
    (sequential (map-indexed vector ancestors)
                (fn [[i a]]
                  (let [t (nth particles a)
                        from (first (it/latent-addresses t))]
                    (if-not from
                      (fn [resolve _] (resolve t))
                      (fn [resolve reject]
                        ((trace/replay t from (it/policy {:keep? true :temperature (:position target)})
                                       {:anchor? it/anchor? :seed (sp/derive-seed seed ::copy [step i])})
                         (fn [copy]
                           (if-let [e (:trace/error copy)] (reject e) (resolve copy)))
                         reject)))))))
  (move-particles [_ _ target particles kernel {:keys [step]}]
    ;; More block/chain kernels can be admitted once their coordinate, warm-up
    ;; and tempered-target contracts are represented. Never silently use beta=1.
    (when-not (and (k/markov-kernel? kernel) (= :single-site-mh (k/transition-id kernel)))
      (throw (ex-info "Trace population moves currently require single-site MH" {:type ::unsupported-kernel})))
    (let [{:keys [iterations] :as compiled} (transition/options kernel)]
      (sequential particles
                  (fn [t]
                    (fn [resolve reject]
                      ((it/mh-chain t iterations
                                    (assoc compiled :temperature (:position target) :keep-old? true
                                           :shared-anchors? true))
                       #(resolve (:trace %)) reject))))))
  (project-particle [_ _ t]
    (m/sample-particle (:trace/result t) (it/legacy-trace t))))

(defn trace-target
  "A complete-trace target p(choices)*L(choices)^beta, not a streaming prefix.
  Only fresh worlds are supported here. Model effects obey the Spin discipline."
  [model & [{:keys [beta executor label world-policy] :or {beta 0.0 world-policy :fresh} :as opts}]]
  (options! opts #{:beta :executor :label :world-policy})
  (when-not (= :fresh world-policy)
    (throw (ex-info "Trace population programs currently use fresh worlds" {:type ::unsupported-world-policy})))
  (when-not (and (number? beta) (<= 0.0 beta 1.0))
    (throw (ex-info "Temperature must be in [0,1]" {:type ::invalid-temperature})))
  (let [id (random-uuid)]
    {:id id :family id :label label :position beta :score #(trace-score beta %)
     :representation (->TraceRepresentation model executor)}))

(defn- derive-target [target position score]
  (assoc target :id (random-uuid) :parent (:id target) :position position :score score))

(defn retarget
  "Derive a value target in the same state space. Position is a label, not
  target identity. Changing support requires a valid proposal/transport
  construction. Trace targets must use temperature, not arbitrary scores."
  [target position score]
  (when-not (instance? ValueRepresentation (:representation target))
    (throw (ex-info "Generic retargeting requires a score-driven value representation"
                    {:type ::unsupported-retarget})))
  (derive-target target position score))

(defn temperature [target beta]
  (when-not (and (instance? TraceRepresentation (:representation target))
                 (number? beta) (<= (:position target) beta 1.0))
    (throw (ex-info "Trace temperatures advance monotonically to one" {:type ::invalid-temperature})))
  (derive-target target beta #(trace-score beta %)))

(defn run
  "Run (body scope) with a root seed and an asynchronous resource lifetime.
  Scope is passed explicitly; nested programs receive independent lifetimes."
  ([body] (run {} body))
  ([opts body]
   (options! opts #{:seed :world-policy})
   (when-not (contains? #{nil :fresh} (:world-policy opts))
     (throw (ex-info "This program representation does not import caller systems" {:type ::unsupported-world-policy})))
   (scope/run (fn [resources]
                (spin
                 (let [result (await (body {:resources resources :seed (or (:seed opts) (random/fresh-seed))
                                            :owner (random-uuid)
                                            :population-counter (atom -1)
                                            :population-ids (atom #{})}))]
                   (when-not (instance? InferenceResult result)
                     (throw (ex-info "Finish the population before returning from inference"
                                     {:type ::unfinished-population})))
                   result))))))

(defn- owned! [scope population]
  (when-not (= (:owner scope) (:owner population))
    (throw (ex-info "A population belongs to another inference scope" {:type ::wrong-scope})))
  (live! scope))

(defn initialize
  "Initialize an independent population. Optional :id names its logical RNG
  stream for concurrent programs; a name may be used once per scope. Default
  identities follow sequential initialization order."
  ([scope target n] (initialize scope target n {}))
  ([scope target n {:keys [id] :as opts}]
   (options! opts #{:id})
   (when-not (pos-int? n)
     (throw (ex-info "A population needs positive size" {:type ::invalid-particles})))
   (spin
    (let [_ (live! scope)
          population-id (if (some? id) [:named id]
                            [:automatic (swap! (:population-counter scope) inc)])
          _ (swap! (:population-ids scope)
                   (fn [ids]
                     (when (contains? ids population-id)
                       (throw (ex-info "A population RNG identity is already used"
                                       {:type ::duplicate-population-id :id population-id})))
                     (conj ids population-id)))
          seed (sp/derive-seed (:seed scope) ::population population-id)
          draws (await (initialize-particles (:representation target) (:resources scope) target n
                                             (assoc scope :seed seed)))
          _ (population-size! draws n)]
      {:target target :particles (mapv :particle draws) :mass (w/from-log (mapv :log-weight draws))
       :owner (:owner scope) :population-id population-id :seed seed :step 0 :history []
       :inference-contract {:semantics :weighted-population
                            :weight-contract :requires-operation-assumptions
                            :normalizer {:status :unverified}}}))))

(defn advance
  "Reweight on an explicitly derived target in the same state space. Adaptive
  target selection needs its own justification; no unbiasedness is inferred."
  [scope population target]
  (owned! scope population)
  (when-not (and (= (:id (:target population)) (:parent target))
                 (= (:family (:target population)) (:family target))
                 (= (:representation (:target population)) (:representation target)))
    (throw (ex-info "Advance requires a target derived from the current one" {:type ::wrong-target})))
  (let [score (:score target) old-score (:score (:target population))
        increments (mapv (fn [x lw] (if (= ##-Inf lw) 0.0 (- (score x) (old-score x))))
                         (:particles population) (get-in population [:mass :log-weights]))]
    (-> population (assoc :target target :mass (w/reweight (:mass population) increments))
        (update :step inc) (update :history conj {:operation :advance :target (:id target)}))))

(defn ess [population]
  (if (= :zero-mass (get-in population [:mass :status])) 0.0
      (m/compute-ess (w/selection (:mass population)))))

(defn resample
  ([scope population] (resample scope population {}))
  ([scope population {:keys [particles method] :or {method :systematic} :as opts}]
   (options! opts #{:particles :method})
   (owned! scope population)
   (let [n (or particles (count (:particles population)))
         _ (when-not (pos-int? n)
             (throw (ex-info "Resampling needs a positive population size" {:type ::invalid-particles})))
         selection (w/selection (:mass population))
         ancestors (random/with-stream* (:seed population) [::resample (:step population)]
                     #(m/resample method selection n))]
     (spin
      (let [copies (await (copy-particles (get-in population [:target :representation]) (:resources scope)
                                          (:target population) (:particles population) ancestors
                                          (assoc scope :seed (:seed population) :step (:step population))))
            _ (population-size! copies n)]
        (-> population (assoc :particles copies :mass (w/resampled (:mass population) n))
            (update :step inc) (update :history conj {:operation :resample :ancestors ancestors})))))))

(defn move [scope population kernel]
  (owned! scope population)
  (w/selection (:mass population))
  (spin
   (let [moved (await (move-particles (get-in population [:target :representation]) (:resources scope)
                                      (:target population) (:particles population) kernel
                                      (assoc scope :seed (:seed population) :step (:step population))))
         _ (population-size! moved (count (:particles population)))]
     (-> population (assoc :particles moved) (update :step inc)
         (update :history conj {:operation :move})))))

(defn finish
  "Project settled values and trace data. Scope.run releases all live resources
  before delivering these results; log mass is not an unbiasedness certificate."
  [scope population]
  (owned! scope population)
  (let [{:keys [target particles mass]} population
        logs (:log-weights mass)
        ;; EmpiricalMeasure's legacy convention is log-mean, not log-sum.
        measure (assoc (m/empirical (mapv (fn [x lw] [(project-particle (:representation target) target x)
                                                      (+ lw (Math/log (count particles)))]) particles logs))
                       :log-normalizer (:log-mass mass)
                       :inference-contract (:inference-contract population))]
    (->InferenceResult measure (dissoc target :score :representation)
                       (:log-mass mass) (:inference-contract population) (:history population))))
