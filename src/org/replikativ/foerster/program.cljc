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
            [org.replikativ.foerster.transition :as transition]
            [org.replikativ.foerster.population.path :as path]
            [org.replikativ.foerster.trace-owner :as owner]
            [org.replikativ.foerster.transport :as transport]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.trace :as it]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.protocols :as rtp]
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
  (transport-particles [representation source destination particles kernel opts])
  (project-particle [representation target particle]))

(defn- sequential [xs f]
  (spin
   (loop [remaining (seq xs) out []]
     (if-not remaining
       out
       (let [value (await (f (first remaining)))]
         (recur (next remaining) (conj out value)))))))

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
  (transport-particles [_ source destination particles kernel {:keys [seed step]}]
    (sequential (map-indexed vector particles)
                (fn [[i x]]
                  (fn [resolve reject]
                    (try (resolve (random/with-stream* seed [::transport step i]
                                    #(transport/value-step source destination kernel copy x)))
                         (catch #?(:clj Throwable :cljs :default) e (reject e)))))))
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
  (transduce (map (it/tempered-score beta)) + 0.0 (it/entries t)))

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
  (copy-particles [_ _ _target particles ancestors {:keys [seed step]}]
    (sequential (map-indexed vector ancestors)
                (fn [[i a]]
                  (let [t (nth particles a)]
                    (it/copy-complete t {:seed (sp/derive-seed seed ::copy [step i])})))))

  (move-particles [_ _ target particles _ {:keys [prepared on-visit]}]
    (sequential (map-indexed vector particles)
                (fn [[i t]]
                  (fn [resolve reject]
                    ((transition/step (nth prepared i) t
                                      {:temperature (:position target) :keep-old? true
                                       :shared-anchors? true :on-visit (when on-visit #(on-visit i %))})
                     #(resolve (:trace %)) reject)))))
  (transport-particles [this source destination particles kernel {:keys [seed step]}]
    (when-not (= this (:representation destination))
      (throw (ex-info "Trace transport must retain its model representation" {:type ::wrong-target})))
    (sequential (map-indexed vector particles)
                (fn [[i t]]
                  (let [proposal-seed (sp/derive-seed seed ::transport [step i])]
                    (fn [resolve reject]
                      ((transport/trace-step source destination kernel t proposal-seed)
                       resolve reject))))))
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
                 (let [context {:resources resources :seed (or (:seed opts) (random/fresh-seed))
                                :owner (random-uuid) :population-counter (atom -1)
                                :population-ids (atom #{}) :live-populations (atom {}) :retired (atom [])}
                       result (await (body context))]
                   (when-not (satisfies? m/PMeasure result)
                     (throw (ex-info "Finish the population before returning from inference"
                                     {:type ::unfinished-population})))
                   result))))))

(defn- owned! [scope population]
  (when-not (= (:owner scope) (:owner population))
    (throw (ex-info "A population belongs to another inference scope" {:type ::wrong-scope})))
  (live! scope))

(defn- claim! [scope population]
  (owned! scope population)
  (when-let [lease (:lease population)]
    (swap! lease
           (fn [state]
             (when-not (and (= :idle (:phase state)) (= (:version population) (:version state)))
               (throw (ex-info "Trace population is stale or already in flight" {:type ::stale-population})))
             (assoc state :phase :in-flight)))))

(defn- remember! [scope population]
  (when (:lease population)
    (swap! (:live-populations scope) assoc (:population-id population) (:particles population)))
  population)

(defn- publish [scope previous next]
  (let [next (if-let [lease (:lease previous)]
               (let [version (inc (:version previous))]
                 (reset! lease {:phase :idle :version version})
                 (assoc next :version version)) next)]
    (remember! scope next)))

(defn- reclaim [scope previous seen next]
  (remember! scope next)
  (if (:lease previous)
    (let [[retired _] (swap-vals! (:retired scope) (constantly []))]
      (owner/reclaim (into (:particles previous) retired) seen
                     (into [] cat (vals @(:live-populations scope)))))
    (fn [resolve _] (resolve nil))))

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
      (remember! scope
                 {:target target :particles (mapv :particle draws) :mass (w/from-log (mapv :log-weight draws))
                  :owner (:owner scope) :population-id population-id :seed seed :step 0 :history []
                  :inference-contract {:semantics :weighted-population
                                       :weight-contract :requires-operation-assumptions
                                       :normalizer {:status :unverified}}
                  :lease (when (instance? TraceRepresentation (:representation target))
                           (atom {:phase :idle :version 0})) :version 0 :transitions {} :forks (atom 0)})))))

(defn advance
  "Reweight in the same state space. A trace population version is consumed;
  target adaptation needs its own statistical justification."
  [scope population target]
  (claim! scope population)
  (when-not (and (= (:id (:target population)) (:parent target))
                 (= (:family (:target population)) (:family target))
                 (= (:representation (:target population)) (:representation target)))
    (throw (ex-info "Advance requires a target derived from the current one" {:type ::wrong-target})))
  (let [score (:score target) old-score (:score (:target population))
        increments (mapv (fn [x lw] (if (= ##-Inf lw) 0.0 (- (score x) (old-score x))))
                         (:particles population) (get-in population [:mass :log-weights]))]
    (publish scope population
             (-> (path/reweight population increments) (assoc :target target)
                 (update :step inc) (update :history conj {:operation :advance :target (:id target)})))))

(defn ess [population]
  (if (= :zero-mass (get-in population [:mass :status])) 0.0
      (m/compute-ess (w/selection (:mass population)))))

(defn resample
  ([scope population] (resample scope population {}))
  ([scope population {:keys [particles method] :or {method :systematic} :as opts}]
   (options! opts #{:particles :method})
   (spin
    (claim! scope population)
    (let [n (or particles (count (:particles population)))
          plan (path/selection-plan population n {:method method :seed (:seed population)
                                                  :key [::resample (:step population)]})
          copies (await (copy-particles (get-in population [:target :representation]) (:resources scope)
                                        (:target population) (:particles population) (:ancestors plan)
                                        (assoc scope :seed (:seed population) :step (:step population))))
          next (-> (path/selected population plan copies) (assoc :transitions {})
                   (update :step inc) (update :history conj {:operation :resample :ancestors (:ancestors plan)}))]
      (await (reclaim scope population [] next))
      (publish scope population next)))))

(defn- visit-trace!
  "Replace only the visited slot. Consumed intermediate states can then be
  reclaimed immediately, while other slots and independent forks remain live."
  [scope population current i result]
  (let [next (:trace result)
        previous (nth @current i)]
    ;; A completed replay no longer needs its old world as a reuse source.
    (rtp/swap-state! (:trace/world next) [:engine/reuse-source] (constantly nil))
    (swap! current assoc i next)
    (swap! (:live-populations scope) assoc (:population-id population) @current)
    (owner/reclaim! [previous]
                    (into [] cat (vals @(:live-populations scope))))))

(defn move [scope population kernel]
  (spin
   (claim! scope population)
   (w/selection (:mass population))
   (let [traced? (some? (:lease population))
         prepared (when traced?
                    (or (get (:transitions population) kernel)
                        (mapv (fn [_] (transition/prepare kernel)) (:particles population))))
         _ (when (some transition/adaptation? prepared)
             (throw (ex-info "Population transitions require frozen tuning; run chain warm-up separately"
                             {:type ::adaptive-population-kernel})))
         current (when traced? (atom (:particles population)))
         moved (await (move-particles (get-in population [:target :representation]) (:resources scope)
                                      (:target population) (:particles population) kernel
                                      (assoc scope :seed (:seed population) :step (:step population)
                                             :prepared prepared
                                             :on-visit (when traced?
                                                         #(visit-trace! scope population current %1 %2)))))
         next (cond-> (-> (path/moved population moved) (update :step inc)
                          (update :history conj {:operation :move}))
                traced? (assoc-in [:transitions kernel] prepared))]
     (await (reclaim scope population [] next))
     (publish scope population next))))

(def transport-kernel transport/kernel)

(defn transport
  "Propose and reweight through scored forward/backward laws. Unlike move,
  this operation can change targets and value dimension; weights are corrected."
  [scope population destination kernel]
  (spin
   (owned! scope population)
   (when-not (and (= (:id (:target population)) (:source kernel))
                  (= (:id destination) (:destination kernel)))
     (throw (ex-info "Transport proposal belongs to different targets" {:type ::wrong-target})))
   (when-not (= (type (get-in population [:target :representation]))
                (type (:representation destination)))
     (throw (ex-info "Transport requires compatible particle representations"
                     {:type ::incompatible-representation})))
   (when (some #(or (= ##-Inf %) (= ##Inf %)) (get-in population [:mass :log-weights]))
     (throw (ex-info "Transport requires positive source weights; resample first"
                     {:type ::zero-weight-transport})))
   (claim! scope population)
   (let [results (await (transport-particles (get-in population [:target :representation])
                                             (:target population) destination (:particles population) kernel
                                             (assoc scope :seed (:seed population) :step (:step population))))
         _ (population-size! results (count (:particles population)))
         next (-> (path/moved population (mapv :particle results))
                  (path/reweight (mapv :log-increment results))
                  (assoc :target destination :transitions {}) (update :step inc)
                  (update :history conj {:operation :transport :target (:id destination)}))]
     (await (reclaim scope population [] next))
     (publish scope population next))))

(defn fork
  "Copy a live population with independent future random streams. The source
  remains usable. Trace copies share read-only anchors under the owner scope."
  [scope population]
  (spin
   (claim! scope population)
   (let [i (swap! (:forks population) inc)
         id [:fork (:population-id population) i]
         seed (sp/derive-seed (:seed population) ::fork i)
         copies (await (copy-particles (get-in population [:target :representation]) (:resources scope)
                                       (:target population) (:particles population)
                                       (vec (range (count (:particles population))))
                                       (assoc scope :seed seed :step (:step population))))
         child (assoc population :population-id id :seed seed :particles copies
                      :lease (when (:lease population) (atom {:phase :idle :version 0}))
                      :version 0 :forks (atom 0) :transitions {})]
     (population-size! copies (count (:particles population)))
     (when-let [lease (:lease population)]
       (reset! lease {:phase :idle :version (:version population)}))
     (remember! scope child))))

(defn finish
  "Project to the shared settled measure API. run joins resource cleanup before
  delivery. Live trace handles are consumed; numeric value snapshots persist."
  [scope population]
  (claim! scope population)
  (let [{:keys [target particles mass]} population
        logs (:log-weights mass)
        measure (assoc (m/empirical (mapv (fn [x lw] [(project-particle (:representation target) target x)
                                                      (+ lw (Math/log (count particles)))]) particles logs))
                       :log-normalizer (:log-mass mass) :inference-contract (:inference-contract population)
                       :target (dissoc target :score :representation) :log-mass (:log-mass mass)
                       :history (:history population))]
    (when-let [lease (:lease population)]
      (reset! lease {:phase :finished :version (:version population)})
      (swap! (:live-populations scope) dissoc (:population-id population))
      (swap! (:retired scope) into particles))
    measure))
