(ns org.replikativ.foerster.transition
  "Compilation and execution of Markov kernels. Site proposal policies and
  weighted population operations do not pass through this interpreter."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.hmc :as hmc]
            [org.replikativ.foerster.nuts :as nuts]
            [org.replikativ.foerster.block-mh :as block-mh]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.state-backend :as backend]
            [org.replikativ.spindel.engine.executor :as sched]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.combinators :as comb]
            [org.replikativ.spindel.effects.await :refer [await await-finalization]]))

(defn- block-gibbs-options
  "Translate a BlockGibbsKernel into `itrace/mh-step` options. The classifier
  and the selector see the trace in its legacy shape; only latent
  sites are classified, and a step whose block is empty or has no kernel moves
  nothing. The selection probability is taken to be the same in both traces,
  which holds when a move does not change which sites belong to the block."
  [{:keys [block-selector block-kernels address-classifier]}]
  (let [proposals (atom nil)]
    {:select
     (fn [current iteration]
       (let [legacy (itrace/legacy-trace current)
             latent (set (itrace/latent-addresses current))
             blocks (reduce-kv (fn [acc address entry]
                                 (if-let [block-id (and (latent address)
                                                        (address-classifier address entry))]
                                   (update acc block-id (fnil conj #{}) address)
                                   acc))
                               {} legacy)
             block-id (k/select-block block-selector legacy iteration)
             targets (get blocks block-id #{})
             kernel (get block-kernels block-id)]
         (reset! proposals (when (and kernel (seq targets)
                                      (not (instance? org.replikativ.foerster.kernel.PriorBlockKernel kernel)))
                             (k/propose-block kernel legacy targets)))
         ;; Blocks are selected by id, independently of the trace.
         {:targets targets :log-selection (constantly 0.0)}))
     :propose
     (fn [sp-value old-entry]
       (if-let [proposed @proposals]
         ;; A block kernel's move is taken to be symmetric (the random walk
         ;; is; a custom kernel must be).
         {:value (get proposed (:savepoint/address sp-value)) :symmetric? true}
         (itrace/prior-proposal sp-value old-entry)))}))

(declare options)

(defn- span
  "How many moves one iteration of a Markov-chain kernel makes at most: its
  moves are numbered within that span, so every move of a chain has its own
  number (`itrace/mh-chain`'s `:first-iteration`)."
  [kernel]
  (case (k/transition-id kernel)
    :cycle (reduce + (map #(* (:num-iterations %) (span %)) (:kernels kernel)))
    :mixture (apply max (map #(* (:num-iterations %) (span %)) (:kernels kernel)))
    1))

(declare iteration-of)

(defn- run-kernel
  "CPS: `kernel`'s whole run from `trace`, its moves numbered from `base`;
  resolves {:trace :moves :accepted-moves}."
  [kernel trace base]
  (fn [resolve reject]
    (if (#{:cycle :mixture} (k/transition-id kernel))
      (letfn [(run [current i moves accepted]
                (if (= i (:num-iterations kernel))
                  (resolve {:trace current :moves moves :accepted-moves accepted})
                  ((iteration-of kernel current (+ base (* i (span kernel))))
                   (fn [r] (run (:trace r) (inc i) (+ moves (:moves r)) (+ accepted (:accepted-moves r))))
                   reject)))]
        (run trace 0 0 0))
      (let [{:keys [iterations] :as opts} (or (::options kernel) (options kernel))]
        ((itrace/mh-chain trace iterations (-> opts (dissoc :iterations) (assoc :first-iteration base)))
         (fn [{:keys [trace moves accepted]}]
           (resolve {:trace trace :moves moves :accepted-moves accepted}))
         reject)))))

(defn- iteration-of
  "CPS: one iteration of a composed kernel from `trace`, its moves numbered
  from `base`: every component in turn (cycle) or one picked by weight
  (mixture)."
  [kernel trace base]
  (fn [resolve reject]
    (let [ks (:kernels kernel)
          offsets (reductions + 0 (map #(* (:num-iterations %) (span %)) ks))
          parts (case (k/transition-id kernel)
                  :cycle (map vector ks offsets)
                  :mixture [[(nth ks (m/sample-categorical (:weights kernel))) 0]])]
      (letfn [(run [current [[component offset] & more] moves accepted]
                (if-not component
                  (resolve {:trace current :moves moves :accepted-moves accepted})
                  ((run-kernel component current (+ base offset))
                   (fn [r] (run (:trace r) more (+ moves (:moves r)) (+ accepted (:accepted-moves r))))
                   reject)))]
        (run trace parts 0 0)))))

(defn- compiled
  "A composed kernel with each component's options made once (`::options`):
  a component that adapts as it runs (NUTS) keeps its state from one of the
  composition's iterations to the next."
  [kernel]
  (if (#{:cycle :mixture} (k/transition-id kernel))
    (update kernel :kernels #(mapv compiled %))
    (assoc kernel ::options (options kernel))))

(defn options
  "Compile a Markov-kernel description once per chain. This creates any
  component adaptation state; classification must use kernel/markov-kernel?."
  [kernel]
  (case (k/transition-id kernel)
    (:cycle :mixture) (let [kernel (compiled kernel)]
                        {:iterations (:num-iterations kernel)
                         :step (fn [trace {:keys [iteration]}]
                                 (iteration-of kernel trace (* iteration (span kernel))))})
    :single-site-mh {:iterations (:num-iterations kernel)}
    :random-walk-mh {:iterations (:num-iterations kernel)
                     :step (block-mh/within-gibbs {:step-size (:step-size kernel)
                                                   :burn (:burn kernel 0)})}
    :delayed-acceptance {:iterations (:num-iterations kernel)
                         :step (block-mh/delayed-acceptance
                                (select-keys kernel [:coarse :scales :steps :step-size]))}
    :block-gibbs (assoc (block-gibbs-options kernel)
                        :iterations (:num-iterations kernel))
    :hmc {:iterations (:num-iterations kernel)
          :step (hmc/within-gibbs (select-keys kernel [:step-size :steps]))}
    :nuts {:iterations (:num-iterations kernel)
           :step (nuts/within-gibbs {:burn (:burn kernel 0)
                                     :target-accept (:target-accept kernel)
                                     :max-depth (:max-depth kernel)
                                     :metric (:metric kernel)})}
    (throw (ex-info "Unsupported Markov transition" {:type ::unsupported-kernel
                                                     :kernel (k/transition-id kernel)}))))

(defn- project-posterior-context
  "A parentless immutable context holding what the posterior needs of
  `world`'s inference state: the world itself would retain its ancestry."
  [world]
  (assoc world
         :backend (backend/create-immutable-backend
                   {:inference (select-keys (rtp/get-state world [:inference])
                                            [:log-weight :trace :result :mcmc])}
                   {:source-fork-id (:fork-id world)
                    :projection :inference-posterior})
         :parent-ctx nil
         :bindings {}
         :metadata {:inference/projection true}
         :running nil
         :drain-active nil))

(defn- run-markov-chain
  "One chain in its own world: run the model, move it `iterations` times,
  give every world back. Returns the chain's particles: its projected final
  state, or (kernel `:samples :all`) every state after `:burn` as a Sample."
  [model-task kernel executor seed]
  (spin-core/with-causal-descendant-egress
    (spin
     (let [;; per chain: a block Gibbs description closes over its own state
           {:keys [iterations] :as step-opts} (options kernel)
           _ (when (and (= :all (:samples kernel)) (>= (:burn kernel 0) iterations))
               (throw (ex-info ":burn leaves no draws: it must be below :iterations"
                               {:type :org.replikativ.foerster.core/burn-exceeds-iterations :burn (:burn kernel) :iterations iterations})))
           root (ctx/create-execution-context :executor executor)
           session (sp/open! root {:purpose :mcmc :seed seed :fork-opts {:systems :none}
                                   :retain-released? false})]
       (try
         (let [initial (await (trace/run session model-task (itrace/policy {:init? true})
                                         {:anchor? itrace/anchor?}))
               _ (when (contains? initial :trace/error)
                   (throw (ex-info "Inference failed during model execution"
                                   {:type :org.replikativ.foerster.core/inference-failed}
                                   (:trace/error initial))))
             ;; From an impossible state every ratio is NaN and nothing is
             ;; ever accepted; say so instead of returning that state.
               _ (when (= ##-Inf (itrace/log-joint initial))
                   (throw (ex-info "The initial state of the chain has zero density"
                                   {:type :org.replikativ.foerster.core/impossible-initial-state})))
               samples (volatile! [])
               step-opts (cond-> step-opts
                           (= :all (:samples kernel))
                           (assoc :on-step
                                  (let [i (volatile! 0)]
                                    (fn [{t :trace :keys [sampler-stats]}]
                                      (when (> (vswap! i inc) (:burn kernel 0))
                                        (vswap! samples conj
                                                [(cond-> (m/sample-particle (:trace/result t)
                                                                            (itrace/legacy-trace t))
                                                   sampler-stats (assoc :sampler-stats sampler-stats))
                                                 0.0]))))))
               {final :trace moves :moves accepted :accepted}
               (await (itrace/mh-chain initial iterations step-opts))
               world (:trace/world final)]
           (rtp/swap-state! world [:inference]
                            (fn [state]
                              (assoc state
                                     :result (:trace/result final)
                                     :trace (itrace/legacy-trace final)
                                     :mcmc {:completed-iterations iterations
                                            :moves moves
                                            :acceptance-count accepted})))
           (if (= :all (:samples kernel))
             @samples
             [[(project-posterior-context world) 0.0]]))
         (finally
         ;; Closing the session cancels and joins every world of the chain.
         ;; The root is not stopped here: `stop-context!` waits for the
         ;; context's drains, and this body may be running inside one.
           (await-finalization (sp/close! session))))))))

(defn infer
  [model-task kernel num-chains opts]
  ;; Chains run in fresh worlds of their own. Running them in forks of the
  ;; caller's world, as `:world-policy :fork` does for particles, is not
  ;; implemented; refuse it, do not ignore it.
  (when (or (= :fork (:world-policy opts)) (some? (:world-opts opts)))
    (throw (ex-info "Markov-chain kernels run in fresh worlds"
                    {:type :org.replikativ.foerster.core/invalid-world-policy
                     :world-policy (:world-policy opts)
                     :supported #{:fresh}})))
  (spin-core/with-causal-descendant-egress
    (spin
     (let [own-executor (when-not (:executor opts)
                          #?(:clj (sched/thread-pool-executor {:threads 2}) :cljs nil))
           executor (or (:executor opts) own-executor)]
       (try
         (let [;; derived from :seed, else drawn here in order: the chains
               ;; then run concurrently
               seeds (if-let [seed (:seed opts)]
                       (mapv #(sp/derive-seed seed :org.replikativ.foerster.core/chain %) (range num-chains))
                       (vec (repeatedly num-chains random/fresh-seed)))
               chains (await (apply comb/parallel
                                    (mapv #(run-markov-chain model-task kernel executor %) seeds)))]
           ;; the draws in chain order, each chain's count kept: the
           ;; convergence diagnostics need them by chain (`foerster.diagnostics`)
           (cond-> (assoc (m/markov-measure (into [] cat chains))
                          :chain-lengths (mapv count chains))
             (some? (:seed opts)) (assoc :seed (:seed opts))))
         (finally
           (when own-executor
             #?(:clj (.close ^java.lang.AutoCloseable own-executor)
                :cljs nil))))))))

