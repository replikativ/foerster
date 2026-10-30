(ns org.replikativ.foerster.transition
  "Compilation and execution of Markov kernels. Site proposal policies and
  weighted population operations do not pass through this interpreter."
  (:refer-clojure :exclude [await])
  (:require [is.simm.partial-cps.async :as pcps]
            [org.replikativ.foerster.transition-context :as tc]
            [org.replikativ.foerster.kernel :as k]
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
            #?(:clj [org.replikativ.spindel.engine.executor :as sched])
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.combinators :as comb]
            [org.replikativ.spindel.effects.await :refer [await await-finalization]])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

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

(defn- leaf-options [kernel]
  (case (k/transition-id kernel)
    :single-site-mh {}
    :random-walk-mh {:step (block-mh/within-gibbs {:step-size (:step-size kernel)
                                                   :burn (:burn kernel 0)})}
    :delayed-acceptance {:step (block-mh/delayed-acceptance
                                (select-keys kernel [:coarse :scales :steps :step-size]))}
    :block-gibbs (block-gibbs-options kernel)
    :hmc {:step (hmc/within-gibbs (select-keys kernel [:step-size :steps]))}
    :nuts {:step (nuts/within-gibbs {:burn (:burn kernel 0)
                                     :target-accept (:target-accept kernel)
                                     :max-depth (:max-depth kernel)
                                     :metric (:metric kernel)})}
    (throw (ex-info "Unsupported Markov transition"
                    {:type ::unsupported-kernel :kernel (k/transition-id kernel)}))))

(defn- compile-tree [kernel path]
  (if (#{:cycle :mixture} (k/transition-id kernel))
    (let [children (mapv (fn [i child] (compile-tree child (conj path i)))
                         (range) (:kernels kernel))
          spans (map #(* (:num-iterations %) (::span %)) children)]
      (assoc kernel :kernels children ::path path
             ::span (if (= :cycle (k/transition-id kernel))
                      (reduce + spans) (apply max spans))))
    (assoc kernel ::options (leaf-options kernel) ::local-counter (atom -1)
           ::span 1 ::path path)))

(defn- leaves [tree]
  (if (#{:cycle :mixture} (k/transition-id tree))
    (mapcat leaves (:kernels tree)) [tree]))

(defn prepare
  "Compile once per independently moving chain or particle. The result owns
  local tuning counters and must not be shared by resampled descendants.
  `step` reuses this state across calls. `:adaptation?` reports warm-up in
  any component; such a transition is not a fixed invariant Markov kernel."
  [kernel]
  (when-not (k/markov-kernel? kernel)
    (throw (ex-info "Only Markov kernels can be prepared" {:type ::unsupported-kernel})))
  (let [tree (compile-tree kernel [])]
    {::tree tree
     :adaptation? (boolean (some #(and (#{:nuts :random-walk-mh} (k/transition-id %))
                                       (pos? (:burn % 0))) (leaves tree)))
     :capabilities {:retention #{:keep-old? :shared-anchors?}
                    :tempering? (every? #(#{:single-site-mh :block-gibbs}
                                          (k/transition-id %)) (leaves tree))}}))

(defn adaptation? [prepared] (:adaptation? prepared))

(defn- validate-context! [prepared context]
  (let [beta (get context :temperature 1.0)]
    (when-not (and (number? beta) (<= 0.0 beta 1.0))
      (throw (ex-info "Transition temperature must be between zero and one"
                      {:type ::invalid-temperature :temperature beta})))
    (when (and (not= 1.0 (double beta))
               (not (get-in prepared [:capabilities :tempering?])))
      (tc/require-untempered! context (k/transition-id (::tree prepared))))))

(declare run-tree)

(defn- iteration-of [tree current base context]
  (pcps/async
   (if (#{:cycle :mixture} (k/transition-id tree))
     (let [children (:kernels tree)
           offsets (reductions + 0 (map #(* (:num-iterations %) (::span %)) children))
           parts (if (= :cycle (k/transition-id tree))
                   (map vector children offsets)
                   [[(nth children
                          (random/in-world-stream (:trace/world current)
                                                  [::mixture base (::path tree)]
                                                  #(m/sample-categorical (:weights tree)))) 0]])]
       (loop [t current remaining (seq parts) moves 0 accepted 0]
         (if-not remaining
           {:trace t :moves moves :accepted-moves accepted}
           (let [[child offset] (first remaining)
                 r (pcps/await (run-tree child t (+ base offset) context))]
             (recur (:trace r) (next remaining)
                    (+ moves (:moves r)) (+ accepted (:accepted-moves r)))))))
     (let [opts (::options tree)
           local (swap! (::local-counter tree) inc)
           opts (cond-> opts
                  (= :block-gibbs (k/transition-id tree))
                  (update :select (fn [select] (fn [t _] (select t local)))))
           r (pcps/await ((get opts :step itrace/mh-step)
                          current (assoc (merge opts context)
                                         :iteration base :adaptation-iteration local)))
           result (assoc r :moves (:moves r 1)
                         :accepted-moves (:accepted-moves r (if (:accepted? r) 1 0)))]
       (when-let [visit (:on-visit context)] (visit result))
       result))))

(defn- run-tree [tree current base context]
  (pcps/async
   (loop [t current i 0 moves 0 accepted 0]
     (if (= i (:num-iterations tree))
       {:trace t :moves moves :accepted-moves accepted}
       (let [r (pcps/await (iteration-of tree t (+ base (* i (::span tree))) context))]
         (recur (:trace r) (inc i) (+ moves (:moves r)) (+ accepted (:accepted-moves r))))))))

(defn- reserve-span! [current size]
  ;; Share mh-chain's allocator: distinct prepared descriptors can move the
  ;; same world after rejection without replaying an earlier random stream.
  (- (rtp/swap-state! (:trace/world current)
                      [:inference :org.replikativ.foerster.trace/chain-moves]
                      #(+ (or % 0) size)) size))

(defn step
  "Run a prepared descriptor's configured iterations. Context is the target
  (:temperature, :policy-options, :constraints), replay frontier (:until,
  :anchor?), and ownership (:keep-old?, :shared-anchors?). :on-visit sees each
  leaf result, including intermediate states. :first-iteration optionally
  supplies an already reserved RNG base. Returns CPS {:trace :moves
  :accepted :accepted-moves}. Call sequentially on a prepared interpreter."
  [prepared current context]
  (validate-context! prepared context)
  (let [tree (::tree prepared)
        size (* (:num-iterations tree) (::span tree))
        base (or (:first-iteration context) (reserve-span! current size))]
    (pcps/async
     (let [r (pcps/await (run-tree tree current base (select-keys context tc/keys)))]
       (assoc r :accepted (:accepted-moves r))))))

(defn options
  "Compile a Markov descriptor once per chain, using the same interpreter
  as population `prepare`/`step`. The chain driver collects outer iterations."
  [kernel]
  (let [prepared (prepare kernel)
        tree (::tree prepared)]
    {:iterations (:num-iterations kernel)
     :iteration-stride (::span tree)
     :step (fn [current context]
             (validate-context! prepared context)
             (iteration-of tree current (:iteration context)
                           (select-keys context tc/keys)))}))

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

