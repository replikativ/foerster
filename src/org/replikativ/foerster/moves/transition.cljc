(ns org.replikativ.foerster.moves.transition
  "Compilation and execution of Markov kernels. Site proposal policies and
  weighted population operations do not pass through this interpreter."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.runtime.execution :as execution]
            [is.simm.partial-cps.async :as pcps]
            [org.replikativ.foerster.moves.transition-context :as tc]
            [org.replikativ.foerster.moves.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.moves.hmc :as hmc]
            [org.replikativ.foerster.moves.nuts :as nuts]
            [org.replikativ.foerster.moves.block-mh :as block-mh]
            [org.replikativ.foerster.runtime.gfi :as gfi]
            [org.replikativ.foerster.moves.involutive :as involutive]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.spindel.v2.savepoint :as sp]
            [org.replikativ.spindel.v2.trace :as trace]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]
            [org.replikativ.foerster.runtime.async :as async :refer [await-op]])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

(defn- block-gibbs-options
  "Translate a BlockGibbsKernel into `itrace/mh-step` options. The classifier
  and the selector see the trace in its legacy shape; only latent
  sites are classified, and a step whose block is empty or has no kernel moves
  nothing. Blocks are selected by id, independently of the trace; a move that
  changes which sites belong to the selected block is rejected (its selection
  term is −∞), because its reverse would move a different set of sites.
  Rejecting it in both directions keeps detailed balance."
  [{:keys [block-selector block-kernels address-classifier]}]
  (let [proposals (atom nil)
        block-of (fn [t legacy block-id]
                   (let [latent (set (itrace/latent-addresses t))]
                     (into #{} (keep (fn [[address entry]]
                                       (when (and (some? block-id)
                                                  (latent address)
                                                  (= block-id (address-classifier address entry)))
                                         address)))
                           legacy)))]
    {:select
     (fn [current iteration]
       (let [legacy (itrace/legacy-trace current)
             block-id (k/select-block block-selector legacy iteration)
             targets (block-of current legacy block-id)
             kernel (get block-kernels block-id)]
         (reset! proposals (when (and kernel (seq targets)
                                      (not (instance? org.replikativ.foerster.moves.kernel.PriorBlockKernel kernel)))
                             (k/propose-block kernel legacy targets)))
         {:targets targets
          :log-selection (fn [t]
                           (if (or (identical? t current)
                                   (= targets (block-of t (itrace/legacy-trace t) block-id)))
                             0.0
                             ##-Inf))}))
     :propose
     (fn [sp-value old-entry]
       (if-let [proposed @proposals]
         ;; A block kernel's move is taken to be symmetric (the random walk
         ;; is; a custom kernel must be).
         {:value (get proposed (:savepoint/address sp-value)) :symmetric? true}
         (itrace/prior-draw sp-value old-entry)))}))

(declare builtin-leaf-options)

(defn- leaf-options [kernel]
  (if (k/custom-kernel? kernel)
    {:step (k/custom-step kernel)}
    (builtin-leaf-options kernel)))

(defn- only-site
  "A selection of one latent site among those `only` (a set or a predicate
  of an address) names, uniformly."
  [only]
  (let [movable (fn [t] (filterv only (:addresses (itrace/latent-index t))))]
    (fn [t _]
      (let [ms (movable t)]
        (if (empty? ms)
          {:targets #{} :log-selection (constantly 0.0)}
          {:targets #{(random/pick-uniformly ms)}
           :log-selection #(- (Math/log (double (count (movable %)))))})))))

(defn- block-kernel
  "The block kernel of a leaf: `within-gibbs`'s, or the leaf itself."
  [leaf]
  (if (= :within-gibbs (k/transition-id leaf)) (:kernel leaf) leaf))

(defn- builtin-leaf-options [kernel]
  (case (k/transition-id kernel)
    :single-site-mh (if-let [only (:only kernel)] {:select (only-site only)} {})
    :within-gibbs (builtin-leaf-options (assoc (:kernel kernel) ::others? true))
    :finite-gibbs {:step (fn [current opts]
                           (itrace/update-gibbs current (:address kernel) opts))}
    :random-walk-mh {:step (block-mh/within-gibbs {:step-size (:step-size kernel)
                                                   :burn (:burn kernel 0)})}
    :delayed-acceptance {:step (block-mh/delayed-acceptance
                                (select-keys kernel [:coarse :scales :steps :step-size]))}
    :block-gibbs (block-gibbs-options kernel)
    :hmc {:step (hmc/within-gibbs (assoc (select-keys kernel [:step-size :steps :only])
                                         :others? (::others? kernel false)))}
    :involutive {:step (fn [current _] (involutive/step current (:move kernel)))}
    :proposal-mh {:step (fn [current _] (apply gfi/mh-proposal current (:proposal kernel) (:args kernel)))}
    :nuts {:step (nuts/within-gibbs {:burn (:burn kernel 0)
                                     :target-accept (:target-accept kernel)
                                     :max-depth (:max-depth kernel)
                                     :metric (:metric kernel)
                                     :step-size (:step-size kernel)
                                     :inverse-metric (:inverse-metric kernel)
                                     :only (:only kernel)
                                     :others? (::others? kernel false)})}
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

(defn- adaptive-leaf?
  "Whether a leaf kernel tunes itself during its `:burn` (NUTS and the
  adaptive random walk); other kernels' `:burn` only drops output."
  [leaf]
  (contains? #{:nuts :random-walk-mh} (k/transition-id (block-kernel leaf))))

(declare leaf-capabilities)

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
     :adaptation? (boolean (some #(and (adaptive-leaf? %) (pos? (:burn % 0))) (leaves tree)))
     :capabilities (leaf-capabilities (leaves tree))}))

(defn- leaf-capabilities
  "What a kernel whose leaves are `ls` can do, as `prepare` reports it:
  `:retention`, the ownership options every leaf honours (programmable moves
  give back the state they leave: they cannot keep it alive for a
  population's other owners; a custom kernel honours them when it declares
  `:retains?`); `:tempering?`, whether every leaf targets β < 1 correctly;
  `:verified?`, whether every leaf is built in and leaves its target
  invariant by construction (an involution, a proposal program or a custom
  step is its author's obligation)."
  [ls]
  (let [user? #(or (k/custom-kernel? %) (#{:involutive :proposal-mh} (k/transition-id %)))
        retains? #(if (k/custom-kernel? %)
                    (:retains? (k/custom-capabilities %))
                    (not (#{:involutive :proposal-mh} (k/transition-id %))))
        tempers? #(if (k/custom-kernel? %)
                    (:tempering? (k/custom-capabilities %))
                    (#{:single-site-mh :block-gibbs :finite-gibbs} (k/transition-id %)))]
    {:retention (if (every? retains? ls) #{:keep-old? :shared-anchors?} #{})
     :tempering? (boolean (every? tempers? ls))
     :verified? (not-any? user? ls)}))

(defn adaptation? [prepared] (:adaptation? prepared))

(defn- warming-up-fn
  "A function of no arguments: whether some leaf of `tree` with a warm-up
  (`:burn`, NUTS and adaptive random-walk MH) has not yet finished it. A
  leaf's local counter is the index of its latest move; the state after move
  `burn` − 1 is still warm-up."
  [tree]
  (let [adaptive (filterv #(and (adaptive-leaf? %) (pos? (:burn % 0))) (leaves tree))]
    (fn [] (boolean (some #(< @(::local-counter %) (:burn %)) adaptive)))))

(defn- validate-context! [prepared context]
  (let [beta (itrace/target-temperature context)]
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
                                                  #(weights/sample-categorical (:weights tree)))) 0]])]
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
  (- (v2/update-in-world! (:trace/world current)
                      [:inference :org.replikativ.foerster.runtime.trace/chain-moves]
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
     :warming-up? (warming-up-fn tree)
     :step (fn [current context]
             (validate-context! prepared context)
             (iteration-of tree current (:iteration context)
                           (select-keys context tc/keys)))}))

(defn- project-posterior-context
  "A world of its own holding only what the posterior needs of `world`'s
  inference state: `world` itself would retain the chain's worlds."
  [world]
  (doto (v2/world)
    (v2/reset-world! {:inference (select-keys (v2/get-in-world world [:inference])
                                              [:log-weight :trace :result :mcmc])})))

(defn- canonical-session
  "A chain's root and session: a host fork of `source` (the inference's
  frozen fork of the caller's world, `:world-policy :fork`). Closing the
  session releases the worlds it made; the root is released with it."
  [source seed]
  (let [root (v2/fork! source)]
    (itrace/mark-canonical! root)
    [root (sp/open! root {:seed seed})]))

(defn- covers
  "A predicate (fn [trace legacy address]) of the latent sites `leaf` can
  move, or nil when that is its author's (a custom step, an involution, a
  proposal program)."
  [leaf]
  (case (k/transition-id leaf)
    :single-site-mh (if-let [only (:only leaf)] (fn [_ _ a] (boolean (only a))) (constantly true))
    (:hmc :nuts) (let [only (:only leaf)]
                   (fn [t _ a] (and (hmc/block-site? t a) (or (nil? only) (boolean (only a))))))
    :finite-gibbs (fn [_ _ a] (= a (:address leaf)))
    :block-gibbs (fn [_ legacy a]
                   (contains? (:block-kernels leaf) ((:address-classifier leaf) a (get legacy a))))
    (:within-gibbs :random-walk-mh :delayed-acceptance) (constantly true)
    nil))

(defn- uncovered
  "The latent sites of `trace` no leaf of `kernel` moves, or nil when a leaf
  does not say which it moves."
  [kernel trace]
  (let [cs (mapv covers (leaves kernel))]
    (when (every? some? cs)
      (let [legacy (itrace/legacy-trace trace)]
        (seq (remove (fn [a] (some #(% trace legacy a) cs)) (itrace/latent-addresses trace)))))))

(defn- run-markov-chain
  "One chain in its own world: run the model, move it `iterations` times,
  give every world back. Returns the chain's particles: its projected final
  state, or (kernel `:samples :all`) every state after `:burn` as a Sample.
  A state is never a draw while a component kernel is still in its own
  warm-up (a NUTS or adaptive random-walk leaf's `:burn` inside a `cycle` or
  `mixture`); a chain that ends before every warm-up finished is an error.
  With `source` (`:world-policy :fork`) the chain's world is a fork of it
  and the model runs from a start site, the anchor a move re-executes it
  from when no later one precedes its sites."
  [model-task kernel executor seed source _world-opts]
  (do
    (pcps/async
     (let [;; per chain: a block Gibbs description closes over its own state
           {:keys [iterations warming-up?] :as step-opts} (options kernel)
           step-opts (dissoc step-opts :warming-up?)
           _ (when (and (= :all (:samples kernel)) (>= (:burn kernel 0) iterations))
               (throw (ex-info ":burn leaves no draws: it must be below :iterations"
                               {:type :org.replikativ.foerster.core/burn-exceeds-iterations :burn (:burn kernel) :iterations iterations})))
           [root session] (if source
                            (canonical-session source seed)
                            (let [root (v2/world {:executor executor})]
                              [root (sp/open! root {:seed seed})]))
           model-task (if source
                        ;; every chain runs the whole model, from a site a
                        ;; move can re-execute it from (`itrace/reexecute`)
                        (fn [] (task (sp/savepoint :org.replikativ.foerster.inference.smc/start nil)
                                     (v2/await (model-task))))
                        model-task)]
       (try
         (let [initial (pcps/await (async/op-of (trace/run session model-task (itrace/policy {:init? true})
                                         {:anchor? (if source itrace/canonical-anchor? itrace/anchor?)})))
               _ (when (contains? initial :trace/error)
                   (throw (ex-info "Inference failed during model execution"
                                   {:type :org.replikativ.foerster.core/inference-failed}
                                   (:trace/error initial))))
             ;; From an impossible state every ratio is NaN and nothing is
             ;; ever accepted; say so instead of returning that state.
               _ (when (= ##-Inf (itrace/log-joint initial))
                   (throw (ex-info "The initial state of the chain has zero density"
                                   {:type :org.replikativ.foerster.core/impossible-initial-state})))
               ;; a site no part of the kernel moves keeps its initial value:
               ;; the chain would not target the posterior
               _ (when-let [missing (uncovered kernel initial)]
                   (throw (ex-info (str "No part of the kernel moves the latent sites "
                                        (vec (take 5 missing)) (when (next (drop 4 missing)) " …")
                                        ": add a kernel that does (single-site-mh-kernel, finite-gibbs-kernel,"
                                        " or within-gibbs around a block kernel)")
                                   {:type ::uncovered-sites :sites (vec missing)})))
               ;; NUTS chains start jittered around their :init (Stan's
               ;; random initialization); population moves never do
               initial (if (some #(let [b (block-kernel %)] (and (= :nuts (k/transition-id b)) (:jitter b true)))
                                 (leaves kernel))
                         (pcps/await (async/op-of (nuts/jitter initial)))
                         initial)
               samples (volatile! [])
               step-opts (cond-> step-opts
                           (= :all (:samples kernel))
                           (assoc :on-step
                                  (let [i (volatile! 0)]
                                    (fn [{t :trace :keys [sampler-stats]}]
                                      (when (and (> (vswap! i inc) (:burn kernel 0))
                                                 (not (warming-up?)))
                                        (vswap! samples conj
                                                [(cond-> (m/sample-particle (:trace/result t)
                                                                            (itrace/legacy-trace t))
                                                   sampler-stats (assoc :sampler-stats sampler-stats))
                                                 0.0]))))))
               {final :trace moves :moves accepted :accepted}
               (pcps/await (async/op-of (itrace/mh-chain initial iterations step-opts)))
               _ (when (warming-up?)
                   (throw (ex-info (str "A kernel's warm-up (:burn of a NUTS or adaptive random-walk kernel) "
                                        "did not finish within the chain: the chain must run past it. In a cycle "
                                        "or mixture the warm-up counts that kernel's own moves, which a mixture "
                                        "makes only when it picks it: raise the iterations or lower its :burn")
                                   {:type :org.replikativ.foerster.core/warm-up-unfinished
                                    :iterations iterations})))
               world (:trace/world final)]
           (v2/update-in-world! world [:inference]
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
           ;; releases every world of the chain (engine v2 finalizers do not
           ;; await: the close is requested, not joined)
           ((sp/close! session) (fn [_]) (fn [_]))
           (v2/release-world! root)))))))

(defn infer
  "`num-chains` chains of the Markov `kernel` on `model-task`, each in fresh
  worlds of its own, or, with `:org.replikativ.foerster.core/source` (a
  frozen fork of the caller's world, `:world-policy :fork`; see
  `foerster.core/kernel-infer`), each in a fork of that source with
  `:world-opts`. Returns a spin resolving the chains' measure."
  [model-task kernel num-chains opts]
  (when (and (= :fork (:world-policy opts)) (not (:org.replikativ.foerster.core/source opts)))
    (throw (ex-info "Markov chains under :world-policy :fork need the caller's frozen fork (foerster.core/kernel-infer)"
                    {:type :org.replikativ.foerster.core/invalid-world-policy
                     :world-policy :fork})))
  (do
    (pcps/async
     (let [own-executor (when-not (:executor opts)
                          #?(:clj (java.util.concurrent.Executors/newFixedThreadPool 2) :cljs nil))
           executor (or (:executor opts) own-executor)]
       (try
         (let [;; derived from :seed, else drawn here in order: the chains
               ;; then run concurrently
               seeds (if-let [seed (:seed opts)]
                       (mapv #(sp/derive-seed seed :org.replikativ.foerster.core/chain %) (range num-chains))
                       (vec (repeatedly num-chains random/fresh-seed)))
               chain-tasks (mapv #(run-markov-chain model-task kernel executor %
                                                     (:org.replikativ.foerster.core/source opts)
                                                     (:world-opts opts))
                                 seeds)
               chains (pcps/await (execution/parallel nil (count chain-tasks)
                                                      (fn [i res rej] ((nth chain-tasks i) res rej))))]
           ;; the draws in chain order, each chain's count kept: the
           ;; convergence diagnostics need them by chain (`foerster.diagnostics`)
           (cond-> (assoc (m/markov-measure (into [] cat chains))
                          :chain-lengths (mapv count chains))
             (some? (:seed opts)) (assoc :seed (:seed opts))))
         (finally
           (when own-executor
             #?(:clj (.shutdown ^java.util.concurrent.ExecutorService own-executor)
                :cljs nil))))))))

