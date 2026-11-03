(ns org.replikativ.foerster.inference.pmcmc
  "Particle MCMC drivers: PIMH and particle Gibbs (Andrieu, Doucet &
  Holenstein 2010), PGAS (Lindsten, Jordan & Schön 2014) and iPMCMC
  (Rainforth et al. 2016). In fresh worlds PIMH, particle Gibbs and PGAS
  run on savepoint SMC (`foerster.inference.smc`); in canonical worlds, and
  iPMCMC always, they run over SMC sweeps (`foerster.inference.particle-mcmc`)."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.core :as-alias core]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.inference.smc :as smc]
            [org.replikativ.foerster.inference.particle-mcmc :as chains]
            [org.replikativ.foerster.inference.worlds :refer [check-options! particle-options smc-options
                                                              on-savepoints? on-savepoints particles]]
            [org.replikativ.foerster.runtime.canonical :refer [share balance-op frozen-source with-frozen-caller]]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.combinators :as comb]
            [org.replikativ.spindel.effects.await :refer [await]]
            [replikativ.logging :as log]))

(defn- particle-sweep [model n opts]
  (fn [key retained seed]
    (particles model n (cond-> (assoc opts :seed (sp/derive-seed seed ::core/sweep key))
                         retained (assoc :retained retained)))))

(defn pimh-infer
  "Particle Independent Metropolis-Hastings (Andrieu et al. 2010).

  Each iteration proposes a fresh SMC sweep and accepts it with probability
  min(1, Ẑ_new / Ẑ_current); the current sweep's particles, normalized, are
  emitted every iteration.

  Args:
    model-task, num-particles (per sweep), num-iterations
    opts: the particle methods' options (`:seed`, `:executor`,
          `:resample-threshold`, `:policy`, `:world-policy`, `:world-opts`,
          `:authority`, `:grant`) and SMC's (`:resampling`, `:genealogy?`,
          and in fresh worlds `:anchors`, `:rejuvenate`, `:smcp3`)

  Returns: Spin<EmpiricalMeasure>"
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts (into particle-options smc-options))
  (if (on-savepoints? opts)
    (on-savepoints (smc/pimh model-task num-particles num-iterations opts))
    (with-frozen-caller opts
      (fn [opts]
        (chains/pimh
         (particle-sweep model-task num-particles opts)
         num-iterations {:seed (:seed opts) :accept-key ::core/pimh-accept})))))

(defn pgibbs-infer
  "Particle Gibbs (conditional SMC, Andrieu et al. 2010).

  Args:
    model-task, num-particles (per sweep, including the retained one),
    num-iterations (sweeps)
    opts: the particle methods' options, and :anchors, :rejuvenate for
          resample-move inside each sweep, the retained particle included
          (`foerster.inference.smc/smc`; :fresh only)

  Returns: Spin<EmpiricalMeasure> of every sweep's particles, each sweep
  normalized to total weight one."
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts (into particle-options #{:anchors :rejuvenate}))
  (if (on-savepoints? opts)
    (on-savepoints (smc/pgibbs model-task num-particles num-iterations opts))
    (with-frozen-caller opts
      (fn [opts]
        (chains/pgibbs
         (particle-sweep model-task num-particles opts)
         num-iterations {:seed (:seed opts) :pick-key ::core/pick})))))

;; =============================================================================
;; IPMCMC - Interacting Particle MCMC
;; =============================================================================

(defn- gibbs-update-csmc-indices
  "Perform Gibbs sweep on CSMC node indices.

   For each CSMC slot, consider swapping with an SMC node based on log-Z values.
   Returns [new-csmc-indices zeta-sums] where zeta-sums are weights for
   Rao-Blackwellization.

   Args:
     log-Zs - Vector of log marginal likelihood estimates from all nodes
     num-csmc-nodes - Number of CSMC nodes

   Returns:
     [csmc-indices zeta-sums]"
  [log-Zs num-csmc-nodes]
  (let [num-nodes (count log-Zs)]
    (loop [i 0
           csmc-indices (vec (range num-csmc-nodes))
           smc-indices (vec (range num-csmc-nodes num-nodes))
           zeta-sums (vec (repeat num-nodes 0.0))]
      (if (= i num-csmc-nodes)
        [csmc-indices zeta-sums]
        ;; Consider swapping CSMC node i with an SMC node
        (let [;; Candidate indices: current SMC nodes + current CSMC node i
              proposal-indices (conj smc-indices (csmc-indices i))
              proposal-log-Zs (mapv #(nth log-Zs %) proposal-indices)
              probs (weights/normalize-log-weights proposal-log-Zs)

              ;; Update zeta sums for Rao-Blackwellization
              new-zeta-sums (reduce (fn [zs [idx p]]
                                      (update zs idx + p))
                                    zeta-sums
                                    (map vector proposal-indices probs))

              ;; Sample which node to use as CSMC
              k (weights/sample-categorical probs)]

          (if (= k (count smc-indices))
            ;; Keep current CSMC node (k points to the appended csmc index)
            (recur (inc i) csmc-indices smc-indices new-zeta-sums)
            ;; Swap: SMC node k becomes CSMC, current CSMC becomes SMC
            (recur (inc i)
                   (assoc csmc-indices i (smc-indices k))
                   (assoc smc-indices k (csmc-indices i))
                   new-zeta-sums)))))))

(defn- run-sweep
  "Run a single SMC or CSMC sweep.

   Args:
     model-task - The probabilistic program
     num-particles - Particles per sweep
     retained-trace - Retained trace for CSMC (nil for plain SMC)
     opts - Inference options

   Returns: Spin<Measure>"
  [model-task num-particles retained-trace opts]
  (cond
    ;; SMC sweep (no retained trace)
    (nil? retained-trace)
    (particles model-task num-particles opts)

    ;; CSMC sweep with retained trace
    :else
    (particles model-task num-particles
               (assoc opts :retained (chains/retained-choices retained-trace)))))

(defn- run-parallel-sweeps
  "Run SMC/CSMC sweeps in parallel across all nodes.

   Uses the parallel combinator to launch all sweeps concurrently,
   scaling with the underlying executor.

   Args:
     model-task - The probabilistic program
     num-particles - Particles per sweep
     retained-traces - Vector of retained traces (nil for SMC, trace for CSMC)
     opts - Inference options

   Returns: Spin<Vector<Measure>> - A spin that completes with all sweep measures"
  [model-task num-particles retained-traces opts]
  ;; Create individual sweep spins for each node
  ;; each node's seed derives from the sweep's (`:seed`) and its index, so
  ;; the nodes do not depend on the order in which they run
  (let [sweep-spins (vec (map-indexed (fn [i retained-trace]
                                        (run-sweep model-task num-particles retained-trace
                                                   (assoc opts :seed (sp/derive-seed (:seed opts) ::core/node i))))
                                      retained-traces))]
    ;; Use parallel combinator to run all sweeps concurrently
    (apply comb/parallel sweep-spins)))

(declare ipmcmc)

(defn ipmcmc-infer
  "Interacting Particle MCMC inference.

   Runs M nodes in parallel, where M_c nodes run conditional SMC (with retained
   particles) and M_s nodes run plain SMC. After each sweep, performs Gibbs
   updates on which nodes become CSMC based on marginal likelihood estimates.

   This creates 'interaction' between parallel chains: nodes with higher log-Z
   are more likely to have their particles retained in future sweeps.

   Algorithm:
   1. Initialize: Run SMC on all nodes
   2. For each iteration:
      a. Run CSMC on M_c nodes (with retained particles from previous sweep)
      b. Run SMC on M_s nodes (fresh)
      c. Collect log-Z estimates from each node
      d. Gibbs update: sample which nodes become CSMC for next sweep
      e. Extract retained particles for selected CSMC nodes
   3. Output: Weighted samples from all nodes with Rao-Blackwellized weights

   Args:
     model-task - Spin representing probabilistic program
     num-particles - Number of particles per sweep (per node)
     num-iterations - Number of IPMCMC iterations
     opts - Optional map with:
       :num-nodes - Total number of nodes (default 8)
       :num-csmc-nodes - Number of CSMC nodes (default num-nodes/2)
       :executor - Shared executor
       :all-particles? - Return all particles or one per node (default true)
       and the other particle methods' options except :resample-threshold

   Every node, conditional or not, resamples at every barrier by multinomial
   draws: iPMCMC's extended target needs each conditional node to be the
   conditional of the very SMC the free nodes run, since the Gibbs draw of
   the conditional nodes compares their evidence estimates (Rainforth et al.
   2016, §3; Anglican's ipmcmc.clj runs both node types alike). An
   ESS-adaptive free node would not be, so `:resample-threshold` is refused.

   Returns: Spin<EmpiricalMeasure>

   Reference:
     Rainforth et al., 'Interacting Particle Markov Chain Monte Carlo', ICML 2016"
  [model-task num-particles num-iterations & [opts]]
  (when (contains? opts :resample-threshold)
    (throw (ex-info (str "IPMCMC resamples every node at every barrier (the conditional nodes "
                         "must be conditionals of the free nodes' SMC): :resample-threshold does not apply")
                    {:type ::core/unknown-options :unknown [:resample-threshold]})))
  (check-options! opts (into (disj particle-options :resample-threshold)
                             #{:num-nodes :num-csmc-nodes :all-particles?}))
  (with-frozen-caller opts #(ipmcmc model-task num-particles num-iterations %)))

(defn- ipmcmc
  [model-task num-particles num-iterations opts]
  (spin
   (let [num-nodes (or (:num-nodes opts) 8)
         num-csmc-nodes (or (:num-csmc-nodes opts) (quot num-nodes 2))
         num-smc-nodes (- num-nodes num-csmc-nodes)
         all-particles? (get opts :all-particles? true)
         seed (or (:seed opts) (random/fresh-seed))
         ;; every node resamples at every barrier (ESS ≤ N < ∞·N),
         ;; multinomially, as conditional SMC does (`smc/smc` :retained)
         node-opts (assoc opts :resample-threshold ##Inf :resampling :multinomial)
         ;; the nodes of a sweep run at once: in a frozen fork of the
         ;; caller's world with an authority, each takes an even share of
         ;; what is left, decided before any of them starts
         node-grant (fn [opts]
                      (spin (when (and (frozen-source opts) (:authority opts))
                              (share (await (balance-op (:authority opts) (frozen-source opts))) num-nodes))))
         sweep-opts (fn [iteration grant]
                      (cond-> (assoc node-opts :seed (sp/derive-seed seed ::core/sweep iteration))
                        grant (assoc ::core/grant grant)))]
     (assert (> num-csmc-nodes 0) ":num-csmc-nodes must be > 0")
     (assert (< num-csmc-nodes num-nodes) ":num-csmc-nodes must be < :num-nodes")
     (loop [iteration 0
            ;; iteration 0 runs plain SMC on every node
            measures (await (run-parallel-sweeps model-task num-particles
                                                 (vec (repeat num-nodes nil))
                                                 (sweep-opts :initial (await (node-grant opts)))))
            all-samples []]
       (if (>= iteration num-iterations)
         (m/markov-measure all-samples)
         (let [log-Zs (mapv m/log-marginal measures)
               ;; the iteration's Gibbs draws from its own stream of the run's seed
               [csmc-indices _zeta-sums samples retained-traces]
               (random/with-stream*
                 seed [::core/gibbs iteration]
                 (fn []
                   (let [[csmc-indices zeta-sums] (gibbs-update-csmc-indices log-Zs num-csmc-nodes)
                   ;; emit THESE sweeps, each node weighted by its Rao-Blackwellized
                   ;; probability of being a conditional node (ζ_j), particles
                   ;; normalized within the node
                         samples (vec (mapcat
                                       (fn [node-idx]
                                         (let [zeta (nth zeta-sums node-idx)
                                               node (chains/normalized (nth measures node-idx))]
                                           (when (pos? zeta)
                                             (if all-particles?
                                               (map (fn [[smp lw]] [smp (+ lw (Math/log zeta))]) node)
                                               [[(first (nth node (weights/sample-categorical
                                                                   (weights/normalize-log-weights (mapv second node)))))
                                                 (Math/log zeta)]]))))
                                       (range num-nodes)))
                         retained-traces (vec (concat
                                               (map (fn [node-idx]
                                                      (let [ps (m/get-particles (nth measures node-idx))]
                                                        (m/get-trace (first (nth ps (weights/sample-categorical
                                                                                     (weights/normalize-log-weights (mapv second ps))))))))
                                                    csmc-indices)
                                               (repeat num-smc-nodes nil)))]
                     [csmc-indices zeta-sums samples retained-traces])))]
           (log/trace :ichains/gibbs-update {:iteration iteration :csmc-indices csmc-indices})
           (recur (inc iteration)
                  (await (run-parallel-sweeps model-task num-particles retained-traces
                                              (sweep-opts iteration (await (node-grant opts)))))
                  (into all-samples samples))))))))

(defn pgas-infer
  "Particle Gibbs with Ancestor Sampling (Lindsten et al. 2014).

  Like `pgibbs-infer`, but at every barrier the retained particle redraws
  which particle's past it continues from, with weights
  w_i · p(retained future | particle i's past) computed by re-running each
  particle's future on the retained values. Improves mixing on state-space
  models; costs a forward re-run per particle per barrier.

  Returns: Spin<EmpiricalMeasure> of every sweep's particles, each sweep
  normalized to total weight one."
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts particle-options)
  (if (on-savepoints? opts)
    (on-savepoints (smc/pgas model-task num-particles num-iterations opts))
    (with-frozen-caller opts
      (fn [opts]
        (chains/pgibbs (particle-sweep model-task num-particles (assoc opts :ancestor-sampling? true))
                       num-iterations {:seed (:seed opts) :pick-key ::core/pick})))))
