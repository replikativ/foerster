(ns org.replikativ.foerster.core
  "Compositional probabilistic inference algorithms.

  Every method runs a probabilistic program as a savepoint handler: SMC
  (`foerster.smc`) for the particle methods — smc-infer,
  importance-sampling, pimh-infer, pgibbs-infer, pgas-infer, ipmcmc-infer,
  bbvi-infer and kernel-infer with a PInferenceKernel — and replay plus
  accept over traces (`foerster.trace`) for the Markov-chain kernels.

  Pure inference (`:world-policy :fresh`, the default) runs in fresh worlds;
  `:world-policy :fork` in canonical forks of the caller's world (see
  `in-canonical-worlds`). Particle measures hold `Sample`s (result, trace,
  and a canonical particle's world descriptor).

  All functions return Spin<EmpiricalMeasure> for composability;
  post-processing is measure-centric (query, predict)."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.transition :as transition]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.particle-mcmc :as pmcmc]
            [org.replikativ.foerster.smc2 :as smc2]
            [org.replikativ.foerster.enumerate :as enumerate]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.parallel-tempering :as parallel-tempering]
            [org.replikativ.foerster.tempering :as tempering]
            [org.replikativ.foerster.gradient :as grad]
            [org.replikativ.foerster.trace :as itrace]
            #?(:clj [org.replikativ.foerster.analysis :as analysis])
            #?(:clj [org.replikativ.foerster.compiled :as compiled])
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.combinators :as comb]
            [org.replikativ.spindel.effects.await :refer [await await-finalization]]
            [replikativ.logging :as log]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :as effects]
            [org.replikativ.spindel.engine.effects :as eff]
            [clojure.set :as set]))

;; =============================================================================
;; Modelling primitives
;; =============================================================================

;; Re-exported from foerster.effects so that a model needs one require besides
;; the distributions:
;;   (require '[org.replikativ.foerster.core :as infer
;;              :refer [sample observe factor]])
;; Effects are recognized by their qualified symbol, so the re-exported names
;; are registered with the same handlers (as spindel.core does for await).

(def sample "See `org.replikativ.foerster.effects/sample`." effects/sample)
(def observe "See `org.replikativ.foerster.effects/observe`." effects/observe)
(def factor "See `org.replikativ.foerster.effects/factor`." effects/factor)
(def deterministic "See `org.replikativ.foerster.effects/deterministic`." effects/deterministic)
(def intervene! "See `org.replikativ.foerster.effects/intervene!`." effects/intervene!)

(eff/register-effect-by-symbol!
 'org.replikativ.foerster.core/sample effects/choose-handler effects/sample-adapter)
(eff/register-effect-by-symbol!
 'org.replikativ.foerster.core/observe effects/choose-handler effects/observe-adapter)
(eff/register-effect-by-symbol!
 'org.replikativ.foerster.core/factor effects/factor-handler effects/factor-adapter)
(eff/register-effect-by-symbol!
 'org.replikativ.foerster.core/deterministic effects/deterministic-handler effects/deterministic-adapter)

;; =============================================================================
;; Markov chains: replay plus accept
;; =============================================================================

(defn- world-policy
  "The `:world-policy` of `opts`, :fresh by default; refuses any other."
  [opts]
  (let [policy (get opts :world-policy :fresh)]
    (when-not (#{:fresh :fork} policy)
      (throw (ex-info "Unknown inference world policy"
                      {:type ::invalid-world-policy
                       :world-policy policy
                       :supported #{:fresh :fork}})))
    policy))

(def ^:private particle-options
  "What every particle method takes."
  #{:world-policy :world-opts :authority :grant :executor :resample-threshold :policy :seed})

(def ^:private rejuvenation-options
  "Resample-move and SMCP3 (`foerster.smc`): SMC and PIMH in fresh worlds
  only."
  #{:anchors :rejuvenate :smcp3})

(def ^:private smc-options
  "What SMC and PIMH take beyond every particle method's options."
  (into #{:resampling :genealogy?} rejuvenation-options))

(def ^:private fork-options
  "What applies to canonical worlds only (`:world-policy :fork`)."
  #{:world-opts :authority :grant})

(defn- check-options!
  "Refuse options `allowed` does not name, and canonical-world options under
  `:world-policy :fresh`: an option that would be ignored is a mistake."
  [opts allowed]
  (when-let [unknown (seq (remove allowed (keys opts)))]
    (throw (ex-info (str "Unknown inference options: " (vec unknown))
                    {:type ::unknown-options
                     :unknown (vec unknown)
                     :allowed allowed})))
  (when (= :fresh (world-policy opts))
    (when-let [misplaced (seq (filter fork-options (keys opts)))]
      (throw (ex-info (str "Options for :world-policy :fork only: " (vec misplaced))
                      {:type ::fork-only-options
                       :options (vec misplaced)}))))
  (when (= :fork (world-policy opts))
    (when-let [misplaced (seq (filter rejuvenation-options (keys opts)))]
      (throw (ex-info (str "Options for :world-policy :fresh only: " (vec misplaced))
                      {:type ::fresh-only-options
                       :options (vec misplaced)}))))
  opts)

(declare particles)

(defn- kernel-policy
  "An `foerster.trace` policy that asks `kernel` (a PInferenceKernel) for
  every latent site's value; the value's `:log-weight-delta` (default 0) is
  what it adds to the particle's weight."
  [kernel]
  (itrace/policy
   {:draw (fn [sp _old-entry]
            (let [world (:savepoint/world sp)
                  {:keys [dist options]} (:savepoint/payload sp)
                  {:keys [value log-weight-delta]}
                  (k/step kernel world
                          {:source dist :options options :address (:savepoint/address sp)}
                          (itrace/legacy-trace (rtp/get-state world [:savepoint/trace])))]
              {:value value
               :log-proposal (- (dist/logpdf dist value) (or log-weight-delta 0.0))}))}))

(defn kernel-infer
  "Run inference with a kernel.

  Markov-chain kernels (`single-site-mh-kernel`, `random-walk-mh-kernel`,
  `block-gibbs-kernel`, `hmc-kernel`) run `num-particles` independent chains.
  Any other PInferenceKernel runs savepoint SMC whose latent sites take the
  value the kernel's `step` gives (the prior kernel: a draw from the prior).

  Args:
  - model-task: Spin (from model function) - Probabilistic program to infer
  - kernel: a kernel (e.g., prior-kernel, single-site-mh-kernel)
  - num-particles: Number of particles (chains)
  - opts: Optional map with:
    - :barrier-policy - :every-observe (default, SMC) | :none (importance
      sampling)
    - :resample-threshold - ESS threshold (default 0.5)
    - :executor - Shared executor for all particles
    - :seed - the run's seed: the same seed, the same draws, whatever the
      executor; without it the process generator (`random/set-seed!`) seeds
      the run
    - :world-policy - :fresh (default) for pure inference, or :fork to
      execute each particle in a frozen canonical Yggdrasil world that is
      discarded after the final particle values are captured
    - :world-opts - Optional :systems/:rights/:snapshots policy forwarded to
      canonical particle forks; lifecycle fields are owned by inference

  Returns: Spin<EmpiricalMeasure>

  Examples:
    ;; Importance sampling with prior kernel
    (spin
      (let [model (coin-flip-model)
            measure (await (kernel-infer model (prior-kernel) 100
                                         {:barrier-policy :none}))]
        (query measure identity)))"
  [model-task kernel num-particles & [opts]]
  (check-options! opts (if (k/markov-kernel? kernel)
                         #{:executor :world-policy :seed}
                         (conj particle-options :barrier-policy)))
  (let [smc-opts (cond-> (dissoc opts :barrier-policy)
                   (= :none (:barrier-policy opts)) (assoc :resample-threshold 0.0))]
    (cond
      (k/markov-kernel? kernel)
      ;; Markov-chain kernels are replay plus accept over traces; each of the
      ;; `num-particles` is an independent chain.
      (transition/infer model-task kernel num-particles opts)

      (= :prior (k/kernel-id kernel))
      (particles model-task num-particles smc-opts)

      :else
      (particles model-task num-particles (assoc smc-opts :policy (kernel-policy kernel))))))

;; =============================================================================
;; Convenience Functions (Delegate to kernel-infer)
;; =============================================================================

(defn- on-savepoints?
  "Whether a particle method whose canonical worlds are not on savepoints
  yet runs on savepoint SMC: pure inference in fresh worlds does."
  [opts]
  (= :fresh (world-policy opts)))

(defn- on-savepoints
  "A spin resolving the savepoint CPS `operation`, a failure reported as
  `::inference-failed` with the model's error as its cause."
  [operation]
  (spin-core/with-causal-descendant-egress
    (spin
     (try
       (await operation)
       (catch #?(:clj Throwable :cljs :default) e
         (throw (ex-info "Inference failed during particle execution"
                         {:type ::inference-failed} e)))))))

;; -----------------------------------------------------------------------------
;; Canonical particle worlds (`:world-policy :fork`)
;; -----------------------------------------------------------------------------
;;
;; The model runs in a frozen fork of the caller's world, the root, owned by
;; a world scope of the inference; savepoint SMC opens its session there and
;; its worlds join that scope, so every particle world descends from the root,
;; sees the caller's systems as they were, and nothing a particle writes
;; reaches the caller. Each particle runs the whole model (`smc/start-site`).
;; Particles are made by copying worlds (JVM): a world holding a system that
;; may not be copied is refused, and with an `:authority` the root is granted
;; `:grant` from the caller's wallet and every copy an even share of its
;; world's. When inference ends, however it ends, the session is closed and
;; every world discarded before the result or error is delivered; particles
;; keep their worlds' descriptors.

(defn- close-canonical!
  "CPS: give up the inference's `lease` on `scope`, then close the session in
  `root` (cancelling and joining its worlds, discarding the scope) — or,
  with no session, discard the scope once quiescent, a root fork still in
  flight included. Idempotent."
  [scope lease root]
  (fn [resolve reject]
    (world-scope/end-activity! scope lease)
    (if-let [session (some-> root sp/session)]
      ((sp/close! session) resolve reject)
      ((world-scope/discard-when-quiescent! scope) resolve reject))))

(defn- canonical-recovery
  "What a host needs when a canonical inference failed: the worlds'
  descriptors and the operations that finish their cleanup, should the
  automatic one have failed. Process-local."
  [scope lease root]
  {:status (:status @scope)
   :manager scope
   :await-quiescent (world-scope/await-quiescence scope)
   :cancel! #(world-scope/request-cancel! scope)
   :discard! #(close-canonical! scope lease root)
   :descriptors (world-scope/descriptors scope)})

(defn- with-world-descriptors
  "`measure` whose particles carry their worlds' settled descriptors."
  [measure descriptors]
  (let [by-id (into {} (map (juxt :fork/id identity)) descriptors)]
    (update measure :particles
            (fn [particles]
              (mapv (fn [[s w]]
                      [(if-let [id (:world-id s)]
                         (-> s (dissoc :world-id) (assoc :world-descriptor (get by-id id)))
                         s)
                       w])
                    particles)))))

(defn- in-canonical-worlds
  "Savepoint SMC (`smc/smc`) of `model-task` with `n` particles in canonical
  worlds of the caller's (see above). `:world-opts` are the forks' options
  (`:systems`, `:rights`, `:snapshots`); `:executor` the worlds' executor;
  `:authority` a `world.scope/PResourceAuthority` and `:grant` what the
  inference may spend of the caller's wallet."
  [model-task n opts]
  (spin-core/with-causal-descendant-egress
    (spin
     (let [caller rtc/*execution-context*
           world-opts (or (:world-opts opts) {})
           scope (world-scope/create {:purpose :particle
                                      :authority (:authority opts)
                                      :fork-opts (cond-> world-opts
                                                   (:executor opts) (assoc :executor (:executor opts)))})
         ;; holds the scope open until the session joins it
           lease (world-scope/begin-activity! scope :inference)
           root (volatile! nil)
           measure (volatile! nil)]
       (try
         (vreset! root (:child-ctx (await (fn [resolve reject]
                                            (world-scope/fork! scope caller {:grant (:grant opts)}
                                                               resolve reject)))))
         (rtp/swap-state! @root [:inference :canonical?] (constantly true))
         (vreset! measure
                  (await (smc/smc (binding [rtc/*execution-context* @root]
                                  ;; every particle runs the whole model: a
                                  ;; canonical model's effects may be random
                                  ;; without a sample site
                                    (spin (sp/savepoint smc/start-site nil)
                                          (await model-task)))
                                  n
                                  (-> opts
                                      (dissoc :world-policy :world-opts :executor :authority :grant)
                                      (assoc :root @root :scope scope
                                             :copy? #?(:clj true :cljs false))))))
         (catch #?(:clj Throwable :cljs :default) e
           (throw (if (= spin-core/spin-cancelled (:type (ex-data e)))
                    e
                    (ex-info "Inference failed during particle execution"
                             {:type ::inference-failed
                              :world/recovery (canonical-recovery scope lease @root)}
                             e))))
         (finally
           (await-finalization (close-canonical! scope lease @root))))
       (with-world-descriptors @measure (world-scope/descriptors scope))))))

(defn- particles
  "Savepoint SMC of `model-task` with `n` particles in the worlds `opts`'
  `:world-policy` names."
  [model-task n opts]
  (case (world-policy opts)
    :fresh (on-savepoints (smc/smc model-task n opts))
    :fork (in-canonical-worlds model-task n opts)))

(defn smc-infer
  "Sequential Monte Carlo: `num-particles` particles run `model-task` (a
  spin); at every observation the population is resampled when its
  effective sample size falls below `:resample-threshold`·N. Runs on
  savepoints (`foerster.smc/smc`); the measure holds `Sample`s, and its
  `m/log-marginal` estimates the evidence.

  Options:
    :resample-threshold  ESS fraction below which to resample (default 0.5)
    :world-policy        :fresh (default): fresh worlds; :fork: canonical
                         forks of the caller's world (doc/worlds.md)
    :world-opts          fork options for :fork (:systems, :rights, :snapshots)
    :authority, :grant   a world.scope/PResourceAuthority and the budget the
                         inference draws from the caller's wallet (:fork)
    :executor            the executor the worlds run on (default: spindel's)
    :policy              a `foerster.trace/policy` deciding the sites
                         (constraints, interventions, proposals)
    :anchors, :rejuvenate  resample-move: after each resampling, MH moves
                         from anchored sites (`foerster.smc/smc`; :fresh only)

  Returns a spin resolving the EmpiricalMeasure.

    (sp/with-context world @(smc-infer (model) 1000))   ; at the REPL
    (spin (query (await (smc-infer (model) 1000)) identity))"
  [model-task num-particles & [opts]]
  (check-options! opts (conj (into particle-options smc-options) :batch))
  (when (and (:batch opts) (not (on-savepoints? opts)))
    (throw (ex-info ":batch runs in fresh worlds" {:type ::invalid-world-policy :batch (:batch opts)})))
  (particles model-task num-particles opts))

(defn tempered-infer
  "Tempered SMC (`foerster.tempering`): `num-particles` complete runs of
  `model-task`, moved from the prior to the posterior through the targets
  p(x)·L(x)^β with an adaptive schedule of β. Explores posteriors whose
  modes a Markov chain cannot cross, and estimates the evidence.

  Options: `:ess-target` (0.5), `:moves` (single-site MH moves per particle
  per step; default a sweep), `:scale` (2.38), `:waste-free` P, `:max-steps`,
  `:executor`, `:seed` (fixes every draw); fresh worlds only.

  Returns a spin resolving the EmpiricalMeasure; `:temperatures` holds the
  schedule."
  [model-task num-particles & [opts]]
  (check-options! opts #{:ess-target :moves :scale :waste-free :max-steps :executor :seed :world-policy})
  (when (= :fork (world-policy opts))
    (throw (ex-info "Tempered SMC runs in fresh worlds" {:type ::invalid-world-policy
                                                         :world-policy :fork :supported #{:fresh}})))
  (on-savepoints (tempering/tempered model-task num-particles (dissoc opts :world-policy))))

(defn parallel-tempering-infer
  "Non-reversible parallel tempering (`foerster.parallel-tempering`) of a
  `model-task` that is one block site with a `:sample` and
  `:sample-log-density`: `num-chains` chains on the path from that draw
  density to the posterior, swapping states between neighbours, with the
  schedule tuned over `:rounds` rounds of doubling length (default 10).

  Options: `:rounds`, `:steps` (random-walk steps per chain per scan),
  `:executor`, `:seed` (fixes every draw); fresh worlds only.

  Returns a spin resolving the EmpiricalMeasure of the last round's
  target-chain states; `m/log-marginal` is the stepping-stone evidence and
  `:temperatures`, `:barrier`, `:local-barrier`, `:round-trips` and
  `:swap-acceptance` diagnose the run."
  [model-task num-chains & [opts]]
  (check-options! opts #{:rounds :steps :executor :seed :world-policy})
  (when (= :fork (world-policy opts))
    (throw (ex-info "Parallel tempering runs in fresh worlds" {:type ::invalid-world-policy
                                                               :world-policy :fork :supported #{:fresh}})))
  (on-savepoints (parallel-tempering/parallel-tempering model-task (assoc (dissoc opts :world-policy) :chains num-chains))))

(defn importance-sampling
  "Importance sampling: `num-samples` runs of `model-task`, each weighted by
  its observations, never resampled (savepoint SMC with
  `:resample-threshold` 0). Options and result as for `smc-infer`."
  [model-task num-samples & [opts]]
  (check-options! opts (disj particle-options :resample-threshold))
  ;; savepoint SMC that never resamples: ESS never falls below 0
  (particles model-task num-samples (assoc opts :resample-threshold 0.0)))

;; =============================================================================
;; Helper Functions
;; =============================================================================

(defn query
  "Weighted statistics of a numeric function of the program's value over
  `measure`: `query-fn` is `identity` (the value itself), a keyword (a field
  of a map value) or a function of the value.

  Returns {:mean :variance :std-dev :quantiles :samples :weights :type};
  `:samples` and `:weights` are the particles' values and normalized
  weights."
  [measure query-fn]
  (let [extract-fn (cond
                     ;; identity means "get the main result"
                     (= query-fn identity) m/get-value
                     ;; keyword means "extract field from result"
                     (keyword? query-fn) (fn [ctx] (get (m/get-value ctx) query-fn))
                     ;; function: compose with get-value
                     :else (fn [ctx] (query-fn (m/get-value ctx))))]
    (m/measure-stats measure extract-fn)))

(defn predict
  "`num-samples` draws from `measure`, resampled by weight, each passed to
  `pred-fn` — which gets the particle (a `Sample`, or a context), so
  `m/get-value` reads its program value and `m/get-trace` its trace."
  [measure pred-fn num-samples]
  (let [samples (m/sample-measure measure num-samples)]
    (mapv (fn [[ctx _]] (pred-fn ctx)) samples)))

(defn predictive
  "`n` posterior predictive draws: particles of `measure` drawn by weight,
  each replayed through `model` with its latent choices held and its
  observed sites drawing fresh values instead of scoring the data. A prior
  predictive draw runs `model` with `(itrace/policy {:simulate-observed?
  true})` (`gfi/run-policy`); `gfi/simulate` would score the data instead.

  `model` need not be the program that was conditioned: the same program on
  new inputs (more weeks, other covariates) predicts them, its latent sites
  held where the posterior has them and drawn from their prior where it
  does not (a new group's effect). `:interventions` ({selector transform},
  as `foerster.trace/policy` takes them) sets sites by the do-operator
  first — a scenario under the posterior.

  Returns a spin resolving a vector of {:value v :observations {address x}}:
  the program's value and what each observed site drew."
  [model measure n & [{:keys [interventions]}]]
  (spin
   (loop [[[particle _] & more] (m/sample-measure measure n) out []]
     (if-not particle
       out
       (let [_ (when (nil? (m/get-trace particle))
                 (throw (ex-info "Posterior predictive replay needs traces; use amortized/predictive with an explicit forward binding for learned draws"
                                 {:type ::missing-predictive-trace})))
             latents (into {} (keep (fn [[address {:keys [value observed? deterministic?]}]]
                                      (when-not (or observed? deterministic?) [address value])))
                           (m/get-trace particle))
             t (await (gfi/run-policy model (itrace/policy {:constraints latents
                                                            :interventions interventions
                                                            :simulate-observed? true})))
             drawn (into {} (mapcat (fn [[address {:keys [value note]}]]
                                      (cond
                                        (:simulated? note) [[address value]]
                                        ;; a block's own observations, when it can draw them
                                        (block/block-dist? (:dist note))
                                        (when-let [simulate (block/capability (:block (:dist note)) :simulate)]
                                          (map (fn [[k v]] [[address k] v])
                                               (simulate (double-array value) (:inputs (:dist note))))))))
                         (:trace/entries t))]
         (await (gfi/close! t))
         (recur more (conj out {:value (:trace/result t) :observations drawn})))))))

;; =============================================================================
;; Particle MCMC Methods
;; =============================================================================

(defn- particle-sweep [model n opts]
  (fn [key retained seed]
    (particles model n (cond-> (assoc opts :seed (sp/derive-seed seed ::sweep key))
                         retained (assoc :retained retained)))))

(defn pimh-infer
  "Particle Independent Metropolis-Hastings (Andrieu et al. 2010).

  Each iteration proposes a fresh SMC sweep and accepts it with probability
  min(1, Ẑ_new / Ẑ_current); the current sweep's particles, normalized, are
  emitted every iteration.

  Args:
    model-task, num-particles (per sweep), num-iterations
    opts: :executor, :resample-threshold

  Returns: Spin<EmpiricalMeasure>"
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts (into particle-options smc-options))
  (if (on-savepoints? opts)
    (on-savepoints (smc/pimh model-task num-particles num-iterations opts))
    (pmcmc/pimh
     (particle-sweep model-task num-particles opts)
     num-iterations {:seed (:seed opts) :accept-key ::pimh-accept})))

(defn pgibbs-infer
  "Particle Gibbs (conditional SMC, Andrieu et al. 2010).

  Args:
    model-task, num-particles (per sweep, including the retained one),
    num-iterations (sweeps)
    opts: the particle methods' options, and :anchors, :rejuvenate for
          resample-move inside each sweep, the retained particle included
          (`foerster.smc/smc`; :fresh only)

  Returns: Spin<EmpiricalMeasure> of every sweep's particles, each sweep
  normalized to total weight one."
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts (into particle-options #{:anchors :rejuvenate}))
  (if (on-savepoints? opts)
    (on-savepoints (smc/pgibbs model-task num-particles num-iterations opts))
    (pmcmc/pgibbs
     (particle-sweep model-task num-particles opts)
     num-iterations {:seed (:seed opts) :pick-key ::pick})))

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
              probs (m/normalize-log-weights proposal-log-Zs)

              ;; Update zeta sums for Rao-Blackwellization
              new-zeta-sums (reduce (fn [zs [idx p]]
                                      (update zs idx + p))
                                    zeta-sums
                                    (map vector proposal-indices probs))

              ;; Sample which node to use as CSMC
              k (m/sample-categorical probs)]

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
               (assoc opts :retained (pmcmc/retained-choices retained-trace)))))

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
                                                   (assoc opts :seed (sp/derive-seed (:seed opts) ::node i))))
                                      retained-traces))]
    ;; Use parallel combinator to run all sweeps concurrently
    (apply comb/parallel sweep-spins)))

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

   Returns: Spin<EmpiricalMeasure>

   Reference:
     Rainforth et al., 'Interacting Particle Markov Chain Monte Carlo', ICML 2016"
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts (into particle-options #{:num-nodes :num-csmc-nodes :all-particles?}))
  (spin
   (let [num-nodes (or (:num-nodes opts) 8)
         num-csmc-nodes (or (:num-csmc-nodes opts) (quot num-nodes 2))
         num-smc-nodes (- num-nodes num-csmc-nodes)
         all-particles? (get opts :all-particles? true)
         seed (or (:seed opts) (random/fresh-seed))
         sweep-opts #(assoc opts :seed (sp/derive-seed seed ::sweep %))]
     (assert (> num-csmc-nodes 0) ":num-csmc-nodes must be > 0")
     (assert (< num-csmc-nodes num-nodes) ":num-csmc-nodes must be < :num-nodes")
     (loop [iteration 0
            ;; iteration 0 runs plain SMC on every node
            measures (await (run-parallel-sweeps model-task num-particles
                                                 (vec (repeat num-nodes nil)) (sweep-opts :initial)))
            all-samples []]
       (if (>= iteration num-iterations)
         (m/markov-measure all-samples)
         (let [log-Zs (mapv m/log-marginal measures)
               ;; the iteration's Gibbs draws from its own stream of the run's seed
               [csmc-indices _zeta-sums samples retained-traces]
               (random/with-stream*
                 seed [::gibbs iteration]
                 (fn []
                   (let [[csmc-indices zeta-sums] (gibbs-update-csmc-indices log-Zs num-csmc-nodes)
                   ;; emit THESE sweeps, each node weighted by its Rao-Blackwellized
                   ;; probability of being a conditional node (ζ_j), particles
                   ;; normalized within the node
                         samples (vec (mapcat
                                       (fn [node-idx]
                                         (let [zeta (nth zeta-sums node-idx)
                                               node (pmcmc/normalized (nth measures node-idx))]
                                           (when (pos? zeta)
                                             (if all-particles?
                                               (map (fn [[smp lw]] [smp (+ lw (Math/log zeta))]) node)
                                               [[(first (nth node (m/sample-categorical
                                                                   (m/normalize-log-weights (mapv second node)))))
                                                 (Math/log zeta)]]))))
                                       (range num-nodes)))
                         retained-traces (vec (concat
                                               (map (fn [node-idx]
                                                      (let [ps (m/get-particles (nth measures node-idx))]
                                                        (m/get-trace (first (nth ps (m/sample-categorical
                                                                                     (m/normalize-log-weights (mapv second ps))))))))
                                                    csmc-indices)
                                               (repeat num-smc-nodes nil)))]
                     [csmc-indices zeta-sums samples retained-traces])))]
           (log/trace :ipmcmc/gibbs-update {:iteration iteration :csmc-indices csmc-indices})
           (recur (inc iteration)
                  (await (run-parallel-sweeps model-task num-particles retained-traces (sweep-opts iteration)))
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
    (pmcmc/pgibbs (particle-sweep model-task num-particles (assoc opts :ancestor-sampling? true))
                  num-iterations {:seed (:seed opts) :pick-key ::pick})))

;; =============================================================================
;; Black Box Variational Inference (BBVI)
;; =============================================================================

(defn- variational-draw
  "The proposal of a savepoint BBVI iteration: a latent site whose q (the
  site's prior, the first time its address is seen) has a gradient draws from
  q; `foerster.trace/policy` then weights it by p/q."
  [q-dists]
  (fn [sp _old-entry]
    (let [address (:savepoint/address sp)
          prior (:dist (:savepoint/payload sp))
          q (get (swap! q-dists #(if (contains? % address) % (assoc % address prior))) address)]
      (when (grad/has-gradient? q)
        (let [v (dist/draw q)]
          {:value v :log-proposal (dist/logpdf q v)})))))

(defn- q-gradients
  "{address ∇log q(value)} of the latent sites of a Sample's trace that
  `qs` has a differentiable q for."
  [qs trace]
  (into {} (keep (fn [[address {:keys [value observed?]}]]
                   (let [q (get qs address)]
                     (when (and (not observed?) q (grad/has-gradient? q))
                       [address (grad/compute-gradient q value)]))))
        trace))

(defn- optimal-scaling
  "Control variate coefficient Cov(f,g)/Var(g)."
  [f g]
  (let [n (count f)
        f-bar (/ (reduce + f) n)
        g-bar (/ (reduce + g) n)
        g-centered (mapv #(- % g-bar) g)]
    (/ (reduce + 1e-12 (map * (map #(- % f-bar) f) g-centered))
       (reduce + 1e-12 (map * g-centered g-centered)))))

(defn- aggregate-gradients
  "Per address, the score-function ELBO gradient
     (1/n) Σ (log w_i − a*) ∇log q(z_i),   log w = log p(x, y) − log q(x)
   with the variance-minimizing control variate a* = Cov(f, g)/Var(g),
   f = log w · g (Ranganath et al. 2014)."
  [particle-gradients particle-log-weights]
  (let [all-addrs (reduce set/union (map #(set (keys %)) particle-gradients))]
    (reduce
     (fn [result addr]
       (let [valid (filter (fn [[grads lw]] (and (contains? grads addr) (grad/finite? lw)))
                           (map vector particle-gradients particle-log-weights))
             n (count valid)]
         (if (< n 2)
           result
           (let [grads (mapv #(get (first %) addr) valid)
                 lws (mapv second valid)]
             (assoc result addr
                    (mapv (fn [dim]
                            (let [g (mapv #(nth % dim) grads)
                                  f (mapv * lws g)
                                  a (optimal-scaling f g)]
                              (/ (reduce + (map (fn [x y] (- x (* a y))) f g)) n)))
                          (range (count (first grads)))))))))
     {}
     all-addrs)))

(defn- update-variational-dists!
  "One ascent step on every q with a gradient. `adagrad`: true accumulates
   squared gradients (AdaGrad), a number γ decays them (RMSprop), false
   takes plain steps of size `lr`."
  [q-dists accumulators gradients lr adagrad]
  (doseq [[addr g] gradients]
    (let [dist (get @q-dists addr)]
      (when (grad/has-gradient? dist)
        (let [g2 (mapv #(* % %) g)
              prev (get @accumulators addr)
              acc (cond (nil? prev) g2
                        (number? adagrad) (mapv #(+ (* adagrad %1) (* (- 1.0 adagrad) %2)) prev g2)
                        :else (mapv + prev g2))
              rate (if adagrad (mapv #(/ lr (+ 1e-8 (Math/sqrt %))) acc) lr)]
          (swap! accumulators assoc addr acc)
          (swap! q-dists assoc addr (grad/grad-step dist g rate)))))))

(defn bbvi-infer
  "Black Box Variational Inference (Ranganath et al., AISTATS 2014).

   Learns a mean-field q(z) = Π_addr q_addr by stochastic ascent on the ELBO
   with the score-function estimator and control variates: `num-iterations`
   updates, each from `num-particles` programs that sample latents from q
   and weight by p(x, y)/q(x). Each site's q starts as the prior it has when
   first reached and from then on moves by gradient only: mean-field, it
   does not follow a prior that depends on other latents.

   Args:
     model-task, num-particles, num-iterations
     opts: :base-lr (1.0) — step size at iteration t is
           base-lr / (t+1)^robbins-monro
           :robbins-monro (0.0)
           :adagrad (true) — true: AdaGrad, γ ∈ (0,1): RMSprop with decay γ,
           false: plain steps
           :executor

   Returns: Spin<EmpiricalMeasure> — `num-particles` importance-weighted
   samples from the final q (with 0 iterations: from the priors); the learned
   q is under `:variational-dists` (see `get-variational-dists`)."
  [model-task num-particles num-iterations & [opts]]
  (check-options! opts (-> particle-options
                           (disj :policy :resample-threshold)
                           (into #{:base-lr :robbins-monro :adagrad})))
  (spin
   (let [base-lr (or (:base-lr opts) 1.0)
         robbins-monro (or (:robbins-monro opts) 0.0)
         adagrad (get opts :adagrad true)
         q-dists (atom {})
         accumulators (atom {})]
     (loop [iteration 0]
       (let [qs @q-dists
             measure (await (particles model-task num-particles
                                       (assoc opts
                                              :resample-threshold 0.0
                                              :policy (itrace/policy {:draw (variational-draw q-dists)}))))]
         (if (>= iteration num-iterations)
           (assoc measure :variational-dists @q-dists)
           (let [ps (m/get-particles measure)
                 grads (mapv (fn [[c _]] (q-gradients (merge @q-dists qs) (m/get-trace c)))
                             ps)]
             (update-variational-dists! q-dists accumulators
                                        (aggregate-gradients grads (mapv second ps))
                                        (/ base-lr (Math/pow (inc iteration) robbins-monro))
                                        adagrad)
             (recur (inc iteration)))))))))

(defn get-variational-dists
  "The learned {address -> distribution} of a `bbvi-infer` result."
  [measure]
  (:variational-dists measure))

;; =============================================================================
;; One call shape
;; =============================================================================

(def ^:private infer-methods
  #{:enumerate :importance :smc :tempered :parallel-tempering :pimh :pgibbs :pgas :ipmcmc :bbvi :mh :rmh :nuts :kernel :pmmh})

(def ^:private sizes-of-every-method
  {:particles true :iterations true :chains true :burn true :step-size true :kernel true})

(def ^:private method-sizes
  "The size options each method of `infer` takes; a kernel's `:burn` and
  output belong to the kernel itself."
  (let [particles #{:particles}
        sweeps #{:particles :iterations}
        chains #{:iterations :chains :burn}]
    {:enumerate #{} :importance particles :smc particles :tempered particles
     :parallel-tempering #{:chains}
     :pimh sweeps :pgibbs sweeps :pgas sweeps :ipmcmc sweeps :bbvi sweeps
     :mh chains :rmh (conj chains :step-size) :nuts chains
     :kernel #{:kernel :chains :particles}
     :pmmh #{:particles :iterations :burn}}))

(defn- compiled-rmh
  "`:rmh` on the model compiled (`foerster.compiled`): the spin must come
  from an `analysis/model` whose structure is static given its arguments."
  #?(:clj [model iterations burn step-size chains opts]
     :cljs [_model _iterations _burn _step-size _chains _opts])
  #?(:clj
     (let [[m args] (or (analysis/applied model)
                        (throw (ex-info ":compiled? needs a model made by foerster.analysis/model"
                                        {:type ::not-an-analysis-model})))]
       (on-savepoints (compiled/rmh (apply compiled/compile-model m args)
                                    (cond-> {:iterations iterations :burn (or burn 0) :chains (or chains 4)
                                             :seed (:seed opts)}
                                      step-size (assoc :step-size step-size)))))
     :cljs (throw (ex-info ":compiled? runs on the JVM" {:type ::compiled-on-cljs}))))

(defn infer
  "Run `model` under the inference method `(:method opts)` — one call shape
  for every method, as Anglican's `doquery`:

    (infer/infer (model) {:method :smc :particles 1000})
    (infer/infer (model) {:method :mh :iterations 4000 :chains 4 :burn 1000})
    (infer/infer (model) {:method :pmmh :particles 100 :iterations 2000
                          :params #{:drift}})

  Methods and their sizes:
    :enumerate                          (exact; finite supports only,
                                        :max-branches)
    :importance :smc :tempered          :particles
    :parallel-tempering                 :chains (temperatures, default 10),
                                        :rounds; one block site
    :pimh :pgibbs :pgas :ipmcmc :bbvi   :particles :iterations
    :mh :rmh :nuts                      :iterations per chain, :chains (default
                                        4), :burn, :step-size (:rmh); every
                                        draw after :burn is kept; :nuts adapts
                                        during :burn (block sites). :rmh with
                                        :compiled? true (JVM) runs on the
                                        model compiled to plain functions
                                        (`foerster.compiled`): a model from
                                        `analysis/model` whose structure is
                                        static, else refused with the
                                        reasons; its traces hold the latent
                                        values only
    :kernel                             :kernel (a `foerster.kernel` kernel)
                                        and :chains or :particles
    :pmmh                               :particles per SMC, :iterations,
                                        :params (and `smc2/pmmh` options)
  The other options go to the method (see its function). Returns a spin
  resolving the measure."
  [model {:keys [method particles iterations chains burn step-size kernel] :as opts}]
  (when-not (infer-methods method)
    (throw (ex-info (str "Unknown inference method " method)
                    {:type ::unknown-method :method method :methods infer-methods})))
  (let [allowed (method-sizes method)
        misplaced (remove allowed (filter #(contains? opts %) (keys sizes-of-every-method)))]
    (when (seq misplaced)
      (throw (ex-info (str "Options " (vec misplaced) " do not apply to " method)
                      {:type ::unknown-options :method method :options (vec misplaced)
                       :supported allowed}))))
  (let [rest-opts (dissoc opts :method :particles :iterations :chains :burn :step-size :kernel)
        chain-output {:samples :all :burn (or burn 0)}]
    (case method
      :enumerate (on-savepoints (enumerate/enumerate model rest-opts))
      :importance (importance-sampling model particles rest-opts)
      :smc (smc-infer model particles rest-opts)
      :tempered (tempered-infer model particles rest-opts)
      :parallel-tempering (parallel-tempering-infer model (or chains 10) rest-opts)
      :pimh (pimh-infer model particles iterations rest-opts)
      :pgibbs (pgibbs-infer model particles iterations rest-opts)
      :pgas (pgas-infer model particles iterations rest-opts)
      :ipmcmc (ipmcmc-infer model particles iterations rest-opts)
      :bbvi (bbvi-infer model particles iterations rest-opts)
      :mh (kernel-infer model (k/single-site-mh-kernel iterations chain-output) (or chains 4) rest-opts)
      :rmh (if (:compiled? rest-opts)
             (compiled-rmh model iterations burn step-size chains (dissoc rest-opts :compiled?))
             (kernel-infer model (k/random-walk-mh-kernel iterations (merge chain-output (when step-size {:step-size step-size})))
                           (or chains 4) rest-opts))
      :nuts (kernel-infer model (k/nuts-kernel iterations (merge chain-output (select-keys rest-opts [:target-accept :max-depth :metric])))
                          (or chains 4) (dissoc rest-opts :target-accept :max-depth :metric))
      :kernel (kernel-infer model kernel (or chains particles 4) rest-opts)
      :pmmh (on-savepoints (smc2/pmmh model particles iterations (cond-> rest-opts burn (assoc :burn burn)))))))

;; =============================================================================
;; Nested inference
;; =============================================================================

(defn conditional
  "Nested inference: a spin resolving the posterior of `model`'s value under
  `opts` (as for `infer`) as a distribution — a categorical over the values
  the inner inference found, weighted (Anglican's `conditional`). The outer
  program samples from it, or observes against it:

    (let [guess (await (infer/conditional (inner-model x) {:method :enumerate}))]
      (sample guess :id :their-guess))

  The inner inference runs in fresh worlds of its own. Under exact
  enumeration the distribution is exact; otherwise it is the inner measure,
  so it has as many atoms as distinct values. Memoize it (`process/mem`) when
  the outer program asks the same question repeatedly. A finite inner
  approximation may bias nonlinear outer computations; more outer draws do
  not remove fixed inner error, and memoization does not make it exact. The
  returned distribution carries :inference/nesting provenance metadata."
  [model opts]
  (spin
   (let [measure (await (infer model opts))
         ps (m/get-particles measure)
         ws (weights/normalize (mapv second ps))
         by-value (reduce (fn [acc [[particle _] w]]
                            (update acc (m/get-value particle) (fnil + 0.0) w))
                          {} (map vector ps ws))]
     (with-meta (dist/categorical (vec (filter (comp pos? second) by-value)))
       {:inference/nesting {:kind :empirical-conditional :method (:method opts)
                            :exact? (= :enumerate (:method opts)) :particles (count ps)}}))))
