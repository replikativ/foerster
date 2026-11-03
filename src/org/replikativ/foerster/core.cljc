(ns org.replikativ.foerster.core
  "The public entry to inference: one call shape for every method (`infer`),
  the methods' own entry points, and the modelling primitives re-exported.

  Every method runs a probabilistic program as a savepoint handler: SMC
  (`foerster.inference.smc`) for the particle methods — smc-infer,
  importance-sampling, pimh-infer, pgibbs-infer, pgas-infer, ipmcmc-infer
  (`foerster.inference.pmcmc`), bbvi-infer (`foerster.inference.bbvi`) and
  kernel-infer with a PInferenceKernel — and replay plus accept over traces
  (`foerster.runtime.trace`) for the Markov-chain kernels.

  Pure inference (`:world-policy :fresh`, the default) runs in fresh worlds;
  `:world-policy :fork` in canonical forks of the caller's world
  (`foerster.inference.worlds`; methods that run several times fork the
  caller once, `foerster.runtime.canonical`). Particle measures hold
  `Sample`s (result, trace, and a canonical particle's world descriptor).
  Errors keep this namespace's keywords wherever they are raised.

  All functions return Spin<EmpiricalMeasure> for composability;
  post-processing is measure-centric (query, predict:
  `foerster.inference.posterior`)."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.moves.kernel :as k]
            [org.replikativ.foerster.moves.transition :as transition]
            [org.replikativ.foerster.inference.smc2 :as smc2]
            [org.replikativ.foerster.inference.worlds
             :refer [check-options! refuse-authority! particle-options smc-options
                     on-savepoints particles enumeration]]
            [org.replikativ.foerster.inference.pmcmc :as pmcmc]
            [org.replikativ.foerster.inference.bbvi :as bbvi]
            [org.replikativ.foerster.inference.posterior :as posterior]
            [org.replikativ.foerster.inference.parallel-tempering :as parallel-tempering]
            [org.replikativ.foerster.inference.tempering :as tempering]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.foerster.runtime.canonical :as canonical :refer [frozen-source with-frozen-caller]]
            #?(:clj [org.replikativ.foerster.analysis :as analysis])
            #?(:clj [org.replikativ.foerster.inference.compiled :as compiled])
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :as effects]
            [org.replikativ.spindel.engine.effects :as eff]))

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
;; Kernels: Markov chains and site proposal policies
;; =============================================================================

(defn- kernel-policy
  "An `foerster.runtime.trace` policy that asks `kernel` (a PInferenceKernel) for
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
                          (itrace/world-legacy-trace world))]
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
      execute each particle (each chain) in a frozen canonical Yggdrasil
      world that is discarded after the final values are captured; chains
      fork one frozen fork of the caller made when inference starts and
      reuse nothing incrementally (`foerster.runtime.trace/reexecute`)
    - :world-opts - Optional :systems/:rights/:snapshots policy forwarded to
      canonical forks; lifecycle fields are owned by inference
    - :authority, :grant - budgets (doc/worlds.md); particle kernels only

  Returns: Spin<EmpiricalMeasure>

  Examples:
    ;; Importance sampling with prior kernel
    (spin
      (let [model (coin-flip-model)
            measure (await (kernel-infer model (prior-kernel) 100
                                         {:barrier-policy :none}))]
        (query measure identity)))"
  [model-task kernel num-particles & [opts]]
  (when (k/markov-kernel? kernel)
    (refuse-authority! opts "A Markov chain"))
  (check-options! opts (if (k/markov-kernel? kernel)
                         #{:executor :world-policy :world-opts :seed}
                         (conj particle-options :barrier-policy)))
  (let [smc-opts (cond-> (dissoc opts :barrier-policy)
                   (= :none (:barrier-policy opts)) (assoc :resample-threshold 0.0))]
    (cond
      (k/markov-kernel? kernel)
      ;; Markov-chain kernels are replay plus accept over traces; each of the
      ;; `num-particles` is an independent chain. Under :fork every chain
      ;; forks the inference's one frozen fork of the caller's world.
      (with-frozen-caller opts #(transition/infer model-task kernel num-particles %))

      (= :prior (k/kernel-id kernel))
      (particles model-task num-particles smc-opts)

      :else
      (particles model-task num-particles (assoc smc-opts :policy (kernel-policy kernel))))))

;; =============================================================================
;; Particle methods (worlds and options: `foerster.inference.worlds`)
;; =============================================================================

(defn smc-infer
  "Sequential Monte Carlo: `num-particles` particles run `model-task` (a
  spin); at every observation the population is resampled when its
  effective sample size falls below `:resample-threshold`·N. Runs on
  savepoints (`foerster.inference.smc/smc`); the measure holds `Sample`s, and its
  `m/log-marginal` estimates the evidence.

  Options:
    :resample-threshold  ESS fraction below which to resample (default 0.5)
    :world-policy        :fresh (default): fresh worlds; :fork: canonical
                         forks of the caller's world (doc/worlds.md)
    :world-opts          fork options for :fork (:systems, :rights, :snapshots)
    :authority, :grant   a world.scope/PResourceAuthority and the budget the
                         inference draws from the caller's wallet (:fork)
    :executor            the executor the worlds run on (default: spindel's)
    :policy              a `foerster.runtime.trace/policy` deciding the sites
                         (constraints, interventions, proposals)
    :anchors, :rejuvenate  resample-move: after each resampling, MH moves
                         from anchored sites (`foerster.inference.smc/smc`; :fresh only)

  Returns a spin resolving the EmpiricalMeasure.

    (sp/with-context world @(smc-infer (model) 1000))   ; at the REPL
    (spin (query (await (smc-infer (model) 1000)) identity))"
  [model-task num-particles & [opts]]
  (check-options! opts (conj (into particle-options smc-options) :batch))
  (particles model-task num-particles opts))

(defn tempered-infer
  "Tempered SMC (`foerster.inference.tempering`): `num-particles` complete runs of
  `model-task`, moved from the prior to the posterior through the targets
  p(x)·L(x)^β with an adaptive schedule of β. Explores posteriors whose
  modes a Markov chain cannot cross, and estimates the evidence.

  Options: `:ess-target` (0.5), `:moves` (single-site MH moves per particle
  per step; default a sweep), `:scale` (2.38), `:waste-free` P, `:max-steps`,
  `:executor`, `:seed` (fixes every draw), `:world-policy` (:fork: every
  particle in a fork of one frozen fork of the caller's world, without
  incremental reuse) and `:world-opts`; no `:authority` yet.

  Returns a spin resolving the EmpiricalMeasure; `:temperatures` holds the
  schedule."
  [model-task num-particles & [opts]]
  (refuse-authority! opts "Tempered SMC")
  (check-options! opts #{:ess-target :moves :scale :waste-free :max-steps :executor :seed :world-policy :world-opts})
  (with-frozen-caller opts
    #(on-savepoints (tempering/tempered model-task num-particles (canonical/run-options %)))))

(defn parallel-tempering-infer
  "Non-reversible parallel tempering (`foerster.inference.parallel-tempering`) of a
  `model-task` that is one block site with a `:sample` and
  `:sample-log-density`: `num-chains` chains on the path from that draw
  density to the posterior, swapping states between neighbours, with the
  schedule tuned over `:rounds` rounds of doubling length (default 10).

  Options: `:rounds`, `:steps` (random-walk steps per chain per scan),
  `:executor`, `:seed` (fixes every draw), `:world-policy` (:fork: the
  program runs in a fork of the caller's world) and `:world-opts`; no
  `:authority` yet.

  Returns a spin resolving the EmpiricalMeasure of the last round's
  target-chain states; `m/log-marginal` is the stepping-stone evidence and
  `:temperatures`, `:barrier`, `:local-barrier`, `:round-trips` and
  `:swap-acceptance` diagnose the run."
  [model-task num-chains & [opts]]
  (refuse-authority! opts "Parallel tempering")
  (check-options! opts #{:rounds :steps :executor :seed :world-policy :world-opts})
  (with-frozen-caller opts
    #(on-savepoints (parallel-tempering/parallel-tempering
                     model-task (assoc (canonical/run-options %) :chains num-chains)))))

(defn importance-sampling
  "Importance sampling: `num-samples` runs of `model-task`, each weighted by
  its observations, never resampled (savepoint SMC with
  `:resample-threshold` 0). Options and result as for `smc-infer`."
  [model-task num-samples & [opts]]
  (check-options! opts (disj particle-options :resample-threshold))
  ;; savepoint SMC that never resamples: ESS never falls below 0
  (particles model-task num-samples (assoc opts :resample-threshold 0.0)))

(def ^{:arglists '([model-task num-particles num-iterations & [opts]])} pimh-infer
  "Particle independent Metropolis-Hastings. See `org.replikativ.foerster.inference.pmcmc/pimh-infer`."
  pmcmc/pimh-infer)

(def ^{:arglists '([model-task num-particles num-iterations & [opts]])} pgibbs-infer
  "Particle Gibbs (conditional SMC). See `org.replikativ.foerster.inference.pmcmc/pgibbs-infer`."
  pmcmc/pgibbs-infer)

(def ^{:arglists '([model-task num-particles num-iterations & [opts]])} pgas-infer
  "Particle Gibbs with ancestor sampling. See `org.replikativ.foerster.inference.pmcmc/pgas-infer`."
  pmcmc/pgas-infer)

(def ^{:arglists '([model-task num-particles num-iterations & [opts]])} ipmcmc-infer
  "Interacting particle MCMC. See `org.replikativ.foerster.inference.pmcmc/ipmcmc-infer`."
  pmcmc/ipmcmc-infer)

(def ^{:arglists '([model-task num-particles num-iterations & [opts]])} bbvi-infer
  "Black-box variational inference. See `org.replikativ.foerster.inference.bbvi/bbvi-infer`."
  bbvi/bbvi-infer)

(def ^{:arglists '([measure])} get-variational-dists
  "The learned {address distribution} of a `bbvi-infer` result. See `org.replikativ.foerster.inference.bbvi/get-variational-dists`."
  bbvi/get-variational-dists)

;; =============================================================================
;; Reading a posterior (`foerster.inference.posterior`)
;; =============================================================================

(def ^{:arglists '([measure query-fn])} query
  "Weighted statistics of a function of the program's value. See `org.replikativ.foerster.inference.posterior/query`."
  posterior/query)

(def ^{:arglists '([measure pred-fn num-samples])} predict
  "Draws from a measure, each passed to `pred-fn`. See `org.replikativ.foerster.inference.posterior/predict`."
  posterior/predict)

(def ^{:arglists '([model measure n & [{:keys [interventions]}]])} predictive
  "Posterior predictive draws of `model`. See `org.replikativ.foerster.inference.posterior/predictive`."
  posterior/predictive)

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
     :mh chains :rmh (conj chains :step-size) :nuts (conj chains :step-size)
     :kernel #{:kernel :chains :particles}
     :pmmh #{:particles :iterations :burn}}))

(defn- pmmh
  "PMMH (`smc2/pmmh`) in the worlds `opts`' `:world-policy` names: under
  `:fork` the initial θ is drawn, and every iteration's SMC runs, in
  canonical worlds forked from one frozen fork of the caller's world.
  No `:authority` yet."
  [model n-particles iterations opts]
  (refuse-authority! opts "PMMH")
  (with-frozen-caller opts
    (fn [o]
      (on-savepoints
       (smc2/pmmh model n-particles iterations
                  (if-let [source (frozen-source o)]
                    (assoc (dissoc o :world-policy ::source :world-opts)
                           :canonical-source source :world-opts (:world-opts o)
                           :filter (fn [model n smc-opts]
                                     (particles model n (assoc smc-opts :world-policy :fork ::source source
                                                               :world-opts (:world-opts o)))))
                    (dissoc o :world-policy)))))))

(defn- compiled-rmh
  "`:rmh` on the model compiled (`foerster.inference.compiled`): the spin must come
  from an `analysis/model` whose structure is static given its arguments.
  Opt-in and JVM-only; continuous latents, single-site random-walk MH with
  a fixed step; the chains run on `:executor` (see `foerster.inference.compiled`)."
  #?(:clj [model iterations burn step-size chains opts]
     :cljs [_model _iterations _burn _step-size _chains _opts])
  #?(:clj
     (let [_ (when-let [misplaced (seq (filter #{:world-policy :world-opts :authority :grant} (keys opts)))]
               ;; Compiled code has no worlds: it cannot fork the caller's
               ;; world, and an option it would ignore is refused. A model
               ;; that needs application state reads it once and passes it
               ;; in as data.
               (when (or (not= #{:world-policy} (set misplaced)) (not= :fresh (:world-policy opts)))
                 (throw (ex-info (str ":compiled? runs without worlds; read application state once and "
                                      "pass it to the model as data. Refused: " (vec misplaced))
                                 {:type ::invalid-world-policy :options (vec misplaced)
                                  :world-policy (:world-policy opts) :supported #{:fresh}}))))
           [m args] (or (analysis/applied model)
                        (throw (ex-info ":compiled? needs a model made by foerster.analysis/model"
                                        {:type ::not-an-analysis-model})))]
       (on-savepoints (compiled/rmh (apply compiled/compile-model m args)
                                    (cond-> {:iterations iterations :burn (or burn 0) :chains (or chains 4)
                                             :seed (:seed opts) :executor (:executor opts)}
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
                                        :max-branches; :policy, :seed,
                                        :world-policy as for :smc)
    :importance :smc :tempered          :particles
    :parallel-tempering                 :chains (temperatures, default 10),
                                        :rounds; one block site
    :pimh :pgibbs :pgas :ipmcmc :bbvi   :particles :iterations
    :mh :rmh :nuts                      :iterations per chain, :chains (default
                                        4), :burn, :step-size (:rmh, :nuts);
                                        every draw after :burn is kept; :nuts
                                        adapts during :burn (block sites),
                                        from :step-size and :inverse-metric,
                                        each chain jittered around :init
                                        unless :jitter false
                                        (`k/nuts-kernel`). :rmh with
                                        :compiled? true (JVM) runs on the
                                        model compiled to plain functions
                                        (`foerster.inference.compiled`): a model from
                                        `analysis/model` whose structure is
                                        static, else refused with the
                                        reasons; its traces hold the latent
                                        values only
    :kernel                             :kernel (a `foerster.moves.kernel` kernel)
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
      :enumerate (enumeration model rest-opts)
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
      :nuts (let [nuts-opts [:target-accept :max-depth :metric :inverse-metric :jitter]]
              (kernel-infer model (k/nuts-kernel iterations (cond-> (merge chain-output (select-keys rest-opts nuts-opts))
                                                              step-size (assoc :step-size step-size)))
                            (or chains 4) (apply dissoc rest-opts nuts-opts)))
      :kernel (kernel-infer model kernel (or chains particles 4) rest-opts)
      :pmmh (pmmh model particles iterations (cond-> rest-opts burn (assoc :burn burn))))))

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
