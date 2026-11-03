(ns org.replikativ.foerster.inference.worlds
  "The worlds a particle method runs in, and the options every driver
  checks. `:world-policy :fresh` (the default) runs pure inference on
  savepoints in fresh worlds; `:fork` runs it in canonical forks of the
  caller's world (`in-canonical-worlds`), from one frozen fork of the caller
  when a method runs several times (`foerster.runtime.canonical`).

  Errors and option keys keep `foerster.core`'s keywords: they are what the
  public entry points raise and take."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.core :as-alias core]
            [org.replikativ.foerster.inference.smc :as smc]
            [org.replikativ.foerster.inference.enumerate :as enumerate]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.foerster.runtime.canonical :refer [close-canonical! canonical-recovery
                                                               balance-op frozen-source]]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await await-finalization]]))

(defn world-policy
  "The `:world-policy` of `opts`, :fresh by default; refuses any other."
  [opts]
  (let [policy (get opts :world-policy :fresh)]
    (when-not (#{:fresh :fork} policy)
      (throw (ex-info "Unknown inference world policy"
                      {:type ::core/invalid-world-policy
                       :world-policy policy
                       :supported #{:fresh :fork}})))
    policy))

(def particle-options
  "What every particle method takes."
  #{:world-policy :world-opts :authority :grant :executor :resample-threshold :policy :proposal :seed})

(def rejuvenation-options
  "Resample-move and SMCP3 (`foerster.inference.smc`): SMC and PIMH in fresh worlds
  only."
  #{:anchors :rejuvenate :smcp3})

(def smc-options
  "What SMC and PIMH take beyond every particle method's options."
  (into #{:resampling :genealogy?} rejuvenation-options))

(def fork-options
  "What applies to canonical worlds only (`:world-policy :fork`)."
  #{:world-opts :authority :grant})

(defn check-options!
  "Refuse options `allowed` does not name, and canonical-world options under
  `:world-policy :fresh`: an option that would be ignored is a mistake."
  [opts allowed]
  (when-let [unknown (seq (remove allowed (keys opts)))]
    (throw (ex-info (str "Unknown inference options: " (vec unknown))
                    {:type ::core/unknown-options
                     :unknown (vec unknown)
                     :allowed allowed})))
  (when (= :fresh (world-policy opts))
    (when-let [misplaced (seq (filter fork-options (keys opts)))]
      (throw (ex-info (str "Options for :world-policy :fork only: " (vec misplaced))
                      {:type ::core/fork-only-options
                       :options (vec misplaced)}))))
  (when (= :fork (world-policy opts))
    (when-let [misplaced (seq (filter rejuvenation-options (keys opts)))]
      (throw (ex-info (str "Options for :world-policy :fresh only: " (vec misplaced))
                      {:type ::core/fresh-only-options
                       :options (vec misplaced)}))))
  opts)

(defn refuse-authority!
  "Methods whose runs fork worlds per move (chains, tempered particles)
  take no `:authority` or `:grant` yet."
  [opts what]
  (when-let [given (seq (filter #(contains? opts %) [:authority :grant]))]
    (throw (ex-info (str what " does not take :authority or :grant yet: every move forks a world")
                    {:type ::core/invalid-world-policy :options (vec given)}))))

(defn on-savepoints?
  "Whether a particle method whose canonical worlds are not on savepoints
  yet runs on savepoint SMC: pure inference in fresh worlds does."
  [opts]
  (= :fresh (world-policy opts)))

(defn on-savepoints
  "A spin resolving the savepoint CPS `operation`, a failure reported as
  `::core/inference-failed` with the model's error as its cause."
  [operation]
  (spin-core/with-causal-descendant-egress
    (spin
     (try
       (await operation)
       (catch #?(:clj Throwable :cljs :default) e
         (throw (ex-info "Inference failed during particle execution"
                         {:type ::core/inference-failed} e)))))))

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

(defn with-world-descriptors
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

(defn in-canonical-worlds
  "A savepoint driver's run of `model-task` in canonical worlds of the
  caller's (see above): `(run model root scope)` is a CPS operation resolving
  the measure of `model` run in `root` by a session joining `scope` —
  savepoint SMC (`smc/smc`) or enumeration (`enumerate/enumerate`).
  `:world-opts` are the forks' options (`:systems`, `:rights`,
  `:snapshots`); `:executor` the worlds' executor; `:authority` a
  `world.scope/PResourceAuthority` and `:grant` what the inference may spend
  of the caller's wallet. Inside `with-frozen-caller` the root is forked
  from the inference's frozen fork instead of the caller's world, and takes
  `::core/grant` of its wallet, or all that is left of it."
  [model-task opts run]
  (spin-core/with-causal-descendant-egress
    (spin
     (let [source (or (frozen-source opts) rtc/*execution-context*)
           grant (if (frozen-source opts)
                   (when (:authority opts)
                     (if (contains? opts ::core/grant)
                       (::core/grant opts)
                       (await (balance-op (:authority opts) source))))
                   (:grant opts))
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
         ;; the fork resolves in its source's world: resume this spin in
         ;; the caller's, also when the source is a frozen fork
         (vreset! root (:child-ctx (await (fn [resolve reject]
                                            (let [[resolve reject] (sp/in-callers-world resolve reject)]
                                              (world-scope/fork! scope source {:grant grant}
                                                                 resolve reject))))))
         (itrace/mark-canonical! @root)
         (vreset! measure
                  (await (run (binding [rtc/*execution-context* @root]
                                ;; every particle runs the whole model: a
                                ;; canonical model's effects may be random
                                ;; without a sample site
                                (spin (sp/savepoint smc/start-site nil)
                                      (await model-task)))
                              @root scope)))
         (catch #?(:clj Throwable :cljs :default) e
           (throw (if (= spin-core/spin-cancelled (:type (ex-data e)))
                    e
                    (ex-info "Inference failed during particle execution"
                             {:type ::core/inference-failed
                              :world/recovery (canonical-recovery scope lease @root)}
                             e))))
         (finally
           (await-finalization (close-canonical! scope lease @root))))
       (with-world-descriptors @measure (world-scope/descriptors scope))))))

(defn canonical-options
  "A savepoint driver's options in the canonical worlds `root` of `scope`:
  particles (and enumeration's branches) are copies of worlds on the JVM, so
  a world holding a system that may not be copied is refused and each copy
  takes an even share of its world's budget; forks in ClojureScript."
  [opts root scope]
  (-> opts
      (dissoc :world-policy :world-opts :executor :authority :grant ::core/source ::core/grant)
      (assoc :root root :scope scope :copy? #?(:clj true :cljs false))))

(defn particles
  "Savepoint SMC of `model-task` with `n` particles in the worlds `opts`'
  `:world-policy` names."
  [model-task n opts]
  (case (world-policy opts)
    :fresh (on-savepoints (smc/smc model-task n opts))
    :fork (in-canonical-worlds model-task opts
                               (fn [model root scope]
                                 (smc/smc model n (canonical-options opts root scope))))))

(def enumeration-options
  #{:max-branches :policy :executor :seed :world-policy :world-opts :authority :grant})

(defn enumeration
  "Exact enumeration (`foerster.inference.enumerate`) of `model-task` in the worlds
  `opts`' `:world-policy` names: under `:fork` the branches start from a
  canonical fork of the caller's world, as SMC's particles do, see each
  their own copy of its registered state, and are discarded before the
  measure is delivered; its outcomes carry their worlds' descriptors."
  [model-task opts]
  (check-options! opts enumeration-options)
  (case (world-policy opts)
    :fresh (on-savepoints (enumerate/enumerate model-task (dissoc opts :world-policy)))
    :fork (in-canonical-worlds model-task opts
                               (fn [model root scope]
                                 (enumerate/enumerate model (canonical-options opts root scope))))))
