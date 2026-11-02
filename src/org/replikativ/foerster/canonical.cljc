(ns org.replikativ.foerster.canonical
  "Canonical worlds for inference that runs several times (`:world-policy
  :fork`): one frozen fork of the caller's world per inference, from which
  every run (a particle-MCMC sweep, an IPMCMC node, a BBVI iteration, a
  Markov chain, a tempered-SMC particle, an inference program) forks a world
  and scope of its own; the scope's cleanup, its recovery capability, and
  the wallet arithmetic of `:authority`. See doc/worlds.md."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await await-finalization]]))

(defn close-canonical!
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

(defn canonical-recovery
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

(defn fork-scope-options
  "The world-scope options of a canonical inference's worlds."
  [opts purpose]
  {:purpose purpose
   :authority (:authority opts)
   :fork-opts (cond-> (or (:world-opts opts) {})
                (:executor opts) (assoc :executor (:executor opts)))})

(defn share
  "One of `k` even shares of the wallet `resources`: integer amounts rounded
  down, as `world.scope/copy!` splits a world's budget."
  [resources k]
  (when resources
    (into {} (map (fn [[r amount]] [r (if (integer? amount) (quot amount k) (/ amount k))]))
          resources)))

(defn balance-op
  "CPS: what is left in `world`'s wallet under `authority`, resolved in the
  caller's world."
  [authority world]
  (fn [resolve reject]
    (let [[resolve reject] (sp/in-callers-world resolve reject)]
      (try
        (let [left (world-scope/balance authority world)]
          (if (or (nil? left) (map? left)) (resolve left) (left resolve reject)))
        (catch #?(:clj Throwable :cljs :default) e (reject e))))))

(defn frozen-source
  "The frozen fork of the caller's world a multi-run inference runs every
  run from (`with-frozen-caller`), or nil."
  [opts]
  (:org.replikativ.foerster.core/source opts))

(defn with-frozen-caller
  "Under `:world-policy :fork`, `(body opts')` with `opts'` naming one frozen
  fork of the caller's world, made once for the whole inference: every run
  of a method that runs several (a particle-MCMC sweep, a BBVI iteration, an
  IPMCMC node, a Markov chain) forks its own world from it, so all of them
  see the caller's state as it was when inference started, whatever the
  caller does meanwhile, and the inference draws `:grant` from the caller's
  wallet once. The fork is owned by a scope of its own, discarded (its
  wallet given back) after `body` ends, however it ends. `body` returns a
  spin. Under `:fresh`, or inside such an inference already, `(body opts)`."
  [opts body]
  (if (or (not= :fork (:world-policy opts :fresh)) (frozen-source opts))
    (body opts)
    (spin-core/with-causal-descendant-egress
      (spin
       (let [caller rtc/*execution-context*
             scope (world-scope/create (fork-scope-options opts :inference))
             lease (world-scope/begin-activity! scope :inference)
             source (volatile! nil)]
         (try
           (vreset! source (:child-ctx (await (fn [resolve reject]
                                                (let [[resolve reject] (sp/in-callers-world resolve reject)]
                                                  (world-scope/fork! scope caller {:grant (:grant opts)}
                                                                     resolve reject))))))
           ;; The runs fork the source; a plain fork does not check the
           ;; grades of the world's systems, a copy does (JVM). So a probe,
           ;; a fork of the source copied once, in a scope of its own
           ;; without authority (it spends nothing) and discarded at once,
           ;; refuses a world holding a system that may not be copied
           ;; before any model runs, as for SMC's particles.
           #?(:clj (let [probes (world-scope/create (dissoc (fork-scope-options opts :inference) :authority))
                         in-caller (fn [operation]
                                     (fn [resolve reject]
                                       (let [[resolve reject] (sp/in-callers-world resolve reject)]
                                         (operation resolve reject))))]
                     (try
                       (let [probe (:child-ctx (await (in-caller #(world-scope/fork! probes @source nil %1 %2))))]
                         (await (in-caller #(world-scope/copy! probes probe 1 %1 %2))))
                       (finally
                         (await-finalization (in-caller (world-scope/discard-when-quiescent! probes)))))))
           (itrace/mark-canonical! @source)
           (await (body (assoc opts :org.replikativ.foerster.core/source @source)))
           (catch #?(:clj Throwable :cljs :default) e
             (throw (if (#{spin-core/spin-cancelled :org.replikativ.foerster.core/inference-failed} (:type (ex-data e)))
                      e
                      (ex-info "Inference failed during particle execution"
                               {:type :org.replikativ.foerster.core/inference-failed
                                :world/recovery (canonical-recovery scope lease nil)}
                               e))))
           (finally
             (await-finalization (close-canonical! scope lease nil)))))))))

(defn run-options
  "`opts` for a driver whose runs are `foerster.gfi` runs: inside
  `with-frozen-caller`, `:canonical-source` names the frozen fork each run
  forks (`gfi/run-policy*`); `:world-policy` is consumed."
  [opts]
  (cond-> (dissoc opts :world-policy :org.replikativ.foerster.core/source)
    (frozen-source opts) (assoc :canonical-source (frozen-source opts))))
