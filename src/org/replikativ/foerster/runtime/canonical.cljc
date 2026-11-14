(ns org.replikativ.foerster.runtime.canonical
  "Canonical worlds for inference that runs several times (`:world-policy
  :fork`): one frozen fork of the caller's world per inference, from which
  every run (a particle-MCMC sweep, an IPMCMC node, a BBVI iteration, a
  Markov chain, a tempered-SMC particle, an inference program) forks a world
  of its own. See doc/worlds.md.

  Engine v2: the frozen fork is a host fork of the caller's world (`v2/fork!`),
  released when the inference ends. Resource authority and grants (wallets,
  budget splitting) are not available on engine v2 yet: `:authority` and
  `:grant` are refused."
  (:require [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]
            [org.replikativ.foerster.runtime.async :refer [await-op]])
  #?(:cljs (:require-macros [org.replikativ.spindel.v2.core :refer [task]]
                            [org.replikativ.foerster.runtime.async :refer [await-op]])))

(defn- no-authority! [opts]
  (when (or (:authority opts) (:grant opts))
    (throw (ex-info ":authority and :grant are not available on engine v2 yet"
                    {:type ::unsupported-authority}))))

(defn share
  "One of `k` even shares of the wallet `resources`: integer amounts rounded
  down."
  [resources k]
  (when resources
    (into {} (map (fn [[r amount]] [r (if (integer? amount) (quot amount k) (/ amount k))]))
          resources)))

(defn balance-op
  "CPS: what is left in the wallet under `authority`; nil without one (the
  only case on engine v2)."
  [authority _world]
  (fn [resolve reject]
    (if authority
      (reject (ex-info ":authority is not available on engine v2 yet" {:type ::unsupported-authority}))
      (resolve nil))))

(defn frozen-source
  "The frozen fork of the caller's world a multi-run inference runs every
  run from (`with-frozen-caller`), or nil."
  [opts]
  (:org.replikativ.foerster.core/source opts))

(defn with-frozen-caller
  "Under `:world-policy :fork`, `(body opts')` with `opts'` naming one frozen
  fork of the caller's world, made once for the whole inference: every run
  forks its own world from it, so all of them see the caller's state as it
  was when inference started, whatever the caller does meanwhile. The fork
  is released after `body` ends, however it ends. `body` returns a v2 handle
  or a CPS operation; this returns a task. Under `:fresh`, or inside such an
  inference already, `(body opts)`. The fork is taken when this is called,
  which must be host code (engine v2 forks a world between slices)."
  [opts body]
  (no-authority! opts)
  (if (or (not= :fork (:world-policy opts :fresh)) (frozen-source opts))
    (body opts)
    (let [source (v2/fork! (v2/current-world))]
      (itrace/mark-canonical! source)
      (task
       (try
         (await-op (body (assoc opts :org.replikativ.foerster.core/source source)))
         (finally (v2/release-world! source)))))))

(defn run-options
  "`opts` for a driver whose runs are `foerster.runtime.gfi` runs: inside
  `with-frozen-caller`, `:canonical-source` names the frozen fork each run
  forks (`gfi/run-policy*`); `:world-policy` is consumed."
  [opts]
  (cond-> (dissoc opts :world-policy :org.replikativ.foerster.core/source)
    (frozen-source opts) (assoc :canonical-source (frozen-source opts))))
