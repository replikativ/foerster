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
  (:require [is.simm.partial-cps.async :as pcps]
            [org.replikativ.foerster.runtime.lifetime :as lt]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]
            [org.replikativ.foerster.runtime.async :as async :refer [await-op]])
  #?(:cljs (:require-macros [org.replikativ.spindel.v2.core :refer [task]]
                            [org.replikativ.foerster.runtime.async :as async :refer [await-op]])))

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
  "The entry of every public inference method: `(body opts')`, run when the
  operation starts, with `opts'` naming the inference's lifetime
  (`foerster.runtime.lifetime`): cancelling the task awaiting the
  operation releases every root world the inference created, and the
  lifetime is finished when `body` ends, however it ends.

  Under `:world-policy :fork`, `opts'` also names one frozen fork of the
  caller's world, made once for the whole inference: every run forks its
  own world from it, so all of them see the caller's state as it was when
  inference started, whatever the caller does meanwhile. The fork is
  released after `body` ends. The caller's world is the current one when
  this is called, possibly inside a body; it is forked when the operation
  starts, as host code (engine v2 forks a world between slices).

  Inside an inference already (`opts` name a lifetime), `(body opts)`:
  the methods a driver runs share its lifetime and frozen fork. `body`
  returns a v2 handle or a CPS operation; this returns a CPS operation."
  [opts body]
  (no-authority! opts)
  (if (lt/of opts)
    (body opts)
    (let [caller (v2/current-world)
          fork? (and (= :fork (:world-policy opts :fresh)) (not (frozen-source opts)))]
      (pcps/async
       (let [l (lt/create)
             _ (async/on-cancel! #(lt/cancel! l))
             opts (lt/with opts l)
             source (when fork? (lt/fork opts caller))]
         (when source (itrace/mark-canonical! source))
         (try
           (pcps/await (async/op-of (body (cond-> opts
                                            source (assoc :org.replikativ.foerster.core/source source)))))
           (finally
             (lt/finish! l)
             (when source (v2/release-world! source)))))))))

(defn run-options
  "`opts` for a driver whose runs are `foerster.runtime.gfi` runs: inside
  `with-frozen-caller`, `:canonical-source` names the frozen fork each run
  forks (`gfi/run-policy*`); `:world-policy` is consumed."
  [opts]
  (cond-> (dissoc opts :world-policy :org.replikativ.foerster.core/source)
    (frozen-source opts) (assoc :canonical-source (frozen-source opts))))
