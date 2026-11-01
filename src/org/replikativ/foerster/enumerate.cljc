(ns org.replikativ.foerster.enumerate
  "Exact inference by enumeration, for programs whose latent choices all have
  finite support (`dist/support`): at every latent sample site the particle's
  world forks once per value, each branch weighted by that value's
  probability, and observations and factors score as usual. Every branch that
  ends is one outcome with its exact joint probability, so the measure is the
  exact posterior and its `m/log-marginal` the exact evidence.

  Branches share the program up to the site where they part — the world fork
  is a copy-on-write of everything computed so far — so the cost is the
  number of branches, not that number times the program's length. That
  number grows exponentially with the latent sites: enumeration is an oracle
  for small discrete models, and the reference other methods are checked
  against.

    (infer/infer (model) {:method :enumerate})

  A latent site with infinite or continuous support is refused
  (`::infinite-support`); `:max-branches` (default 100000) bounds the work."
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.trace :as trace]))

(defn- forks
  "Fork `s` `k` times; resolves the children."
  [s k]
  (fn [resolve reject]
    (if (zero? k)
      (resolve [])
      (let [out (object-array k)
            remaining (atom k)
            failed? (atom false)]
        (dotimes [i k]
          ((sp/fork s)
           (fn [child]
             (aset out i child)
             (when (zero? (swap! remaining dec)) (resolve (vec out))))
           (fn [e] (when (compare-and-set! failed? false true) (reject e)))))))))

(defn- branches
  "`k` savepoints continuing `s`, one per value: `s` itself and k−1 forks,
  or with `copy?` k copies of `s` that consume it (`effects.savepoint/copy`,
  each with an even share of its world's budget), as SMC makes particles in
  canonical worlds."
  [s k copy?]
  (if copy?
    (sp/copy s k {:split? true})
    (fn [resolve reject]
      ((forks s (dec k)) #(resolve (into [s] %)) reject))))

(defn- latent-site? [s]
  (and (= itrace/choose-site (:savepoint/site s))
       (not (:observed? (:savepoint/payload s)))))

(defn enumerate
  "Exact enumeration of `model` (a spin), see the namespace. Options:

    :max-branches  bound on the branches made (default 100000)
    :policy        a `foerster.trace/policy`, the caller's: it decides every
                   site, so its constraints, interventions and `:temperature`
                   apply as under importance sampling. A latent site it fixes
                   (a constraint, a `{:do v}` or `{:policy f}` intervention,
                   the world's `intervene!`) is not branched on; any other
                   latent site branches over the support of the law the
                   policy draws it from (`trace/site-law`), each branch
                   decided by the policy with the branch's value added to its
                   constraints
    :executor      the root world's executor
    :seed          the session's seed (the worlds' random streams)
    :root          a world to run in instead of a fresh one; with `:scope`
                   the world scope the session joins (the root's), and
                   `:copy? true` to make branches by copying worlds — the
                   canonical worlds of `:world-policy :fork`
                   (`foerster.core`), as `smc/smc` takes them

  Resolves an EmpiricalMeasure of `Sample`s, one per complete branch."
  [model & [{:keys [max-branches policy executor seed root scope copy?] :or {max-branches 100000}}]]
  (fn [resolve reject]
    (let [[resolve reject] (sp/in-callers-world resolve reject)
          policy (or policy (itrace/policy))
          options (if (contains? (meta policy) :org.replikativ.foerster.trace/options)
                    (itrace/policy-options policy)
                    (throw (ex-info "Enumeration needs a policy made by foerster.trace/policy"
                                    {:type ::opaque-policy})))
          _ (when (:simulate-observed? options)
              ;; a predictive policy draws the observed sites at random:
              ;; the result would be a sample, not an exact sum
              (throw (ex-info "Enumeration sums the posterior: not with a predictive policy (:simulate-observed?)"
                              {:type ::predictive-policy})))
          root (cond root root
                     executor (ctx/create-execution-context :executor executor)
                     :else (ctx/create-execution-context))
          session (sp/open! root (cond-> {:purpose :enumerate
                                          :seed (or seed (random/fresh-seed))
                                          :fork-opts {:systems :none}
                                          :retain-released? false}
                                   scope (assoc :scope scope)))
          ;; branches alive (forked, not ended), branches made, finished outcomes
          state (atom {:alive 1 :made 1 :done [] :finished? false})
          ;; in the caller's scope (canonical worlds) the outcome comes first:
          ;; the caller discards the scope once it has it (as `smc/smc`)
          finish! (fn [callback value]
                    (when-not (:finished? (first (swap-vals! state assoc :finished? true)))
                      (if scope
                        (do (callback value) ((sp/close! session) (fn [_]) (fn [_])))
                        ((sp/close! session) (fn [_] (callback value)) (fn [_] (callback value))))))
          fail! #(finish! reject %)
          ended! (fn []
                   (let [{:keys [alive done]} (swap! state update :alive dec)]
                     (when (zero? alive)
                       (if (empty? done)
                         (fail! (ex-info "Every branch has probability zero" {:type ::no-outcome}))
                         ;; the evidence is the SUM of the branch weights:
                         ;; log-marginal adds the log mean, so add log n back
                         (finish! resolve (assoc (m/empirical done)
                                                 :log-normalizer (Math/log (count done))
                                                 :enumeration {:branches (count done)}))))))
          ;; a branch whose weight is zero has nothing left to contribute:
          ;; it is dropped before it runs on (where it might build a law that
          ;; is only invalid where it is impossible)
          continue! (fn [s value]
                      (if (= ##-Inf (or (rtp/get-state (:savepoint/world s) [:inference :log-weight]) 0.0))
                        (do (sp/abandon s)
                            (sp/release-world! session (:savepoint/world s))
                            (ended!))
                        (sp/resume s value)))
          decide! (fn [s p]
                    (let [decision (p s nil)]
                      (trace/record! (:savepoint/world s) s decision nil)
                      decision))
          ;; the caller's policy with one more constraint: the branch's value
          branch-policy (fn [address v]
                          (itrace/policy (update options :constraints assoc address v)))
          branch! (fn [s values]
                    (let [address (:savepoint/address s)
                          k (count values)]
                      (if (> (+ (:made @state) (dec k)) max-branches)
                        (fail! (ex-info "Enumeration exceeds :max-branches"
                                        {:type ::too-many-branches :max-branches max-branches}))
                        (do (swap! state #(-> % (update :alive + (dec k))
                                              (update :made + (dec k))))
                            ((branches s k copy?)
                             (fn [children]
                               ;; each branch takes one value, its probability
                               ;; entering the weight as a constraint's does
                               (doseq [[child v] (map vector children values)]
                                 (try
                                   (let [{:keys [value]} (decide! child (branch-policy address v))]
                                     (continue! child value))
                                   (catch #?(:clj Throwable :cljs :default) e (fail! e)))))
                             fail!)))))
          latent! (fn [s]
                    (let [{:keys [fixed?] law :dist} (itrace/site-law policy s)]
                      (if fixed?
                        ;; the caller's policy sets the value: no branching
                        (let [{:keys [value]} (decide! s policy)] (continue! s value))
                        (if-let [values (seq (dist/support law))]
                          (branch! s (vec values))
                          (fail! (ex-info "Enumeration needs finite support at every latent site"
                                          {:type ::infinite-support :address (:savepoint/address s)
                                           :dist law}))))))
          run-site! (fn [s]
                      (try
                        (cond
                          ;; in copied worlds the model runs in a copy of
                          ;; the root, never in the root itself, as SMC's
                          ;; particles do: a world that may not be copied is
                          ;; refused before the model starts
                          (= smc/start-site (:savepoint/site s))
                          (if copy?
                            ((branches s 1 true) (fn [[child]] (sp/resume child nil)) fail!)
                            (sp/resume s nil))
                          (latent-site? s) (latent! s)
                          :else (let [{:keys [value]} (decide! s policy)] (continue! s value)))
                        (catch #?(:clj Throwable :cljs :default) e (fail! e))))]
      (try
        (sp/install-handlers!
         root
         {sp/any-site run-site!
          sp/result-site
          (fn [{world :savepoint/world result :savepoint/payload}]
            (let [w (or (rtp/get-state world [:inference :log-weight]) 0.0)]
              (when (> w ##-Inf)
                (swap! state update :done conj [(smc/sample-of world result) w]))
              (sp/release-world! session world)
              (ended!)))
          sp/error-site (fn [{error :savepoint/payload}] (fail! error))
          sp/abandoned-site (fn [_] nil)})
        (sp/start! session model)
        (catch #?(:clj Throwable :cljs :default) e (fail! e))))))
