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

(defn- latent-site? [s]
  (and (= itrace/choose-site (:savepoint/site s))
       (not (:observed? (:savepoint/payload s)))))

(defn enumerate
  "Exact enumeration of `model` (a spin), see the namespace. Options:
  `:max-branches`, `:policy` (a `foerster.trace/policy` for the sites that
  are not enumerated: observations, factors, deterministic sites), `:executor`.
  Resolves an EmpiricalMeasure of `Sample`s, one per complete branch."
  [model & [{:keys [max-branches policy executor] :or {max-branches 100000}}]]
  (fn [resolve reject]
    (let [[resolve reject] (sp/in-callers-world resolve reject)
          policy (or policy (itrace/policy))
          root (if executor (ctx/create-execution-context :executor executor) (ctx/create-execution-context))
          session (sp/open! root {:purpose :enumerate
                                  :seed (random/fresh-seed)
                                  :fork-opts {:systems :none}
                                  :retain-released? false})
          ;; branches alive (forked, not ended), branches made, finished outcomes
          state (atom {:alive 1 :made 1 :done [] :finished? false})
          finish! (fn [callback value]
                    (when-not (:finished? (first (swap-vals! state assoc :finished? true)))
                      ((sp/close! session) (fn [_] (callback value)) (fn [_] (callback value)))))
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
          branch! (fn [s]
                    (let [{:keys [dist]} (:savepoint/payload s)
                          values (dist/support dist)
                          address (:savepoint/address s)]
                      (cond
                        (nil? values)
                        (fail! (ex-info "Enumeration needs finite support at every latent site"
                                        {:type ::infinite-support :address address :dist dist}))

                        (> (+ (:made @state) (dec (count values))) max-branches)
                        (fail! (ex-info "Enumeration exceeds :max-branches"
                                        {:type ::too-many-branches :max-branches max-branches}))

                        :else
                        (do (swap! state #(-> % (update :alive + (dec (count values)))
                                              (update :made + (dec (count values)))))
                            ((forks s (dec (count values)))
                             (fn [children]
                               ;; each branch takes one value, its probability
                               ;; entering the weight as a constraint's does
                               (doseq [[child v] (map vector (cons s children) values)]
                                 (try
                                   (let [{:keys [value]} (decide! child (itrace/policy {:constraints {address v}}))]
                                     (continue! child value))
                                   (catch #?(:clj Throwable :cljs :default) e (fail! e)))))
                             fail!)))))
          run-site! (fn [s]
                      (try
                        (cond
                          (= smc/start-site (:savepoint/site s)) (sp/resume s nil)
                          (latent-site? s) (branch! s)
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
                (swap! state update :done conj
                       [(m/sample-particle result (itrace/legacy-trace (rtp/get-state world [:savepoint/trace]))) w]))
              (sp/release-world! session world)
              (ended!)))
          sp/error-site (fn [{error :savepoint/payload}] (fail! error))
          sp/abandoned-site (fn [_] nil)})
        (sp/start! session model)
        (catch #?(:clj Throwable :cljs :default) e (fail! e))))))
