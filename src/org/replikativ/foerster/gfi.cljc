(ns org.replikativ.foerster.gfi
  "Gen's generative function interface over savepoint traces.

  Each operation is `spindel.trace/run` or `replay` under a policy of
  `foerster.trace`; what this namespace adds is the weight each operation
  returns and the discard of `update`, with Gen's identities (Cusumano-Towner
  et al. 2019, q the model's internal proposal, here the prior):

    generate(c)       w = log p(t) − log q(t; c)     = Σ log p of constrained and observed sites
    assess(choices)   w = log p(choices)             = the log joint
    update(t, c)      w = log p(t') − log p(t) − log q(fresh choices of t')
    regenerate(t, s)  w = log p(t')/p(t) + log q(t | t')/q(t' | t)

  so importance sampling is `generate`, and MH with a selection is
  `regenerate` accepted on its weight. A model is a spin whose sample sites
  are named with `:id` (or addressed structurally, see `addressing/site-address!`);
  constraints are keyed by those addresses.

  Every operation returns a CPS operation `(fn [resolve reject])`.
  `simulate`, `generate` and `assess` run in a root world and savepoint session
  of their own (a session runs one computation, and a world hosts one
  session); the worlds live until `close!`.
  `update` and `regenerate` replay inside the session of the trace they are
  given and leave that trace intact: the caller keeps one of the two and gives
  the other back with `spindel.trace/release!`."
  (:refer-clojure :exclude [update await])
  (:require [org.replikativ.foerster.scope :as scope]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.world.scope :as world-scope]
            [org.replikativ.spindel.select :as sel]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]))

(defn- then [operation f]
  (fn [resolve reject]
    (operation (fn [x] (try (resolve (f x))
                            (catch #?(:clj Throwable :cljs :default) e (reject e))))
               reject)))

(defn- deliver-run
  "Run `model` under `policy` in `session`; resolve its trace, or close the
  session and reject when the program throws."
  [session model policy anchor? resolve reject]
  (let [fail (fn [error]
               ((sp/close! session) (fn [_] (reject error)) (fn [_] (reject error))))]
    ((trace/run session model policy {:anchor? anchor?})
     (fn [t]
       ;; a program that threw is a failed run, not a trace
       (if-let [error (:trace/error t)]
         (fail (ex-info "The model failed" {:type ::model-failed} error))
         (resolve t)))
     fail)))

(defn- canonical-run*
  "`run*` in a fork of `source`, a frozen fork of the caller's world
  (`:world-policy :fork`): the run's root is forked from it into a world
  scope of its own (`:world-opts`, `:executor`), which its session joins and
  closing it discards. The model runs whole, from a start site its moves can
  re-execute it from (`itrace/canonical-anchor?`). The session exists once
  the fork resolves: `:session` is nil, and the run is released through
  `:inference-scope`, or by closing the trace's session."
  [model policy source {:keys [executor inference-scope world-opts] :as opts}]
  (let [scope (world-scope/create {:purpose :gfi :retain-released? false
                                   :fork-opts (cond-> (or world-opts {})
                                                executor (assoc :executor executor))})
        state (atom {:session nil :released? false})
        release (fn [_]
                  (fn [resolve reject]
                    (let [{:keys [session]} (swap! state assoc :released? true)]
                      (if session
                        ((sp/close! session) resolve reject)
                        ((world-scope/discard-when-quiescent! scope) resolve reject)))))
        session-opts (merge {:purpose :gfi :seed (random/fresh-seed)}
                            (dissoc opts :executor :inference-scope :world-opts :canonical-source
                                    :fork-opts :retain-released?)
                            {:scope scope})]
    (when inference-scope
      (scope/acquire! inference-scope (constantly state) release))
    {:session nil
     :operation
     (fn [resolve reject]
       (let [[resolve reject] (sp/in-callers-world resolve reject)
             give-up (fn [error]
                       ((world-scope/discard-when-quiescent! scope)
                        (fn [_] (reject error)) (fn [_] (reject error))))]
         (world-scope/fork!
          scope source nil
          (fn [{root :child-ctx}]
            (if (:released? @state)
              (give-up (ex-info "The run was released before it started" {:type ::released}))
              (try
                (rtp/swap-state! root [:inference :canonical?] (constantly true))
                (let [session (sp/open! root session-opts)
                      after (swap! state #(if (:released? %) % (assoc % :session session)))]
                  (if (:released? after)
                    ((sp/close! session)
                     (fn [_] (reject (ex-info "The run was released before it started" {:type ::released})))
                     reject)
                    (deliver-run session
                                 (binding [ec/*execution-context* root]
                                   (spin (sp/savepoint :org.replikativ.foerster.smc/start nil)
                                         (await model)))
                                 policy itrace/canonical-anchor? resolve reject)))
                (catch #?(:clj Throwable :cljs :default) error (give-up error)))))
          give-up)))}))

(declare run-fresh*)

(defn- run*
  "{:session s :operation op}: `model` under `policy` in a new root world and
  session `s`, run by the CPS operation `op`. Options: `:executor` for the
  root world, the rest are session options (`effects.savepoint/open!`). A run
  that fails — the program throws — closes its session and rejects
  (`::model-failed`, the program's error its cause). With
  `:canonical-source`, a frozen fork of the caller's world, the run forks
  it instead (`canonical-run*`)."
  [model policy {:keys [canonical-source] :as opts}]
  (if canonical-source
    (canonical-run* model policy canonical-source opts)
    (run-fresh* model policy opts)))

(defn- run-fresh*
  [model policy {:keys [executor inference-scope] :as opts}]
  (let [root (if executor
               (ctx/create-execution-context :executor executor)
               (ctx/create-execution-context))
        allocate #(sp/open! root (merge {:purpose :gfi
                                         :seed (random/fresh-seed)
                                         :fork-opts {:systems :none}
                                         :retain-released? false}
                                        (dissoc opts :executor :inference-scope)))
        session (if inference-scope
                  (scope/acquire! inference-scope allocate sp/close!)
                  (allocate))]
    {:session session
     :operation (fn [resolve reject]
                  (deliver-run session model policy itrace/anchor? resolve reject))}))

(defn- run [model policy opts]
  (:operation (run* model policy opts)))

(defn run-policy*
  "As `run-policy`, but returns {:session s :operation op}: the session is
  known before the run ends, for a caller that must close it whatever
  happens to others."
  ([model policy] (run-policy* model policy nil))
  ([model policy opts] (run* model policy opts)))

(defn run-policy
  "Run `model` under any `foerster.trace/policy` in a root world and session
  of its own (options as for `simulate`). Resolves the trace."
  ([model policy] (run-policy model policy nil))
  ([model policy opts] (run model policy opts)))

(defn close!
  "Give back every world of `trace`'s session: the trace and every trace
  replayed from it. Returns a CPS operation."
  [trace]
  (sp/close! (:trace/session trace)))

(defn- entry-map [trace]
  (into {} (map (juxt :address identity)) (itrace/entries trace)))

(defn- fresh-log-q
  "Σ log q of the latent choices `new` drew afresh: not shared with `old`
  (upstream of a replay the entries are the same objects), not kept, not
  constrained, not set to a given value (`:moved?`)."
  [old new]
  (let [old-by (entry-map old)]
    (transduce (comp (filter itrace/latent?)
                     (remove #(identical? (:note %) (:note (get old-by (:address %)))))
                     (remove (comp :kept? :note))
                     (remove (comp :moved? :note))
                     (map (comp :log-proposal :note)))
               + 0.0 (itrace/entries new))))

(defn- visited-addresses
  "The addresses of the sample sites `t` reached."
  [t]
  (into #{} (keep #(when (= itrace/choose-site (:site %)) (:address %))) (itrace/entries t)))

(defn- unvisited-error
  "An error for the `constraints` addresses `t` never reached, or nil. Gen
  requires every constrained address to be a choice of the new trace; a
  constraint the run does not reach is most often a misspelled address or a
  branch the caller did not expect."
  [operation t constraints]
  (let [visited (visited-addresses t)
        missing (vec (remove visited (keys constraints)))]
    (when (seq missing)
      (ex-info (str operation ": constraints on sample sites the run never reached "
                    "(pass :allow-unvisited? true to ignore them)")
               {:type ::unvisited-constraints :addresses missing}))))

(defn- check-visited
  "A CPS operation resolving `(f t)` for the trace `operation` resolves,
  unless `constraints` hold addresses it never reached: then the trace's
  session — every world of a fresh run — is closed and the operation rejects
  with `::unvisited-constraints`."
  [name operation constraints allow-unvisited? close-on-error f]
  (fn [resolve reject]
    (operation (fn [t]
                 ;; a run the program aborted reached only part of its
                 ;; sites: its own error is what to report, not theirs
                 (if-let [error (when-not (or allow-unvisited? (:trace/error t))
                                  (unvisited-error name t constraints))]
                   (close-on-error t (fn [] (reject error)))
                   (try (resolve (f t))
                        (catch #?(:clj Throwable :cljs :default) e (reject e)))))
               reject)))

(defn simulate
  "Run `model` (a spin) drawing every sample site from its prior. Resolves the
  trace. Options as for `run`: `:executor` and session options."
  ([model] (simulate model nil))
  ([model opts]
   (run model (itrace/policy) opts)))

(defn generate
  "Run `model` with the sample sites in `constraints` ({address value}) fixed.
  Resolves {:trace t :weight w}, w = Σ log p of the constrained and observed
  sites: an importance weight for the prior as proposal.

  Every constrained address must be reached by the run; otherwise the run's
  worlds are closed and the operation rejects with `::unvisited-constraints`.
  `:allow-unvisited? true` ignores such constraints instead. The other options
  are those of `simulate`."
  ([model constraints] (generate model constraints nil))
  ([model constraints {:keys [allow-unvisited?] :as opts}]
   (check-visited "generate"
                  (run model (itrace/policy {:constraints constraints}) (dissoc opts :allow-unvisited?))
                  constraints allow-unvisited?
                  (fn [t done] ((close! t) (fn [_] (done)) (fn [_] (done))))
                  (fn [t] {:trace t :weight (itrace/log-weight t)}))))

(defn assess
  "log p of `choices` ({address value}), which must fix every sample site the
  run reaches. Resolves {:weight w :result r}; the run's session is closed."
  ([model choices] (assess model choices nil))
  ([model choices opts]
   (let [op (run model (itrace/policy {:constraints choices}) opts)]
     (fn [resolve reject]
       (op (fn [t]
             (let [free (itrace/latent-addresses t)
                   out {:weight (itrace/log-joint t) :result (:trace/result t)}]
               ((close! t)
                (fn [_]
                  (if (seq free)
                    (reject (ex-info "assess needs a value for every sample site"
                                     {:type ::unconstrained-choices :addresses free}))
                    (resolve out)))
                reject)))
           reject)))))

(defn- assess-visiting
  "As `assess`, resolving also `:visited`, the addresses of the sample sites
  the run reached; `choices` may hold others, which it ignores."
  [model choices opts]
  (fn [resolve reject]
    ((run model (itrace/policy {:constraints choices}) opts)
     (fn [t]
       (let [free (itrace/latent-addresses t)
             out {:weight (itrace/log-joint t)
                  :visited (into #{} (keep #(when (= itrace/choose-site (:site %)) (:address %))) (itrace/entries t))}]
         ((close! t)
          (fn [_]
            (if (seq free)
              (reject (ex-info "The reverse proposal leaves a sample site unproposed"
                               {:type ::unconstrained-choices :addresses free}))
              (resolve out)))
          reject)))
     reject)))

(defn- replay-from
  "The earliest of `addresses` in `trace`, or else its first sample site that
  is not an observation, so that a replay can reach sites the trace never
  did."
  [trace addresses]
  (or (trace/earliest trace addresses)
      (some (fn [e] (when (and (= itrace/choose-site (:site e)) (not (:observed? (:note e))))
                      (:address e)))
            (itrace/entries trace))))

(defn update
  "Change the sample sites in `constraints` ({address value}) and run the
  computation again from the earliest of them; every other site keeps its
  value, rescored under its distribution as it is now, and a site the
  changed control flow reaches for the first time is drawn from its prior.

  Resolves {:trace t' :weight w :discard d} where
    w = log p(t') − log p(t) − Σ log q of what t' drew afresh
    d = {address value} of the choices of t that t' does not keep: the
        overwritten values of constrained sites, and the sites no longer
        reached.
  As in Gen, a kept value is kept even when the change moved it outside its
  site's support: t' then has density zero and w = −∞. `constraints` are new
  values, not new conditioning: a site that was latent stays latent with its
  new value (so a later move can change it again), and one that was
  constrained stays constrained. A replay the program aborts resolves with
  w = −∞ and the program's error under `:error`.

  A constraint on a site `trace` never reached replays from its first
  sample site, so the program can get there. A constraint the new run still
  does not reach is an error (`::unvisited-constraints`; the replayed worlds
  are given back) unless `:allow-unvisited? true`."
  ([trace constraints] (update trace constraints nil))
  ([trace constraints {:keys [allow-unvisited?]}]
   (let [from (replay-from trace (filter #(contains? (entry-map trace) %) (keys constraints)))]
     (if-not from
       (fn [_ reject]
         (reject (ex-info "update: the trace has no sample site to replay from"
                          {:type ::nothing-to-replay})))
       (check-visited "update"
                      (let [old-by (entry-map trace)
                            fixed? (fn [a] (let [n (:note (get old-by a))]
                                             (and (:constrained? n) (not (:intervened? n)))))
                            fixed (into {} (filter (comp fixed? key)) constraints)
                            moved (into {} (remove (comp fixed? key)) constraints)]
                        (itrace/reexecute trace (let [existing (filter #(contains? old-by %) (keys constraints))]
                                                  (if (seq existing) (set existing) #{from}))
                                          (itrace/policy {:keep? true :constraints fixed :out-of-support :keep
                                                          :draw (fn [sp _]
                                                                  (when-let [[_ v] (find moved (:savepoint/address sp))]
                                                                    {:value v :symmetric? true}))})
                                          {:anchor? itrace/anchor?}))
                      constraints allow-unvisited?
                      (fn [t' done] (trace/release! t' trace) (done))
                      (fn [t']
                        (let [error (:trace/error t')
                              new-by (entry-map t')
                              discard (into {}
                                            (keep (fn [{:keys [address value] :as e}]
                                                    (when (and (= itrace/choose-site (:site e))
                                                               (not (:observed? (:note e)))
                                                               (or (not (contains? new-by address))
                                                                   (contains? constraints address)))
                                                      [address value])))
                                            (itrace/entries trace))]
                          (cond-> {:trace t'
                                   :weight (if error
                                             ##-Inf
                                             (- (itrace/log-joint t') (itrace/log-joint trace) (fresh-log-q trace t')))
                                   :discard discard}
                            error (assoc :error error)))))))))

(defn- selected [trace selection]
  (let [entries (:trace/entries trace)]
    (filterv (if (set? selection)
               selection
               #(sel/selects? selection (sel/describe % (get entries %))))
             (itrace/latent-addresses trace))))

(defn regenerate
  "Draw the selected sample sites afresh from their priors and run the
  computation again from the earliest of them, keeping every other site.
  `selection` is a set of addresses or a selector (`spindel.select`).

  Resolves {:trace t' :weight w}, w = the Metropolis-Hastings log ratio of
  the move without a site-selection term: accepting on it is MH with this
  (fixed) selection. Nothing selected resolves {:trace trace :weight 0.0}."
  [trace selection]
  (let [targets (set (selected trace selection))
        from (trace/earliest trace targets)]
    (if-not from
      (fn [resolve _] (resolve {:trace trace :weight 0.0}))
      (then (itrace/reexecute trace targets
                              (itrace/policy {:keep? true
                                              :draw (fn [sp old-entry]
                                                      (when (contains? targets (:savepoint/address sp))
                                                        (itrace/prior-proposal sp old-entry)))})
                              {:anchor? itrace/anchor?})
            (fn [t'] {:trace t' :weight (itrace/mh-log-ratio trace t' (constantly 0.0))})))))

(defn mh
  "One Metropolis-Hastings move: `regenerate` the selection and accept on its
  weight. The loser's worlds are given back. Resolves {:trace t :accepted? b}."
  [trace selection]
  (then (regenerate trace selection)
        (fn [{t' :trace w :weight}]
          (let [accept? (and (not (identical? t' trace))
                             (not (#?(:clj Double/isNaN :cljs js/isNaN) w))
                             (or (>= w 0.0)
                                 (< (Math/log (random/in-world-stream (:trace/world t') ::accept
                                                                      m/uniform01))
                                    w)))]
            (cond
              (identical? t' trace) nil
              accept? (trace/release! trace t')
              :else (trace/release! t' trace))
            {:trace (if accept? t' trace) :accepted? accept?}))))

(defn mh-proposal
  "One Metropolis-Hastings move with a custom proposal: `proposal` is a
  function `(fn [choices & args])` of the current choices ({address value})
  returning a spin whose sample sites are named by the MODEL addresses they
  propose — any number of them jointly, correlated as the proposal program
  likes (Gen's `metropolis_hastings(trace, proposal, args)`).

  The proposal runs forward on the current choices (its log density
  log q(c' | t)), the model is updated with what it proposed (`update`), and
  the proposal is assessed on the reverse move, the values it replaced
  given the new choices (log q(c | t')). Accepted on
    log α = w(update) + log q(c | t') − log q(c' | t).
  The proposal must be able to propose the reverse move (it reaches the same
  addresses from t'). The loser's worlds are given back. Resolves
  {:trace t :accepted? b}. The proposal's own runs are seeded from the
  trace's world, so a chain seeded by `:seed` reproduces its moves."
  [trace proposal & args]
  (fn [resolve reject]
    (let [world (:trace/world trace)
          move (rtp/swap-state! world [:inference ::proposals] (fnil inc 0))
          seed-of (fn [role] (sp/derive-seed (sp/seed world) ::proposal [move role]))]
    ((simulate (apply proposal (itrace/choices trace) args) {:seed (seed-of :forward)})
     (fn [fwd]
       (let [proposed (itrace/choices fwd)
             fwd-score (itrace/log-joint fwd)]
         ((close! fwd)
          (fn [_]
            ((update trace proposed)
             (fn [{t' :trace w :weight discard :discard}]
               ((assess-visiting (apply proposal (itrace/choices t') args) discard {:seed (seed-of :reverse)})
                (fn [{bwd-score :weight visited :visited}]
                  ;; a discarded site the reverse proposal does not propose
                  ;; would be drawn afresh by the reverse update, as a replay
                  ;; draws it (its prior, its site proposal or a block's
                  ;; :sample): that density belongs to the reverse move
                  (let [old-by (entry-map trace)
                        regenerated (reduce + 0.0 (for [[a v] discard
                                                        :when (not (contains? visited a))]
                                                    (itrace/redraw-log-density (get-in old-by [a :note]) v)))
                        log-alpha (+ w (- (+ bwd-score regenerated) fwd-score))
                        accept? (and (not (#?(:clj Double/isNaN :cljs js/isNaN) log-alpha))
                                     (or (>= log-alpha 0.0)
                                         (< (Math/log (random/in-world-stream (:trace/world t') ::accept
                                                                              m/uniform01))
                                            log-alpha)))]
                    (if accept? (trace/release! trace t') (trace/release! t' trace))
                    (resolve {:trace (if accept? t' trace) :accepted? accept?})))
                (fn [e] (trace/release! t' trace) (reject e))))
             reject))
          reject)))
     reject))))
