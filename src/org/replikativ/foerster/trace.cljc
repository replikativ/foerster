(ns org.replikativ.foerster.trace
  "Probabilistic programs as traced computations.

  `sample`, `observe` and `factor` publish savepoints (sites
  `:inference/choose` and `:inference/factor`), so a probabilistic program is
  run and replayed by `spindel.trace` like any other computation. This
  namespace adds what is specific to inference and nothing else: a policy that
  scores, pure functions over scored traces, and Metropolis-Hastings as replay
  plus an accept step.

  Every entry's `:note` carries

    :dist          the site's distribution (nil for a factor)
    :log-prob      log density of the entry's value under :dist, for EVERY
                   site, sampled ones included (a factor's weight)
    :log-proposal  log density under whatever drew the value; absent when the
                   value was not drawn (observed, constrained, kept)
    :observed? :constrained? :kept? :symmetric? :factor?

  A `deterministic` site is recorded with `{:deterministic? true}` only; it is
  not among `entries`, so it neither scores nor moves.

  and the world accumulates the importance weight at `[:inference
  :log-weight]`: `log p` of what was observed, constrained or factored, and
  `log p - log q` of what a proposal drew. A policy writes it to the world it
  is about to resume, so the weight of a fork starts from the weight at its
  site."
  (:require [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.select :as sel]
            [org.replikativ.foerster.mechanism :as mech]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.dist :as dist]))

(def choose-site :inference/choose)
(def factor-site :inference/factor)
(def deterministic-site :inference/deterministic)

;; =============================================================================
;; The scoring policy
;; =============================================================================

(defn- add-weight! [world w]
  (rtp/swap-state! world [:inference :log-weight] (fn [x] (+ (or x 0.0) w))))

(defn- finite? [x]
  (and (number? x)
       #?(:clj (not (Double/isInfinite (double x))) :cljs (js/isFinite x))
       #?(:clj (not (Double/isNaN (double x))) :cljs true)))

(defn- shifted
  "`dist` moved by `delta`: x = x₀ + δ, x₀ ~ dist."
  [dist delta]
  (reify dist/Distribution
    (-draw [_] (+ (dist/draw dist) delta))
    (-logpdf [_ v] (dist/logpdf dist (- v delta)))))

(defn- intervention-pairs
  "`interventions` as [selector transform] pairs. A key that is not a
  selector is an address; a value that is not a transform map is `{:do v}`
  (the form `foerster.effects/intervene!` writes)."
  [interventions]
  (for [[k v] interventions]
    [(if (fn? k) k (sel/id k))
     (if (and (map? v) (some #{:do :dist :shift :policy} (keys v))) v {:do v})]))

(defn- intervention-for [opts sp]
  (let [world (:savepoint/world sp)
        d (sel/describe sp)]
    (some (fn [[selects? transform]] (when (selects? d) transform))
          (concat (intervention-pairs (:interventions opts))
                  (intervention-pairs (rtp/get-state world [:inference :interventions]))))))

(defn- choices-so-far
  "{address value} of the sample sites this computation decided so far."
  [world]
  (into {} (keep (fn [[address entry]]
                   (when (= :inference/choose (:site entry)) [address (:value entry)])))
        (:trace/entries (rtp/get-state world [:savepoint/trace]))))

(declare decide-choose)

(defn- decide-intervened
  "A site under an intervention. `{:do v}` and `{:policy f}` fix the value
  (f sees the choices made so far) and score nothing: the site is no longer
  random. `{:dist d}` and `{:shift δ}` replace the site's mechanism and decide
  it as a site of the new distribution."
  [opts sp old-entry {:keys [policy dist shift] :as transform}]
  (let [payload (:savepoint/payload sp)
        world (:savepoint/world sp)]
    (cond
      (contains? transform :do)
      {:value (:do transform)
       :note {:dist (:dist payload) :log-prob 0.0 :intervened? true :constrained? true}}

      policy
      {:value (policy (choices-so-far world))
       :note {:dist (:dist payload) :log-prob 0.0 :intervened? true :constrained? true}}

      :else
      (let [new-dist (or dist (shifted (:dist payload) shift))
            decision (decide-choose (assoc opts ::no-interventions true)
                                    (assoc sp :savepoint/payload (assoc payload :dist new-dist))
                                    old-entry)]
        (assoc-in decision [:note :intervened?] true)))))

(defn barrier-site?
  "Whether a choose savepoint is where SMC parks a particle: an observation,
  or a stream site."
  [sp]
  (let [payload (:savepoint/payload sp)]
    (boolean (or (:observed? payload) (:stream (:options payload))))))

(defn- count-barrier!
  "The barrier index of a choose site in its world: how many barrier sites
  (`barrier-site?`) its run has decided, this one included. Kept in the
  world, so a fork or a replay continues the count from where it starts."
  [sp]
  (let [world (:savepoint/world sp)]
    (if (barrier-site? sp)
      (rtp/swap-state! world [:inference :barriers] (fnil inc 0))
      (or (rtp/get-state world [:inference :barriers]) 0))))

(defn policy-options
  "The options a `policy` was made from, or nil for another function."
  [policy]
  (::options (meta policy)))

(defn- tempered
  "`log-lik` under the policy's `:temperature` β: β·log-lik, and 0 at β = 0
  (the prior, whatever the likelihood)."
  [{:keys [temperature]} log-lik]
  (cond (nil? temperature) log-lik
        (zero? temperature) 0.0
        :else (* temperature log-lik)))

(defn tempered-score
  "The log target of a trace entry at temperature `beta`, for entries recorded
  under any temperature: observations and factors by their `:log-lik`."
  [beta]
  (fn [{:keys [note]}]
    (if-let [l (:log-lik note)]
      (cond (zero? beta) 0.0 :else (* beta l))
      (:log-prob note))))

(defn- decide-choose
  [{:keys [constraints keep? draw init?] :as opts} sp old-entry]
  (let [{:keys [dist observed? value]} (:savepoint/payload sp)
        world (:savepoint/world sp)
        address (:savepoint/address sp)
        intervention (when-not (::no-interventions opts) (intervention-for opts sp))]
    (cond
      ;; Pearl's do-operator and its soft/shift/policy variants
      intervention
      (decide-intervened opts sp old-entry intervention)

      ;; a counterfactual world: the site's mechanism under its (possibly
      ;; new) parents, fed the noise abducted from the factual world. This
      ;; comes before `observed?`: an observed site is predicted, not held
      ;; at its data.
      (and (contains? (:noise opts) address) (mech/mechanism? dist))
      (let [v (mech/push dist (get (:noise opts) address))]
        {:value v :note {:dist dist :log-prob (dist/logpdf dist v) :counterfactual? true}})

      observed?
      (let [lp (dist/logpdf dist value)
            tp (tempered opts lp)]
        (add-weight! world tp)
        {:value value :note (cond-> {:dist dist :log-prob tp :observed? true}
                              (:temperature opts) (assoc :log-lik lp))})

      (contains? constraints address)
      (let [v (get constraints address)
            lp (dist/logpdf dist v)]
        (add-weight! world lp)
        {:value v :note {:dist dist :log-prob lp :constrained? true}})

      :else
      (let [drawn (when draw (draw sp old-entry))
            kept-lp (when (and (not drawn) keep? old-entry
                               (not (:observed? (:note old-entry))))
                      (dist/logpdf dist (:value old-entry)))]
        (cond
          drawn
          (let [_ (when-not (or (:symmetric? drawn) (number? (:log-proposal drawn)))
                    (throw (ex-info "A proposal returns :log-proposal or :symmetric? true"
                                    {:type ::malformed-proposal
                                     :address address
                                     :proposal (dissoc drawn :value)})))
                lp (dist/logpdf dist (:value drawn))]
            (when-not (:symmetric? drawn)
              (add-weight! world (- lp (:log-proposal drawn))))
            {:value (:value drawn)
             :note (cond-> {:dist dist :log-prob lp
                            :log-proposal (:log-proposal drawn)}
                     (:symmetric? drawn) (assoc :symmetric? true))})

          ;; A kept value outside the new distribution's support is drawn
          ;; again instead (the site's parents moved it out from under it).
          (finite? kept-lp)
          {:value (:value old-entry)
           :note {:dist dist :log-prob kept-lp :kept? true}}

          :else
          (let [init (when init? (:init (:options (:savepoint/payload sp))))
                v (if (some? init) init (dist/draw dist))
                lp (dist/logpdf dist v)
                ;; a draw from something other than the site's law (a
                ;; block's :sample) is weighed by the difference
                lq (if (some? init) lp (or (dist/-draw-logpdf dist v) lp))]
            (when (not= lp lq)
              (add-weight! world (- lp lq)))
            {:value v
             :note (cond-> {:dist dist :log-prob lp :log-proposal lq}
                     ;; The old value was there and could not be kept. The
                     ;; reverse move must be able to do the same; see
                     ;; `mh-log-ratio`.
                     (and keep? old-entry) (assoc :redrawn? true))}))))))

(defn policy
  "The scoring policy.

  Options:
    :constraints {address value} fix those sample sites; their density enters
                 the weight
    :keep?       reuse the value a sample site had in the replayed trace,
                 rescored under the site's distribution NOW
    :draw        (fn [sp old-entry]) -> nil (not my site) or
                 {:value v :log-proposal lq}, or {:value v :symmetric? true}
                 for a symmetric move around the old value
    :interventions {selector transform}: a selected sample site takes
                 `{:do v}` (that value, no score), `{:policy (fn [choices])}`
                 (a value from the choices made so far), `{:dist d}` (a new
                 mechanism) or `{:shift δ}` (the old one moved by δ). A key
                 that is not a selector (`spindel.select`) is an address; the
                 world's `[:inference :interventions]` (`intervene!`) apply too
    :noise       {address u}: the site takes its mechanism's value for u
                 (`foerster.mechanism`), observed sites included — a
                 counterfactual world. Sites without noise are marked
                 `:unaligned?`
    :temperature β in [0, 1]: observations and factors count β·log p — the
                 target p(x)·L(x)^β of tempered SMC (`foerster.tempering`).
                 Their notes keep `:log-lik`, the untempered log p
    :init?       start a sample site at its `:init` option. Only for the
                 first state of a Markov chain, which may be anything; an
                 `:init` value is not a draw, so it has no place in a move or
                 in a weighted sample.
    :else        policy for every other site (default: its payload)

  With no options every sample site is drawn from its prior: forward
  simulation, likelihood weighting.

  Every choose entry's note records `:barrier`, its barrier index (see
  `barrier-site?`); `policy-options` recovers the options."
  ([] (policy nil))
  ([{fallback :else :as opts}]
   (let [fallback (or fallback trace/payload-policy)]
     (with-meta
       (fn [sp old-entry]
         (let [site (:savepoint/site sp)]
           (cond
             (= choose-site site)
             (cond-> (random/in-world-stream (:savepoint/world sp) (:savepoint/address sp)
                                             #(decide-choose opts sp old-entry))
               true (assoc-in [:note :barrier] (count-barrier! sp))
             ;; with :noise, a site that got no factual noise (it did not
             ;; exist in the factual world, or its law is not a mechanism)
             ;; is reported
               (and (:noise opts) (not (contains? (:noise opts) (:savepoint/address sp))))
               (update :note (fn [n] (if (:intervened? n) n (assoc n :unaligned? true)))))

             (= deterministic-site site)
             {:value (:value (:savepoint/payload sp)) :note {:deterministic? true}}

             (= factor-site site)
             (let [w (:log-weight (:savepoint/payload sp))
                   tw (tempered opts w)]
               (add-weight! (:savepoint/world sp) tw)
               {:value nil :note (cond-> {:log-prob tw :factor? true}
                                   (:temperature opts) (assoc :log-lik w))})

             :else (fallback sp old-entry))))
       {::options opts}))))

;; =============================================================================
;; Scored traces
;; =============================================================================

(defn entries
  "The inference entries of `trace` in program order, each with its :address."
  [trace]
  (into []
        (comp (map (fn [address]
                     (assoc (get-in trace [:trace/entries address]) :address address)))
              (filter #(#{choose-site factor-site} (:site %))))
        (:trace/order trace)))

(defn anchor?
  "Whether a site is worth an anchor: only a site a move can start from. An
  observe or a factor is never replayed from, and an anchor is a forked
  world. Pass as `:anchor?` to `trace/run` and `trace/replay`."
  [sp]
  (and (= choose-site (:savepoint/site sp))
       (not (:observed? (:savepoint/payload sp)))))

(defn latent?
  "Whether a trace entry is a sample site inference may move: not observed,
  not constrained."
  [entry]
  (and (= choose-site (:site entry))
       (not (:observed? (:note entry)))
       (not (:constrained? (:note entry)))))

(defn latent-addresses
  "Addresses of the sample sites of `trace` that inference may move."
  [trace]
  (into [] (comp (filter latent?) (map :address)) (entries trace)))

(defn choices
  "{address value} of the sample sites of `trace`."
  [trace]
  (into {} (comp (filter #(and (= choose-site (:site %))
                               (not (:observed? (:note %)))))
                 (map (juxt :address :value)))
        (entries trace)))

(defn log-joint
  "log p(choices, observations) of `trace`: the sum of every entry's
  :log-prob."
  [trace]
  (transduce (map (comp :log-prob :note)) + 0.0 (entries trace)))

(defn log-weight
  "The importance weight the world of `trace` accumulated."
  [trace]
  (or (rtp/get-state (:trace/world trace) [:inference :log-weight]) 0.0))

;; =============================================================================
;; Metropolis-Hastings: replay plus accept
;; =============================================================================

(defn mh-log-ratio
  "Log acceptance ratio of moving from `old` to `new`, where `new` is `old`
  replayed from its earliest target with kept values elsewhere.

    log a = [log p(new) - log p(old)]
            + [log q(old | new) - log q(new | old)]
            + [log s(targets | new) - log s(targets | old)]

  q(new | old) is the density of everything `new` drew afresh; q(old | new) is
  the density of everything of `old` that `new` did not keep, which the
  reverse move would have to draw (from the prior, hence its :log-prob, or a
  block's :sample density). A
  symmetric move cancels on both sides. s is the probability of selecting the
  targets, `log-selection`; it differs between the traces when the move
  changed how many sites there are to select from.

  A kept value that fell out of its site's support is drawn again
  (`:redrawn?`). That is reversible only if the reverse move would redraw there
  too, i.e. if the new value is outside the OLD support; otherwise the reverse
  keeps it, the old state cannot be reached back, and the move is impossible.

  Entries upstream of the replayed address are the SAME entries in both
  traces and cancel everywhere. That relies on a fork sharing the notes of its
  source by reference, which holds for the in-process worlds a session forks
  and would not survive a serialized trace. With `from`, the replayed
  address, only the entries from it on are compared: the ratio is the same,
  and it costs the replayed suffix instead of the whole trace. `score`
  (fn [entry]) is an entry's log target (default its :log-prob), e.g.
  `tempered-score`."
  ([old new log-selection] (mh-log-ratio old new log-selection nil))
  ([old new log-selection from] (mh-log-ratio old new log-selection from nil))
  ([old new log-selection from score]
   (let [;; with `from`, only the suffix from the replayed address: everything
        ;; upstream is the same entries in both and cancels, so the ratio is
        ;; the same and costs the suffix, not the trace
         suffix (fn [t]
                  (loop [[a & more] (rseq (:trace/order t)) acc ()]
                    (if (nil? a)
                      (vec acc)
                      (let [e (get-in t [:trace/entries a])
                            acc (if (#{choose-site factor-site} (:site e)) (conj acc (assoc e :address a)) acc)]
                        (if (= a from) (vec acc) (recur more acc))))))
         old-entries (if from (suffix old) (entries old))
         new-entries (if from (suffix new) (entries new))
         log-joint (fn [es] (transduce (map (or score (comp :log-prob :note))) + 0.0 es))
         old-by-address (into {} (map (juxt :address identity)) old-entries)
         new-by-address (into {} (map (juxt :address identity)) new-entries)
         same? (fn [a b] (and a b (identical? (:note a) (:note b))))
         forward (transduce
                  (comp (filter latent?)
                        (remove #(same? % (get old-by-address (:address %))))
                        (remove (comp :kept? :note))
                        (remove (comp :symmetric? :note))
                        (map (comp :log-proposal :note)))
                  + 0.0 new-entries)
         backward (transduce
                   (comp (filter latent?)
                         (remove #(same? % (get new-by-address (:address %))))
                         (remove #(:kept? (:note (get new-by-address (:address %)))))
                         (remove #(:symmetric? (:note (get new-by-address (:address %)))))
                        ;; the reverse move draws as the prior proposal does
                         (map (fn [{:keys [note value]}]
                                (or (dist/-draw-logpdf (:dist note) value) (:log-prob note)))))
                   + 0.0 old-entries)
         irreversible? (some (fn [entry]
                               (and (:redrawn? (:note entry))
                                    (when-let [was (get old-by-address (:address entry))]
                                      (finite? (dist/logpdf (:dist (:note was))
                                                            (:value entry))))))
                             new-entries)]
     (if irreversible?
       ##-Inf
       (+ (- (log-joint new-entries) (log-joint old-entries))
          (- backward forward)
          (- (log-selection new) (log-selection old)))))))

(defn uniform-site
  "Select one latent site uniformly. A selection is
  {:targets #{address} :log-selection (fn [trace])}."
  [trace _iteration]
  {:targets #{(m/pick-uniformly (latent-addresses trace))}
   :log-selection (fn [t] (- (Math/log (double (count (latent-addresses t))))))})

(defn prior-proposal
  "Propose a fresh draw from a target site's own distribution."
  [sp _old-entry]
  (let [dist (:dist (:savepoint/payload sp))
        v (dist/draw dist)]
    {:value v :log-proposal (dist/draw-logpdf dist v)}))

(defn random-walk-proposal
  "A symmetric Gaussian step of `step-size` around a real-valued target's
  old value. A target whose law is not continuous (a boolean, an integer
  count, a vector) has no such step; it gets a prior proposal."
  [step-size]
  (fn [sp old-entry]
    (if (dist/continuous? (:dist (:savepoint/payload sp)))
      {:value (+ (:value old-entry) (* step-size (dist/draw (dist/normal 0.0 1.0))))
       :symmetric? true}
      (prior-proposal sp old-entry))))

(defn mh-step
  "One Metropolis-Hastings move on `trace`.

  Options:
    :select    (fn [trace iteration]) -> {:targets #{address}
               :log-selection (fn [trace])}, the sites to move together and
               the log probability of selecting them in a given trace
               (default: `uniform-site`)
    :propose   (fn [sp old-entry]) -> {:value v :log-proposal lq} or
               {:value v :symmetric? true}, for each target (default:
               `prior-proposal`). The reverse move is scored under the prior,
               so a proposal must be the prior or symmetric.
    :constraints as for `policy`; the conditioning of the chain, which
               every move must repeat
    :policy-options the options of the policy the trace was made with
               (interventions, …), which every move repeats; `:keep?`,
               `:draw` and `:constraints` are the move's own
    :iteration passed to :select
    :until, :anchor? passed to the replay (`spindel.trace/run`): a move of
               a partial trace stops where the trace did
    :temperature β: the move targets p(x)·L(x)^β, whatever temperature the
               trace was recorded at (see `policy`)
    :keep-old? an accepted move releases nothing of the old trace: its
               caller releases what no one refers to
    :shared-anchors? the trace's anchors may be shared with other traces
               (SMC particles of one ancestor): an accepted move releases
               the old trace's world only, and the caller its anchors

  The computation is replayed from the earliest target; every other site
  keeps its value and is rescored under its distribution as it is now. The
  loser's worlds are released. Returns a CPS operation resolving
  {:trace t :accepted? boolean :log-ratio r}; a trace with nothing to move
  resolves unchanged."
  ([trace] (mh-step trace nil))
  ([trace {:keys [select propose iteration constraints policy-options until shared-anchors?
                  temperature keep-old?]
           :or {select uniform-site propose prior-proposal iteration 0}
           anchor-pred :anchor?}]
   (fn [resolve reject]
     (let [{:keys [targets log-selection]}
           (when (or until (seq (latent-addresses trace)))
             (random/in-world-stream (:trace/world trace) [::select iteration]
                                     #(select trace iteration)))
           from (if (= 1 (count targets)) (first targets) (trace/earliest trace targets))]
       (if-not from
         (resolve {:trace trace :accepted? false :log-ratio 0.0})
         (let [move (policy (cond-> (assoc policy-options
                                           :keep? true
                                           :constraints (merge (:constraints policy-options) constraints)
                                           :draw (fn [sp old-entry]
                                                   (when (contains? targets (:savepoint/address sp))
                                                     (propose sp old-entry))))
                              temperature (assoc :temperature temperature)))]
           ((trace/replay trace from move (cond-> {:anchor? (or anchor-pred anchor?)}
                                            until (assoc :until until)
                                            ;; the replay's seed from the moving trace's
                                            ;; world and the move index, not from a
                                            ;; shared anchor's fork counter
                                            (sp/seed (:trace/world trace))
                                            (assoc :seed (sp/derive-seed (sp/seed (:trace/world trace))
                                                                         ::move iteration))))
            (fn [proposed]
              (try
                (let [ratio (if (or (:trace/error proposed)
                                    ;; a partial move must end where the trace did
                                    (and until (not (:trace/pending proposed))))
                              ##-Inf
                              (mh-log-ratio trace proposed log-selection from
                                            (when temperature (tempered-score temperature))))
                      accept? (and (not (#?(:clj Double/isNaN :cljs js/isNaN) ratio))
                                   (or (>= ratio 0.0)
                                       (< (Math/log (random/in-world-stream
                                                     (:trace/world proposed) ::accept
                                                     m/uniform01))
                                          ratio)))]
                  (cond
                    (not accept?) (trace/release! proposed trace)
                    keep-old? nil
                    shared-anchors? (do ;; the displaced trace is parked there
                                      (when-let [p (:trace/pending trace)]
                                        (when (sp/pending? p) (sp/abandon p)))
                                      (sp/release-world! (:trace/session trace) (:trace/world trace)))
                    :else (trace/release! trace proposed))
                  (resolve {:trace (if accept? proposed trace)
                            :accepted? accept?
                            :log-ratio ratio}))
                (catch #?(:clj Throwable :cljs :default) error
                  (reject error))))
            reject)))))))

(defn mh-chain
  "`n` Metropolis-Hastings moves from `trace`; `opts` as for `mh-step`.
  Returns a CPS operation resolving {:trace final :moves m :accepted k}: of
  the m moves made, k were accepted. `:on-step` (fn [step-result]) sees every
  step. `:step` (fn [trace opts]) -> CPS resolving a step result replaces
  `mh-step` (e.g. `foerster.hmc/within-gibbs`); a step that makes several
  moves reports `:moves` and `:accepted-moves`, otherwise it is one move,
  accepted when `:accepted?`."
  ([trace n] (mh-chain trace n nil))
  ([trace n {:keys [on-step] move :step :or {move mh-step} :as opts}]
   (fn [resolve reject]
     (letfn [(step [current i moves accepted]
               (if (= i n)
                 (resolve {:trace current :moves moves :accepted accepted})
                 ((move current (assoc opts :iteration i))
                  (fn [{:keys [accepted?] next-trace :trace :as result}]
                    (when on-step (on-step result))
                    (step next-trace (inc i)
                          (+ moves (:moves result 1))
                          (+ accepted (:accepted-moves result (if accepted? 1 0)))))
                  reject)))]
       (step trace 0 0 0)))))

(defn legacy-trace
  "`trace` in the legacy shape of a particle's trace (`[:inference :trace]`):
  {address {:value :distribution :log-prob :observed?}}, and
  {address {:value :deterministic? true}} for deterministic sites."
  [trace]
  (into {}
        (keep (fn [address]
                (let [{:keys [site value note]} (get-in trace [:trace/entries address])]
                  (condp = site
                    choose-site [address {:value value
                                          :distribution (:dist note)
                                          :log-prob (:log-prob note)
                                          :observed? (boolean (:observed? note))}]
                    deterministic-site [address {:value value :deterministic? true}]
                    nil))))
        (:trace/order trace)))
