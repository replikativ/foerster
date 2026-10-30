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
  (:require [is.simm.partial-cps.async :as pcps]
            [org.replikativ.foerster.trace-owner :as owner]
            [org.replikativ.spindel.engine.impl.simple :as engine]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.select :as sel]
            [org.replikativ.foerster.mechanism :as mech]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.dist :as dist])
  #?(:cljs (:require-macros [is.simm.partial-cps.async :as pcps])))

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
  (mech/shifted dist delta))

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
        pairs (::intervention-pairs opts)
        interventions (rtp/get-state world [:inference :interventions])]
    (when (or (seq pairs) (seq interventions))
      (let [d (sel/describe sp)
            match (fn [[selects? transform]] (when (selects? d) transform))]
        (or (some match pairs)
            (some match (intervention-pairs interventions)))))))

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
  [opts sp old-entry {:keys [policy dist shift coupling] :as transform}]
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
            _ (when-not (contains? #{nil :preserve :fresh} coupling)
                (throw (ex-info "Mechanism coupling must be :preserve or :fresh"
                                {:type ::invalid-intervention :coupling coupling})))
            decision (if (= :fresh coupling)
                       (let [v (dist/draw new-dist)]
                         {:value v :note {:dist new-dist :log-prob (dist/logpdf new-dist v) :fresh-noise? true}})
                       (decide-choose (assoc opts ::no-interventions true)
                                      (assoc sp :savepoint/payload (assoc payload :dist new-dist))
                                      old-entry))]
        (assoc-in decision [:note :intervened?] true)))))

(defn barrier-site?
  "Whether a savepoint is where SMC parks a particle: an observation, a
  stream site, or a factor marked `:barrier`."
  [sp]
  (let [payload (:savepoint/payload sp)]
    (boolean (or (:observed? payload) (:stream (:options payload)) (:barrier payload)))))

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

(defn target-temperature
  "Effective target beta: explicit context, then policy options, then one."
  [opts]
  (let [beta (or (:temperature opts) (get-in opts [:policy-options :temperature]) 1.0)]
    (when-not (and (number? beta) (<= 0.0 beta 1.0))
      (throw (ex-info "Target temperature must be in [0,1]" {:type ::invalid-temperature :temperature beta})))
    beta))

(defn tempered-score
  "The log target of a trace entry at temperature `beta`, for entries recorded
  under any temperature: observations and factors by their `:log-lik`, a
  site drawn from its own draw density (a block's :sample) by that density
  plus β times the rest of its target (see `decide-choose`)."
  [beta]
  (fn [{:keys [note value]}]
    (if-let [l (:log-lik note)]
      (+ (:log-base note 0.0) (cond (zero? beta) 0.0 :else (* beta l)))
      (if (and (not= 1.0 (double beta)) (:dist note)
               (not (some note [:observed? :constrained? :intervened? :counterfactual? :simulated?])))
        (if-let [base (dist/-draw-logpdf (:dist note) value)]
          (+ base (if (zero? beta) 0.0 (* beta (- (:log-prob note) base))))
          (:log-prob note))
        (:log-prob note)))))

(declare decide-choose*)

(defn- temper-drawn-site
  "Under `:temperature` β, a latent site whose law has its own draw density q
  (a block's :sample-log-density) follows the geometric path from q to its
  target t: q^(1-β)·t^β. Its note keeps the base log q (`:log-base`) and the
  untempered rest log t - log q (`:log-lik`), as an observation keeps its
  likelihood, and the weight its decision added counts that rest at β. At
  β = 0 the site is its draw density, at β = 1 its target."
  [{:keys [temperature]} world {:keys [value note] :as decision}]
  (let [{:keys [dist log-prob observed?]} note]
    (if (or (nil? temperature) (= 1.0 (double temperature)) observed? (nil? dist)
            (some note [:constrained? :intervened? :counterfactual? :simulated? :log-lik]))
      decision
      (if-let [base (dist/-draw-logpdf dist value)]
        (let [rest (- log-prob base)
              tempered-rest (cond (nil? temperature) rest (zero? temperature) 0.0 :else (* temperature rest))]
          ;; a draw or a non-symmetric proposal weighed the full rest
          (when (and (contains? note :log-proposal) (not (:symmetric? note)))
            (add-weight! world (- tempered-rest rest)))
          (assoc decision :note (assoc note :log-prob (+ base tempered-rest)
                                       :log-base base :log-lik rest)))
        decision))))

(defn- decide-choose
  "A choose site's decision; the note keeps the site's own `:proposal`, the
  law a replay draws it from afresh (see `mh-log-ratio`)."
  [opts sp old-entry]
  (let [decision (temper-drawn-site opts (:savepoint/world sp)
                                    (decide-choose* opts sp old-entry))]
    (if-let [q (:proposal (:options (:savepoint/payload sp)))]
      (assoc-in decision [:note :proposal] q)
      decision)))

(defn- decide-choose*
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

      ;; A twin predicts even an observed site. Leaving the original data
      ;; clamped would silently combine factual and hypothetical worlds.
      (some? (:noise opts))
      (if (:strict? opts)
        (throw (ex-info "Counterfactual site has no compatible factual noise"
                        {:type ::incomplete-coupling :address address
                         :reason (if (contains? (:noise opts) address)
                                   :unsupported-mechanism :missing-noise)}))
        (let [v (dist/draw dist)]
          {:value v :note {:dist dist :log-prob (dist/logpdf dist v) :unaligned? true}}))

      ;; a predictive run: the observed site draws a fresh value from its
      ;; law instead of scoring the data, and adds nothing to the weight
      (and observed? (:simulate-observed? opts))
      (let [v (dist/draw dist)]
        {:value v :note {:dist dist :log-prob (dist/logpdf dist v) :simulated? true}})

      observed?
      (let [lp (dist/logpdf dist value)
            tp (tempered opts lp)]
        (add-weight! world tp)
        {:value value :note {:dist dist :log-prob tp :log-lik lp :observed? true}})

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
                            :log-proposal (:log-proposal drawn)
                            :moved? true}
                     (:symmetric? drawn) (assoc :symmetric? true))})

          ;; A kept value outside the new distribution's support is drawn
          ;; again instead (the site's parents moved it out from under it).
          (finite? kept-lp)
          {:value (:value old-entry)
           :note {:dist dist :log-prob kept-lp :kept? true}}

          :else
          (let [{:keys [init proposal]} (:options (:savepoint/payload sp))
                init (when init? init)
                v (cond (some? init) init
                        proposal (dist/draw proposal)
                        :else (dist/draw dist))
                lp (dist/logpdf dist v)
                ;; a draw from something other than the site's law (the
                ;; site's `:proposal`, a block's :sample) is weighed by the
                ;; difference
                lq (cond (some? init) lp
                         proposal (dist/logpdf proposal v)
                         :else (or (dist/-draw-logpdf dist v) lp))]
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
                 Mechanism replacements preserve factual noise by default;
                 :coupling :fresh explicitly introduces independent noise.
    :noise       {address u}: the site takes its mechanism's value for u
                 (`foerster.mechanism`), observed sites included — a
                 counterfactual world. Sites without noise are marked
                 `:unaligned?`
    :strict?     with :noise, refuse a site without compatible factual noise;
                 otherwise draw fresh, including observed sites
    :simulate-observed? true: observed sites draw a fresh value from their
                 distribution instead of scoring the data (predictive runs:
                 with `:constraints` holding a posterior draw's latents)
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
   (let [fallback (or fallback trace/payload-policy)
         prepared (if (seq (:interventions opts))
                    (assoc opts ::intervention-pairs (vec (intervention-pairs (:interventions opts))))
                    opts)]
     (with-meta
       (fn [sp old-entry]
         (let [site (:savepoint/site sp)]
           (cond
             (= choose-site site)
             (cond-> (random/in-world-stream (:savepoint/world sp) (:savepoint/address sp)
                                             #(decide-choose prepared sp old-entry))
               true (assoc-in [:note :barrier] (count-barrier! sp))
             ;; with :noise, a site that got no factual noise (it did not
             ;; exist in the factual world, or its law is not a mechanism)
             ;; is reported
               (and (:noise opts) (not (contains? (:noise opts) (:savepoint/address sp))))
               (update :note (fn [n] (if (:intervened? n) n (assoc n :unaligned? true)))))

             (= deterministic-site site)
             (let [intervention (intervention-for prepared sp)]
               (if intervention
                 (if (or (contains? intervention :do) (:policy intervention))
                   (update (decide-intervened prepared sp old-entry intervention)
                           :note assoc :deterministic? true)
                   (throw (ex-info "A deterministic intervention needs :do or :policy"
                                   {:type ::invalid-intervention :intervention intervention})))
                 {:value (:value (:savepoint/payload sp)) :note {:deterministic? true}}))

             (= factor-site site)
             (let [w (:log-weight (:savepoint/payload sp))
                   tw (tempered opts w)]
               (add-weight! (:savepoint/world sp) tw)
               {:value nil :note {:log-prob tw :log-lik w :factor? true
                                  :barrier (count-barrier! sp)}})

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
  world. Once a keyed region has been entered, sites take no anchors:
  children can still be in flight, and old downstream checkpoints would be
  invalidated by later key updates. Independent keys use reexecute; a broader
  change needs a valid upstream anchor. This also covers nested child spins.
  Pass as :anchor? to trace/run and trace/replay."
  [sp]
  (and (= choose-site (:savepoint/site sp))
       (not (:observed? (:savepoint/payload sp)))
       (empty? (rtp/get-state (:savepoint/world sp) [:keyed/spins]))))

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

(defn latent-index
  "{:addresses [address …] :position {address i}} of the latent sites of
  `trace`: carried by a trace an incremental move made (patched with what
  the move changed), built from the entries otherwise."
  [trace]
  (or (::latent trace)
      (let [v (latent-addresses trace)] {:addresses v :position (zipmap v (range))})))

(defn latent-count
  "How many latent sites `trace` has."
  [trace]
  (count (:addresses (latent-index trace))))

(defn- patch-latent
  "`index` (of the trace `new` was updated from) for `new`, given the
  addresses the update `changed`: O(changed)."
  [index new changed]
  (reduce (fn [{:keys [addresses position] :as idx} a]
            (let [e (get-in new [:trace/entries a])
                  now? (and e (latent? e))
                  at (get position a)]
              (cond
                (and now? (nil? at))
                {:addresses (conj addresses a) :position (assoc position a (count addresses))}

                (and (not now?) at)
                ;; move the last address into the hole
                (let [last-i (dec (count addresses))
                      last-a (nth addresses last-i)]
                  {:addresses (pop (assoc addresses at last-a))
                   :position (cond-> (dissoc position a)
                               (not= a last-a) (assoc last-a at))})

                :else idx)))
          index changed))

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
  `tempered-score`. With `changed`, the addresses an update rewrote, added
  or removed (`changed-addresses`), only those are compared: every other
  entry is the same in both and cancels."
  ([old new log-selection] (mh-log-ratio old new log-selection nil))
  ([old new log-selection from] (mh-log-ratio old new log-selection from nil))
  ([old new log-selection from score] (mh-log-ratio old new log-selection from score nil))
  ([old new log-selection from score changed]
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
         only (fn [t]
                (into [] (keep (fn [a]
                                 (let [e (get-in t [:trace/entries a])]
                                   (when (#{choose-site factor-site} (:site e))
                                     (assoc e :address a)))))
                      changed))
         old-entries (cond changed (only old) from (suffix old) :else (entries old))
         new-entries (cond changed (only new) from (suffix new) :else (entries new))
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
                         ;; the reverse move draws its own target as the prior
                         ;; proposal does, and any other site it reaches
                         ;; afresh as a replay does: from the site's
                         ;; `:proposal`, if it has one
                         (map (fn [{:keys [address note value]}]
                                (let [moved? (:moved? (:note (get new-by-address address)))]
                                  (if (and (:proposal note) (not moved?))
                                    (dist/logpdf (:proposal note) value)
                                    (or (dist/-draw-logpdf (:dist note) value) (:log-prob note)))))))
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

(defn changed-addresses
  "Addresses rewritten, added or removed. Released Spindel 0.1.140 does not
  report :trace/removed, so deletion accounting conservatively scans keys."
  [old new]
  (into (set (:trace/updated new))
        (or (:trace/removed new)
            (remove #(contains? (:trace/entries new) %) (keys (:trace/entries old))))))

(defn uniform-site
  "Select one latent site uniformly. A selection is
  {:targets #{address} :log-selection (fn [trace])}."
  [trace _iteration]
  {:targets #{(m/pick-uniformly (:addresses (latent-index trace)))}
   :log-selection (fn [t] (- (Math/log (double (latent-count t)))))})

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

(defn- subtree [world ids]
  (loop [todo (vec ids) seen #{}]
    (if-let [id (peek todo)]
      (if (contains? seen id)
        (recur (pop todo) seen)
        (recur (into (pop todo) (engine/live-children (rtp/get-state world [:nodes id])))
               (conj seen id)))
      seen)))

(defn- keyed-layout [t]
  (let [world (:trace/world t)
        trees (into {} (map (fn [id] [id (subtree world [id])]))
                    (keys (rtp/get-state world [:keyed/spins])))
        ;; Prefer the nearest key. Re-running an outer key alone can adopt an
        ;; unchanged inner key and never visit the requested inner site.
        owners (reduce (fn [owners [id nodes]]
                         (reduce (fn [m child] (if (contains? m child) m (assoc m child id))) owners nodes))
                       {} (sort-by (fn [[id nodes]] [(count nodes) (pr-str id)]) trees))
        collections (into {} (mapcat (fn [[collection state]]
                                       (map (fn [child] [(spin-core/spin-id child) collection])
                                            (vals (:children state)))))
                          (rtp/get-state world [:keyed]))]
    {:owners owners :trees trees :collections collections}))

(defn- replay-origin [t targets]
  (let [first-target (if (= 1 (count targets)) (first targets) (trace/earliest t targets))
        pending? (fn [address] (when-let [anchor (get-in t [:trace/entries address :savepoint])]
                                 (sp/pending? anchor)))]
    (if (pending? first-target) first-target
        (last (filter pending? (take-while #(not= first-target %) (:trace/order t)))))))

(defn copy-complete
  "Fork a settled end-world without re-running its effects. The caller owns
  shared anchors and must keep every live population reachable until release."
  [t opts]
  (pcps/async
   (when (or (not (contains? t :trace/result)) (:trace/pending t) (:trace/error t))
     (throw (ex-info "Only complete successful traces can be copied"
                     {:type ::incomplete-copy})))
   (let [copy (pcps/await (trace/update t [] (policy {:keep? true}) opts))]
     (assoc copy ::latent (latent-index t)))))

(defn- policy-contribution [entry]
  (let [{:keys [log-prob log-proposal observed? constrained? factor? symmetric?]} (:note entry)]
    (cond
      (or observed? constrained? factor?) log-prob
      (and (number? log-proposal) (not symmetric?)) (- log-prob log-proposal)
      :else 0.0)))

(defn- update-ledger! [old next changed]
  ;; trace/update starts at the end-world, not a prefix checkpoint. Replace
  ;; rewritten policy contributions instead of accumulating them twice.
  ;; This is bookkeeping, not an evidence certificate for a Markov state.
  (let [sum (fn [t addresses]
              (transduce (map #(policy-contribution (get-in t [:trace/entries %]))) + 0.0 addresses))
        weight (+ (log-weight old) (- (sum old changed)) (sum next changed))
        weight (if (#?(:clj Double/isNaN :cljs js/isNaN) weight)
                 (sum next (keys (:trace/entries next))) weight)]
    (rtp/swap-state! (:trace/world next) [:inference :log-weight] (constantly weight))))

(defn reexecute
  "Rewrite selected sites through one execution interface. Complete independent
  keyed regions re-run their owning keys; other moves replay from the latest
  valid preceding anchor. Incremental parent-side inference effects and keyed
  streaming frontiers are refused unless safe prefix replay is available.
  Result metadata ::changed / ::restart describes the score delta to compare."
  [t targets move opts]
  (pcps/async
   (when-let [missing (seq (remove #(contains? (:trace/entries t) %) targets))]
     (throw (ex-info "A rewrite targets absent sites" {:type ::unknown-target :addresses (vec missing)})))
   (let [world (:trace/world t)
         {:keys [owners trees collections]} (keyed-layout t)
         ids (mapv #(get owners (get-in t [:trace/entries % :spin])) targets)
         keyed? (and (seq ids) (every? some? ids))
         _ (when (and (seq owners) (or (:until opts) (:trace/pending t)))
             (throw (ex-info "Incremental keyed moves require complete traces"
                             {:type ::incremental-frontier})))
         groups (when keyed? (set (map collections ids)))
         overlap? (when keyed?
                    (some (fn [id] (some #(and (not= id %) (contains? (get trees %) id)) ids)) ids))
         selected (when keyed? (subtree world ids))
         region (when (and (= 1 (count groups)) (not (contains? groups nil)))
                  (into #{} (mapcat (fn [[id group]] (when (contains? groups group) (get trees id)))) collections))
         touched (when keyed? (into #{} (keep (fn [[a e]] (when (contains? selected (:spin e)) a)))
                                    (:trace/entries t)))
         suffix (when keyed? (drop-while #(not (contains? touched %)) (:trace/order t)))
         ;; Only siblings of the SAME keyed collection are independent here.
         ;; A resumed parent can drop a whole later keyed collection, whose
         ;; dead sites the released runtime would otherwise retain.
         local? (and keyed? region (not overlap?)
                     (not-any? #(not (contains? region (get-in t [:trace/entries % :spin]))) suffix))
         replay (fn []
                  (let [from (replay-origin t targets)]
                    (when-not from
                      (throw (ex-info "No safe replay anchor precedes the requested sites; use self-contained keyed regions or an anchored prefix"
                                      {:type ::no-replay-origin :targets (vec targets)})))
                    (fn [deliver fail]
                      ;; A replay beginning before a key must visit that key,
                      ;; even when its captured inputs are unchanged.
                      ((trace/replay (if (= #{from} (set targets)) t (dissoc t :trace/world))
                                     from move (assoc opts :anchor? (or (:anchor? opts) anchor?)))
                       #(deliver (assoc % ::restart from)) fail))))]
     (if-not local?
       (pcps/await (replay))
       (let [next (pcps/await (trace/update t (vec (distinct ids)) move (select-keys opts [:seed])))
             allowed (subtree (:trace/world next) ids)
             updated (set (:trace/updated next))
             unvisited? (some #(and (contains? (:trace/entries next) %) (not (contains? updated %))) targets)
             outside? (some #(not (contains? allowed (get-in next [:trace/entries % :spin]))) updated)]
         (if (and (or outside? unvisited?) (not (:trace/error next)))
           (do (owner/reclaim! [next] [t])
               (pcps/await (replay)))
           (let [changed (changed-addresses t next)]
             (update-ledger! t next changed)
             (assoc next ::changed changed ::latent (patch-latent (latent-index t) next changed)))))))))

(defn mh-step
  "One MH move using keyed re-execution for complete independent regions and
  anchored replay otherwise. :select / :propose / :constraints / :temperature
  retain their usual meaning. :keep-old? and :shared-anchors? are ownership
  policies; :until denotes an anchored replay frontier."
  ([t] (mh-step t nil))
  ([t {:keys [select propose iteration constraints policy-options temperature keep-old? shared-anchors? until]
       :or {select uniform-site propose prior-proposal iteration 0} :as opts}]
   (pcps/async
    (let [{:keys [targets log-selection]}
          (when (or until (pos? (latent-count t)))
            (random/in-world-stream (:trace/world t) [::select iteration] #(select t iteration)))]
      (if-not (seq targets)
        {:trace t :accepted? false :log-ratio 0.0}
        (let [move (policy (cond-> (assoc policy-options :keep? true
                                          :constraints (merge (:constraints policy-options) constraints)
                                          :draw (fn [site old-entry]
                                                  (when (contains? targets (:savepoint/address site))
                                                    (propose site old-entry))))
                             (some? temperature) (assoc :temperature temperature)))
              proposed (pcps/await
                        (reexecute t targets move
                                   (cond-> (select-keys opts [:until :anchor?])
                                     (sp/seed (:trace/world t))
                                     (assoc :seed (sp/derive-seed (sp/seed (:trace/world t)) ::move iteration)))))]
          (try
            (let [ratio (if (or (:trace/error proposed) (and until (not (:trace/pending proposed))))
                          ##-Inf
                          (mh-log-ratio t proposed log-selection (::restart proposed)
                                        (tempered-score (target-temperature opts))
                                        (::changed proposed)))
                  accept? (and (not (#?(:clj Double/isNaN :cljs js/isNaN) ratio))
                               (or (>= ratio 0.0)
                                   (random/in-world-stream (:trace/world proposed) ::accept #(random/accept-log? ratio))))]
              (cond
                (not accept?) (trace/release! proposed t)
                keep-old? nil
                shared-anchors? (do (when-let [pending (:trace/pending t)]
                                      (when (sp/pending? pending) (sp/abandon pending)))
                                    (sp/release-world! (:trace/session t) (:trace/world t)))
                :else (trace/release! t proposed))
              {:trace (if accept? proposed t) :accepted? accept? :log-ratio ratio})
            (catch #?(:clj Throwable :cljs :default) error
              (owner/reclaim! [proposed] [t])
              (throw error)))))))))

(defn update-step
  "Incremental-aware MH, also used by the ordinary shared transition path."
  ([t] (mh-step t))
  ([t opts] (mh-step t opts)))

(defn- gibbs-shape [t]
  (into {} (map (fn [[a e]] [a [(:site e) (select-keys (:note e)
                                                       [:observed? :constrained? :intervened? :deterministic?])]]))
        (:trace/entries t)))

(defn update-gibbs
  "Exact finite-support Gibbs on complete traces with fixed site structure.
  Every other latent must stay fixed with no auxiliary redraws. Candidates
  violating this contract are refused; they are not an approximate Gibbs move.
  :temperature, target policy and ownership follow mh-step."
  ([t address] (update-gibbs t address nil))
  ([t address {:keys [iteration constraints policy-options keep-old?] :as opts
               :or {iteration 0}}]
   (pcps/async
    (let [temperature (target-temperature opts)
          entry (get-in t [:trace/entries address])
          current (:value entry)
          _ (when-not (latent? entry)
              (throw (ex-info "Gibbs targets an unconstrained latent site"
                              {:type ::gibbs-fixed-site :address address})))
          _ (when-not (and (number? temperature) (<= 0.0 temperature 1.0))
              (throw (ex-info "Gibbs temperature must be in [0,1]" {:type ::gibbs-invalid-score})))
          values (vec (dist/support (get-in entry [:note :dist])))
          score (tempered-score temperature)
          total (fn [x] (transduce (map score) + 0.0 (entries x)))
          old-total (total t)
          shape (gibbs-shape t)
          others (dissoc (into {} (map (juxt :address :value)) (filter latent? (entries t))) address)
          candidates (atom [])]
      (when (or (:trace/pending t) (:trace/error t) (not (contains? t :trace/result)))
        (throw (ex-info "Gibbs requires a complete successful trace" {:type ::gibbs-frontier})))
      (when-not (and (latent? entry) (not (contains? (merge (:constraints policy-options) constraints) address)))
        (throw (ex-info "Gibbs targets an unconstrained latent site" {:type ::gibbs-fixed-site :address address})))
      (when-not (and (seq values) (some #(= current %) values))
        (throw (ex-info "Gibbs needs finite support containing the current value" {:type ::infinite-support :address address})))
      (when-not (and (number? temperature) (<= 0.0 temperature 1.0) (finite? old-total))
        (throw (ex-info "Gibbs requires a finite current target score and temperature in [0,1]"
                        {:type ::gibbs-invalid-score :temperature temperature})))
      (try
        (loop [remaining (seq (map-indexed vector values))]
          (when-let [[i value] (first remaining)]
            (when-not (= value current)
              (let [move (policy (assoc policy-options :keep? true :temperature temperature
                                        :constraints (merge (:constraints policy-options) constraints)
                                        :draw (fn [site _]
                                                (when (= address (:savepoint/address site))
                                                  {:value value :symmetric? true}))))
                    next (pcps/await (reexecute t #{address} move
                                                {:seed (sp/derive-seed (sp/seed (:trace/world t)) [::gibbs iteration] i)}))]
                (swap! candidates conj [value next])
                (when (or (:trace/error next) (not= value (get-in next [:trace/entries address :value]))
                          (not= shape (gibbs-shape next))
                          (some (fn [[a v]] (or (not= v (get-in next [:trace/entries a :value]))
                                                (get-in next [:trace/entries a :note :redrawn?]))) others))
                  (throw (ex-info "Gibbs requires fixed site structure and unchanged auxiliary choices"
                                  {:type ::gibbs-structure :address address})))))
            (recur (next remaining))))
        (let [options (into [[current t]] @candidates)
              increments (mapv (fn [[_ x]] (- (total x) old-total)) options)
              _ (when-not (every? #(and (number? %) (not (#?(:clj Double/isNaN :cljs js/isNaN) %)) (not= ##Inf %)) increments)
                  (throw (ex-info "Gibbs candidate scores are invalid" {:type ::gibbs-invalid-score})))
              selected (random/in-world-stream (:trace/world t) [::gibbs iteration]
                                               #(m/sample-categorical (m/normalize-log-weights increments)))
              [value chosen] (nth options selected)
              live (cond-> [chosen] keep-old? (conj t))]
          (rtp/swap-state! (:trace/world chosen) [:engine/reuse-source] (constantly nil))
          (owner/reclaim! (mapv second options) live)
          {:trace chosen :value value :log-weights (zipmap (map first options) increments)
           :accepted? true :moves 1 :accepted-moves 1})
        (catch #?(:clj Throwable :cljs :default) error
          (owner/reclaim! (mapv second @candidates) [t])
          (throw error)))))))

(defn mh-chain
  "`n` Metropolis-Hastings moves from `trace`; `opts` as for `mh-step`.
  Returns a CPS operation resolving {:trace final :moves m :accepted k}: of
  the m moves made, k were accepted. `:on-step` (fn [step-result]) sees every
  step. `:step` (fn [trace opts]) -> CPS resolving a step result replaces
  `mh-step` (e.g. `foerster.hmc/within-gibbs`); a step that makes several
  moves reports `:moves` and `:accepted-moves`, otherwise it is one move,
  accepted when `:accepted?`. A move's randomness is keyed by its number in
  the trace's world, so the moves made from one world need distinct numbers:
  :iteration-stride (default 1) reserves that many RNG indices per outer
  iteration, for composed transitions. They are numbered from
  `:first-iteration`, or — when it is not given —
  from a counter the world keeps, so calling `mh-chain` again on the same
  trace (after a rejection, the world is the same) draws afresh."
  ([trace n] (mh-chain trace n nil))
  ([trace n {:keys [on-step first-iteration iteration-stride] move :step
             :or {move mh-step iteration-stride 1} :as opts}]
   (fn [resolve reject]
     (let [span (* n iteration-stride)
           first-iteration (or first-iteration
                               (- (rtp/swap-state! (:trace/world trace) [:inference ::chain-moves] #(+ (or % 0) span))
                                  span))]
       (letfn [(step [current i moves accepted]
                 (if (= i n)
                   (resolve {:trace current :moves moves :accepted accepted})
                   ((move current (assoc opts :iteration (+ first-iteration (* i iteration-stride))))
                    (fn [{:keys [accepted?] next-trace :trace :as result}]
                      (when on-step (on-step result))
                      (step next-trace (inc i)
                            (+ moves (:moves result 1))
                            (+ accepted (:accepted-moves result (if accepted? 1 0)))))
                    reject)))]
         (step trace 0 0 0))))))

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
