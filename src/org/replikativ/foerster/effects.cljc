(ns org.replikativ.foerster.effects
  "Probabilistic programming effects: unified choose primitive.

  This provides the fundamental primitive for compositional probabilistic programming:
  - choose: Unified effect for both sampling and observation

  A choose site is a savepoint when its world handles `:inference/choose`
  (inference: `foerster.inference.smc`, `foerster.runtime.trace`); otherwise it is forward
  simulation."
  (:require [org.replikativ.spindel.v2.core :as v2]
            [org.replikativ.spindel.v2.savepoint :as sp]
            [org.replikativ.foerster.dist :as dist])
  ;; the spin macro knows sample/observe/factor only once this namespace has
  ;; registered them on the JVM, where ClojureScript's macros expand
  #?(:cljs (:require-macros [org.replikativ.foerster.effects])))

;; =============================================================================
;; Public API Shims
;; =============================================================================

(defn choose
  "Unified primitive for probabilistic choice.

  Must only be called inside a spin; outside, this throws.

  Examples:
    ;; Sample from prior
    (choose (normal 0 1))

    ;; Observe value
    (choose (normal 0 1) :observe 1.5)

    ;; With explicit ID
    (choose (normal mu sigma) :id :my-param :observe observed)

    ;; With initial value (the first state of a Markov chain)
    (choose (normal 0 1) :init 0.5)"
  [& _]
  (throw (ex-info "choose called outside of a task body" {})))

(defn sample
  "Sample from a distribution (convenience wrapper around choose).

  Examples:
    (sample (normal 0 1))
    (sample (uniform 0 1) :id :my-param)
    (sample (normal 0 1) :id :mu :proposal (normal 0.8 0.5))

  `:proposal` is a distribution a fresh draw under inference comes from
  instead (a guide, possibly computed from the data: amortized inference);
  the weight takes log p − log q, so the target is unchanged. Replays, moves
  and a policy's `:draw` take precedence."
  [dist & opts]
  (apply choose dist opts))

(defn observe
  "Condition on observed value (convenience wrapper around choose).

  Examples:
    (observe (normal mu sigma) observed-value)"
  [dist value & opts]
  (apply choose dist :observe value opts))

;; =============================================================================
;; Choose Effect Handler
;; =============================================================================

(defn- forward-choose
  "A choose site outside inference: forward simulation. A site is addressed
  as under inference and takes its intervention (`intervene!`); otherwise an
  observed site takes its value and scores it into the world's weight, and a
  latent site a fresh draw from its distribution."
  [source options address]
  (let [w (v2/current-world)
        {:keys [observe]} options
        interventions (v2/get-in-world w [:inference :interventions])]
    (cond
      ;; Pearl's do-operator: the value is fixed, nothing is scored
      (contains? interventions address) [:value (get interventions address)]

      ;; observe can be boolean false, or nil (an outcome of a categorical)
      (contains? options :observe)
      (do (v2/update-in-world! w [:inference :log-weight]
                               (fn [lw] (+ (or lw 0.0) (dist/logpdf source observe))))
          [:value observe])

      :else [:value (dist/draw source)])))

(defn- choose-op
  "A choose site is a savepoint when its world handles `:inference/choose`: it
  is published, and a trace policy decides and scores it (`foerster.runtime.trace`).
  Otherwise it is forward simulation. `:site` (the call site, as v2 hashes
  it) matches it to its static site (`foerster.analysis`)."
  [source options ctx]
  (let [address (sp/address (:id options) ctx)]
    (if (sp/handled? (v2/current-world) :inference/choose)
      (sp/publish-op :inference/choose
                     {:dist source
                      :observed? (contains? options :observe)
                      :value (:observe options)
                      :options (dissoc options :id :observe)
                      :site (:site ctx)}
                     address)
      (forward-choose source options address))))

;; =============================================================================
;; Deterministic sites
;; =============================================================================

(defn deterministic
  "Record `value`, a quantity computed from the program's choices, in the
  trace under its `:id`: it adds no randomness and no weight, and returns
  `value`. Posteriors of it are read from the particles' traces
  (`measure/site-value`); traces carrying the values a model computed are
  also what learned proposals train on.

    (let [h (sample (dist/normal 1.7 0.1) :id :height)
          w (sample (dist/normal 70 10) :id :weight)]
      (deterministic (/ w (* h h)) :id :bmi))

  Must only be called inside a spin; outside, this throws."
  [& _]
  (throw (ex-info "deterministic called outside of a task body" {})))

(defn- deterministic-op [value options ctx]
  (if (sp/handled? (v2/current-world) :inference/deterministic)
    (sp/publish-op :inference/deterministic {:value value} (sp/address (:id options) ctx))
    [:value value]))

;; =============================================================================
;; Interventions
;; =============================================================================

(defn intervene!
  "Set intervention value (Pearl's do-operator).

  This cuts the connection to parent nodes in the graphical model.
  Used for causal inference and counterfactual queries.

  Example:
    (intervene! :treatment 1.0)  ; Force treatment = 1.0
    (let [outcome (choose (normal treatment 1))]
      outcome)"
  [address value]
  (v2/update-in-world! (v2/current-world) [:inference :interventions]
                       (fn [int] (assoc (or int {}) address value)))
  nil)

;; =============================================================================
;; Effect Registration
;; =============================================================================

(def site-options
  "What `sample`, `observe` and `choose` take after the distribution:
  `:id` (the address), `:observe` (the observed value), `:proposal` (a
  distribution fresh draws come from instead), `:init` (a Markov chain's
  first state) and `:stream` (the value arrives from outside, SMC's
  `stream`). Any other key is refused: a misspelt option would otherwise
  be ignored."
  #{:id :observe :proposal :init :stream})

(defn- site-options!
  "`opts` (key value …) as a map, refusing keys `site-options` does not name."
  [opts]
  (when (odd? (count opts))
    (throw (ex-info "Site options come in key-value pairs" {:type ::invalid-site-options :options (vec opts)})))
  (let [options (apply hash-map opts)]
    (when-let [unknown (seq (remove site-options (keys options)))]
      (throw (ex-info (str "Unknown site options " (vec unknown) "; a site takes " site-options)
                      {:type ::invalid-site-options :unknown (vec unknown) :allowed site-options})))
    options))

(defn factor
  "Multiply the weight of this execution by exp(`log-weight`): a score that is
  not the density of a value (a soft constraint, a reward, a likelihood that
  was computed elsewhere).

  `(factor w :barrier true)` is also a barrier of SMC: particles park there
  and the population is resampled, as at an observation — a scored step of a
  program whose data is a score (a verifier's, a value estimate's; see
  `foerster.inference.steer`).

  Must only be called inside a spin; outside, this throws."
  [& _]
  (throw (ex-info "factor called outside of a task body" {})))

(defn- factor-op [log-weight barrier ctx]
  (let [w (v2/current-world)]
    (if (sp/handled? w :inference/factor)
      (sp/publish-op :inference/factor
                     (cond-> {:log-weight log-weight :site (:site ctx)} barrier (assoc :barrier true))
                     (sp/address nil ctx))
      (do (v2/update-in-world! w [:inference :log-weight] (fn [lw] (+ (or lw 0.0) log-weight)))
          [:value nil]))))

(def handlers
  "The v2 effect handlers `(fn [args ctx] op)` of the probabilistic effects,
  by name; a namespace that re-exports an effect registers its own symbol
  with the same handler (`foerster.core`)."
  {'choose (fn [[source & opts] ctx] (choose-op source (site-options! opts) ctx))
   'sample (fn [[source & opts] ctx] (choose-op source (site-options! opts) ctx))
   'observe (fn [[source value & opts] ctx]
              (choose-op source (assoc (site-options! opts) :observe value) ctx))
   'factor (fn [[log-weight & opts] ctx]
             (factor-op log-weight (boolean (:barrier (apply hash-map opts))) ctx))
   'deterministic (fn [[value & opts] ctx] (deterministic-op value (apply hash-map opts) ctx))})

(defn register-probabilistic-effects!
  "Registers choose, sample, observe, factor and deterministic in namespace
  `ns-name` (default this one) as v2 effects of task bodies
  (`v2/register-effect!`). Bodies compiled before this do not see them."
  ([] (register-probabilistic-effects! "org.replikativ.foerster.effects"))
  ([ns-name]
   (doseq [[n h] handlers]
     (v2/register-effect! (symbol ns-name (name n)) h))))

;; Auto-register on namespace load
(register-probabilistic-effects!)
