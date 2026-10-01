(ns org.replikativ.foerster.effects
  "Probabilistic programming effects: unified choose primitive.

  This provides the fundamental primitive for compositional probabilistic programming:
  - choose: Unified effect for both sampling and observation

  A choose site is a savepoint when its world handles `:inference/choose`
  (inference: `foerster.smc`, `foerster.trace`); otherwise it is forward
  simulation."
  (:require [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.engine.effects :as eff]
            [org.replikativ.spindel.engine.addressing :as addressing]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [is.simm.partial-cps.async :as pcps-async]
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
  (throw (ex-info "choose called outside of spin context (should be CPS-transformed)" {})))

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

(defn- forward-choose-fn
  "A choose site outside inference: forward simulation. A site is addressed
  as under inference (`addressing/site-address+path!`) and takes its
  intervention (`intervene!`); otherwise an observed site takes its value and
  scores it into the world's weight, and a latent site a fresh draw from its
  distribution."
  [_runtime args resolve _reject]
  (let [{:keys [source options source-loc]} args
        {:keys [id observe]} options
        ctx rtc/*execution-context*
        [address] (addressing/site-address+path! ctx "sp" :inference/choose source-loc id)
        interventions (rtp/get-state ctx [:inference :interventions])]
    (cond
      ;; Pearl's do-operator: the value is fixed, nothing is scored
      (contains? interventions address)
      (spin-core/resume resolve (get interventions address))

      ;; observe can be boolean false
      (some? observe)
      (do (rtp/swap-state! ctx [:inference :log-weight]
                           (fn [w] (+ (or w 0.0) (dist/logpdf source observe))))
          (spin-core/resume resolve observe))

      :else
      (spin-core/resume resolve (dist/draw source)))))

(defn- choose-handler-fn
  "A choose site is a savepoint when its world handles `:inference/choose`: it
  is published, and a trace policy decides and scores it (`foerster.trace`).
  Otherwise it is forward simulation."
  [runtime args resolve reject]
  (let [ctx rtc/*execution-context*]
    (if (sp/handled? ctx :inference/choose)
      (let [{:keys [source options spin-id source-loc]} args
            {:keys [id observe]} options]
        (sp/publish! ctx
                     {:site :inference/choose
                      :payload {:dist source
                                :observed? (some? observe)
                                :value observe
                                :options (dissoc options :id :observe)}
                      :opts (when id {:id id})
                      :spin-id spin-id
                      :source-loc source-loc}
                     resolve reject))
      (forward-choose-fn runtime args resolve reject))))

;; Wrap handler function with async-effect to create PEffectHandler
(def choose-handler
  "PEffectHandler implementation for choose effect."
  (eff/async-effect choose-handler-fn))

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
  (throw (ex-info "deterministic called outside of spin context (should be CPS-transformed)" {})))

(defn- deterministic-handler-fn
  [_runtime args resolve reject]
  (let [ctx rtc/*execution-context*
        {:keys [value options spin-id source-loc]} args]
    (if (sp/handled? ctx :inference/deterministic)
      (sp/publish! ctx {:site :inference/deterministic
                        :payload {:value value}
                        :opts (when-let [id (:id options)] {:id id})
                        :spin-id spin-id
                        :source-loc source-loc}
                   resolve reject)
      (spin-core/resume resolve value))))

(def deterministic-handler
  (eff/async-effect deterministic-handler-fn))

(defn deterministic-adapter [[value & opts]]
  {:value value :options (apply hash-map opts)})

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
  (rtp/swap-state! rtc/*execution-context* [:inference :interventions]
                   (fn [int] (assoc (or int {}) address value)))
  nil)

;; =============================================================================
;; Effect Registration
;; =============================================================================

(defn choose-adapter
  "Adapter for choose effect: (choose dist & opts) -> {source, options}"
  [args]
  (let [[source & opts] args
        options (apply hash-map opts)]
    {:source source
     :options options}))

(defn sample-adapter
  "Adapter for sample effect: (sample dist & opts) -> {source, options}"
  [args]
  ;; Same as choose adapter - sample is just choose without :observe
  (let [[source & opts] args
        options (apply hash-map opts)]
    {:source source
     :options options}))

(defn observe-adapter
  "Adapter for observe effect: (observe dist value & opts) -> {source, options with :observe}"
  [args]
  ;; observe has different syntax: (observe dist value & opts)
  (let [[source value & opts] args
        options (assoc (apply hash-map opts) :observe value)]
    {:source source
     :options options}))

(defn factor
  "Multiply the weight of this execution by exp(`log-weight`): a score that is
  not the density of a value (a soft constraint, a reward, a likelihood that
  was computed elsewhere).

  `(factor w :barrier true)` is also a barrier of SMC: particles park there
  and the population is resampled, as at an observation — a scored step of a
  program whose data is a score (a verifier's, a value estimate's; see
  `foerster.steer`).

  Must only be called inside a spin; outside, this throws."
  [& _]
  (throw (ex-info "factor called outside of spin context (should be CPS-transformed)" {})))

(defn- factor-handler-fn
  [_runtime args resolve reject]
  (let [ctx rtc/*execution-context*
        {:keys [log-weight barrier spin-id source-loc]} args]
    (if (sp/handled? ctx :inference/factor)
      (sp/publish! ctx {:site :inference/factor
                        :payload (cond-> {:log-weight log-weight}
                                   barrier (assoc :barrier true))
                        :spin-id spin-id
                        :source-loc source-loc}
                   resolve reject)
      (do (rtp/swap-state! ctx [:inference :log-weight]
                           (fn [w] (+ (or w 0.0) log-weight)))
          (spin-core/resume resolve nil)))))

(def factor-handler
  (eff/async-effect factor-handler-fn))

(defn factor-adapter [[log-weight & opts]]
  (let [{:keys [barrier]} (apply hash-map opts)]
    {:log-weight log-weight :barrier (boolean barrier)}))

(defn register-probabilistic-effects!
  "Register probabilistic effects with spindel effect system.

  Registers choose, sample, and observe effects.
  This should be called at library initialization time."
  []
  ;; Register unified choose effect
  ;; NOTE: We use dispatched path (no 4th arg) so adapter is called
  ;; Direct handler path bypasses adapter and expects different signature
  (eff/register-effect-by-symbol!
   'org.replikativ.foerster.effects/choose
   choose-handler  ; PEffectHandler instance
   choose-adapter)

  ;; Register sample (convenience wrapper) - same as choose
  (eff/register-effect-by-symbol!
   'org.replikativ.foerster.effects/sample
   choose-handler
   sample-adapter)

  (eff/register-effect-by-symbol!
   'org.replikativ.foerster.effects/factor
   factor-handler
   factor-adapter)

  (eff/register-effect-by-symbol!
   'org.replikativ.foerster.effects/deterministic
   deterministic-handler
   deterministic-adapter)

  ;; Register observe (different syntax: observe dist value)
  (eff/register-effect-by-symbol!
   'org.replikativ.foerster.effects/observe
   choose-handler
   observe-adapter))

;; Auto-register on namespace load
(register-probabilistic-effects!)
