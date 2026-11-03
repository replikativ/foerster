;; # Interventions and Counterfactuals

;; A probabilistic program can represent a causal model when its mechanisms
;; and their noise coupling have a structural interpretation. Execution order
;; alone does not establish causality. Under those assumptions we can ask
;; what we expect having **seen** something, what happens if we **do**
;; something, and what **would have** happened (Pearl's ladder).
;; The examples below explicitly assume those structural equations and use
;; Foerster's documented per-distribution noise couplings: standardized
;; Gaussian residuals for Normals and threshold noise for binary variables.

(ns foerster.counterfactuals
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.causal.counterfactual :as cf]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.model.mechanism :as mech]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.runtime.trace :as trace]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.select :as select]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def world (sp/create-execution-context))
(random/set-seed! 3)

;; ## A confounded model
;;
;; A hidden common cause Z drives both X and Y, and X also affects Y:
;; Z ~ N(0, 1), X ~ N(Z, 1), Y ~ N(X + Z, 1).

(defn scm []
  (spin
   (let [z (sample (dist/normal 0.0 1.0) :id :z)
         x (sample (dist/normal z 1.0) :id :x)
         y (sample (dist/normal (+ x z) 1.0) :id :y)]
     {:z z :x x :y y})))

(defn expected-y [policy]
  (:mean (infer/query (sp/with-context world
                        @(infer/importance-sampling (scm) 4000 {:policy (trace/policy policy)}))
                      :y)))

;; ## Seeing is not doing
;;
;; **Seeing** X = 1 conditions on it: it says something about Z too (Z is
;; likely positive), so E[Y | X = 1] = 1.5. A *constraint* fixes a site to a
;; value and scores it, like an observation:

(expected-y {:constraints {:x 1.0}})

;; **Doing** X = 1 sets it, cutting it off from Z: E[Y | do(X = 1)] = 1. An
;; *intervention* replaces the site's mechanism and scores nothing:

(expected-y {:interventions {:x {:do 1.0}}})

;; Interventions are keyed by address or by spindel **selector**, so one
;; intervention can reach every site of a kind, or every site under a path:

(expected-y {:interventions {(select/id :x) {:do 1.0}}})

;; Besides `:do`, an intervention can set a new mechanism (`:dist`), shift
;; the old one (`:shift`), or compute the value from the choices made so far
;; (`:policy`, a function of them):

{:new-mechanism (expected-y {:interventions {:x {:dist (dist/normal 3.0 0.1)}}})
 :shifted (expected-y {:interventions {:x {:shift 2.0}}})
 :policy (expected-y {:interventions {:x {:policy (fn [choices] (* 2.0 (:z choices)))}}})}

;; (Exact: 3, 2 and 0.)

;; ## Counterfactuals
;;
;; A counterfactual asks about *this* situation, not a new one: we saw
;; X = 1 and Y = 2.5 — had X been 2, what would Y have been? The answer
;; keeps everything about the situation we saw — including the noise that
;; made Y what it was — and changes only X.
;;
;; To do that, each site is read as a structural equation x = f(u) of
;; exogenous noise u (`foerster.model.mechanism`). `counterfactual`:
;;
;; 1. infers the factual world from the evidence (**abduction**: which noise
;;    explains what we saw),
;; 2. applies the intervention (**action**), and
;; 3. replays the model with the abducted noise (**prediction**).

(def pairs
  (let [p (promise)]
    ((cf/counterfactual (sp/with-context world (scm))
                        {:evidence {:x 1.0 :y 2.5}
                         :interventions {:x {:do 2.0}}
                         :particles 50})
     #(deliver p %) #(deliver p %))
    (deref p 60000 :timeout)))

;; `counterfactual` takes the model spin, as `gfi` does — it runs it in a
;; factual and a counterfactual world per particle — and returns a CPS
;; operation resolving factual/counterfactual pairs, one per particle, with
;; weights:

(select-keys (first pairs) [:factual :counterfactual :weight])

;; Z is uncertain given the evidence, but Y − X = Z + noise is known
;; exactly (1.5), so every particle agrees: had X been 2, Y would have been
;; 3.5.

(set (map #(/ (Math/round (* 1e9 (:y (:counterfactual %)))) 1e9) pairs))

;; ## A binary example: was it necessary?
;;
;; X ~ flip(0.5), and Y = true for sure when X is, with probability 0.3
;; otherwise. We saw X and Y both true. The **probability of necessity** —
;; would Y have been false had X been false? — is 0.7.

(defn binary []
  (sp/with-context world
    (spin
     (let [x (sample (dist/flip 0.5) :id :x)
           y (sample (dist/flip (if x 1.0 0.3)) :id :y)]
       {:x x :y y}))))

(let [p (promise)]
  ((cf/counterfactual (binary) {:evidence {:x true :y true}
                                :interventions {:x {:do false}}
                                :particles 1500})
   #(deliver p %) #(deliver p %))
  (let [pairs (deref p 60000 [])
        ws (weights/normalize-log-weights (mapv :weight pairs))]
    (reduce + (map (fn [pair w] (if (:y (:counterfactual pair)) 0.0 w)) pairs ws))))

;; A discrete site has many noise values for one outcome; abduction draws
;; one from its posterior. Which convention a discrete mechanism uses (here
;; the inverse CDF, x = F⁻¹(u)) changes counterfactual answers and cannot be
;; told from data, so foerster fixes and documents it
;; (`foerster.model.mechanism`) rather than guessing.
;;
;; The lower-level `counterfactual` operation above permits a newly reached
;; site to draw fresh noise and reports it in `:unaligned`. This weakens the
;; factual coupling. Prefer the strict `query` interface below for new code:
;; it refuses unsupported or unaligned mechanisms unless explicitly allowed.

;; ## Reuse a posterior across worlds
;;
;; Factual inference can be chosen separately from the counterfactual query.
;; The ordinary measure keeps site laws and values, enough to reconstruct
;; supported noise. The same abducted draw is shared across all named worlds.
;; `query` is strict by default: an unsupported or newly reached stochastic
;; mechanism is refused instead of silently weakening the coupling.

(def factual-posterior
  (sp/with-context world
    @(infer/importance-sampling (scm) 25
                                {:policy (trace/policy {:constraints {:x 1.0 :y 2.5}})})))

(def named-worlds
  (let [p (promise)]
    ((cf/query (sp/with-context world (scm))
               {:posterior factual-posterior
                :worlds [[:untreated {:x {:do 0.0}}]
                         [:treated {:x {:do 2.0}}]]})
     #(deliver p %) #(deliver p %))
    (let [result (deref p 60000 ::timeout)]
      (when (or (= ::timeout result) (instance? Throwable result))
        (throw (ex-info "Counterfactual query failed" {:result result})))
      result)))

;; The factual residual Y − X is 1.5. Each draw therefore predicts Y = 3.5
;; under X = 2 and a paired effect of 2 between X = 0 and X = 2, despite
;; uncertainty in the confounder Z. Existing posterior weights are retained.

{:treated-y (cf/expectation named-worlds #(get-in % [:worlds :treated :y]))
 :paired-effect (cf/expectation named-worlds
                                #(- (get-in % [:worlds :treated :y])
                                    (get-in % [:worlds :untreated :y])))}

;; See the [causal guide](https://github.com/replikativ/foerster/blob/main/doc/causal.md)
;; for nested worlds, conditioning on hypothetical events, identification
;; from graphs or observation maps, attribution and actual-cause queries.

;; ## References
;;
;; - J. Pearl. *Causality: Models, Reasoning, and Inference.* 2nd ed.,
;;   Cambridge University Press, 2009 — interventions and the three-step
;;   counterfactual (abduction, action, prediction).
