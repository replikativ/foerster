;; # Interventions and Counterfactuals

;; A probabilistic program is also a causal model: each `sample` site is a
;; variable computed from the ones before it. That lets us ask three kinds
;; of question (Pearl's ladder): what we expect having **seen** something,
;; what would happen if we **did** something, and what **would have**
;; happened had things been different.

(ns foerster.counterfactuals
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.mechanism :as mech]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as trace]
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
;; exogenous noise u (`foerster.mechanism`). `counterfactual`:
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
        ws (m/normalize-log-weights (mapv :weight pairs))]
    (reduce + (map (fn [pair w] (if (:y (:counterfactual pair)) 0.0 w)) pairs ws))))

;; A discrete site has many noise values for one outcome; abduction draws
;; one from its posterior. Which convention a discrete mechanism uses (here
;; the inverse CDF, x = F⁻¹(u)) changes counterfactual answers and cannot be
;; told from data, so foerster fixes and documents it
;; (`foerster.mechanism`) rather than guessing.
;;
;; A site that exists only in the counterfactual world — reached through a
;; branch the factual run did not take — has no factual noise; it is drawn
;; fresh and reported in `:unaligned`.
