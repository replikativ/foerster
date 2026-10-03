(ns org.replikativ.foerster.conditional-smc-test
  "Particle Gibbs keeps the caller's policy for the retained particle."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- model []
  ;; x ~ Bernoulli(½), y ~ Flip(x ? 0.9 : 0.1); conditioned on y = true from
  ;; outside, P(x = 1) = 0.9
  (spin
   (let [x (sample (dist/bernoulli 0.5) :id :x)]
     (sample (dist/flip (if (= 1 x) 0.9 0.1)) :id :y)
     x)))

(deftest the-retained-particle-is-conditioned-like-the-others
  (let [measure (b/run-infer 21 #(infer/infer (model) {:method :pgibbs :particles 2 :iterations 4000
                                                       :policy (itrace/policy {:constraints {:y true}})}))
        {:keys [mean]} (d/summary measure identity)]
    (is (< (Math/abs (- mean 0.9)) 0.03) (str "P(x = 1) " mean))))
