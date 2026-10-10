(ns org.replikativ.foerster.conditional-smc-test
  "Particle Gibbs keeps the caller's policy for the retained particle."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
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

(defn- guided-branch []
  ;; k ~ Bernoulli(½); with k = 1 a site b ~ Flip(½) that a replay draws
  ;; from its guide Flip(0.9): the guide must not shift the posterior
  (spin
   (let [k (sample (dist/bernoulli 0.5) :id :k)]
     [k (when (= 1 k) (sample (dist/flip 0.5) :id :b :proposal (dist/flip 0.9)))])))

(deftest mh-scores-a-guided-site-it-drops-under-its-guide
  (let [measure (b/run-infer 22 #(infer/infer (guided-branch) {:method :mh :iterations 6000 :chains 2 :burn 500}))
        vs (map (comp m/get-value first) (m/get-particles measure))
        born (filter #(= 1 (first %)) vs)
        pk (/ (count born) (double (count vs)))
        pb (/ (count (filter second born)) (double (count born)))]
    (is (< (Math/abs (- pk 0.5)) 0.04) (str "P(k = 1) " pk))
    (is (< (Math/abs (- pb 0.5)) 0.04) (str "P(b | k = 1) " pb))))

(deftest particle-gibbs-with-moves-recovers-the-posterior
  ;; the retained particle moves too (`smc/retained-stages`); its exactness is
  ;; checked by enumeration in `sweep-enumeration-test`, this is the API path
  (let [measure (b/run-infer 23 #(infer/infer (model) {:method :pgibbs :particles 3 :iterations 3000
                                                       :anchors :all :rejuvenate {:moves 1}
                                                       :policy (itrace/policy {:constraints {:y true}})}))
        {:keys [mean]} (d/summary measure identity)]
    (is (< (Math/abs (- mean 0.9)) 0.03) (str "P(x = 1) " mean))))
