(ns org.replikativ.foerster.core-reexport-test
  "The modelling primitives work when required from foerster.core and spin
  from spindel.core, the one-namespace-per-library form the docs show."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer :refer [sample observe factor]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.core :as sp]))

(defn- dice
  "Two dice, a hard constraint (the sum is 7) and a soft one (a noisy reading
  of 7): the first die's posterior is uniform over 1..6."
  []
  (sp/spin
   (let [a (sample (dist/uniform-discrete 1 7) :id :a)
         b (sample (dist/uniform-discrete 1 7) :id :b)]
     (factor (if (= 7 (+ a b)) 0.0 ##-Inf))
     (observe (dist/normal (+ a b) 1.0) 7.0 :id :reading)
     a)))

(deftest re-exported-effects-are-effects
  (let [measure (b/run-infer 5 #(infer/infer (dice) {:method :enumerate}))
        ps (filter (comp #(> % ##-Inf) second) (m/get-particles measure))
        ws (m/normalize-log-weights (mapv second ps))
        mean (reduce + (map (fn [[p _] w] (* w (m/get-value p))) ps ws))]
    (is (= 6 (count ps)) "the six pairs that sum to 7")
    (is (< (Math/abs (- mean 3.5)) 1e-12) (str mean))
    ;; every pair sums to 7, so the reading's weight is N(7; 7, 1) on each
    (is (< (Math/abs (- (m/log-marginal measure)
                        (+ (Math/log (/ 6.0 36)) (- (* 0.5 (Math/log (* 2 Math/PI)))))))
           1e-12))))
