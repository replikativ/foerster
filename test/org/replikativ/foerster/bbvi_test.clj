(ns org.replikativ.foerster.bbvi-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(deftest bbvi-learns-flip-and-exponential-sites
  ;; the default AdaGrad rates are vectors; one-parameter guides take the first
  (let [model (fn [] (spin (let [x (sample (dist/flip 0.5) :id :x)
                                 r (sample (dist/exponential 1.0) :id :r)]
                             (observe (dist/flip (if x 0.9 0.1)) true :id :y)
                             (observe (dist/poisson r) 3 :id :n)
                             (if x 1.0 0.0))))
        measure (b/run-infer 23 #(infer/infer (model) {:method :bbvi :particles 50 :iterations 60}))
        {:keys [mean]} (d/summary measure identity)]
    (is (< 0.6 mean) (str "P(x) " mean))))
