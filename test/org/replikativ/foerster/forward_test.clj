(ns org.replikativ.foerster.forward-test
  "A model run outside inference simulates forward: every site draws afresh,
  and sites are addressed as under inference."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [intervene! observe sample]]
            [org.replikativ.spindel.engine.addressing :refer [with-scope]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- in-world [world make]
  (binding [ec/*execution-context* world]
    (deref (make) 10000 ::timed-out)))

(deftest a-named-site-draws-afresh-each-time-it-is-reached
  (let [world (context/create-execution-context)
        loop-draws (in-world world
                             #(spin (loop [i 0 xs []]
                                      (if (= i 5)
                                        xs
                                        (recur (inc i) (conj xs (sample (dist/normal 0.0 1.0) :id :x)))))))
        runs (repeatedly 3 (fn [] (in-world world #(spin (sample (dist/normal 0.0 1.0) :id :x)))))]
    (is (= 5 (count (distinct loop-draws))) "a loop over one named site")
    (is (= 3 (count (distinct runs))) "repeated runs in one world")))

(deftest interventions-address-sites-as-inference-does
  (let [world (context/create-execution-context)]
    (is (= [5.0 7.0]
           (in-world world
                     #(spin
                       (intervene! [:a :x] 5.0)
                       (intervene! :y 7.0)
                       [(with-scope [:a] (sample (dist/normal 0.0 1.0) :id :x))
                        (sample (dist/normal 0.0 1.0) :id :y)])))
        "a scoped name is its path; a top-level name is itself")))

(deftest observations-score-the-world
  (let [world (context/create-execution-context)]
    (in-world world #(spin (observe (dist/normal 0.0 1.0) 0.0) :done))
    (is (< (Math/abs (- (rtp/get-state world [:inference :log-weight])
                        (dist/logpdf (dist/normal 0.0 1.0) 0.0)))
           1e-12))))
