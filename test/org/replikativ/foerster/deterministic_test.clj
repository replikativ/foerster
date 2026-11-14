(ns org.replikativ.foerster.deterministic-test
  "A deterministic site records a computed value in every particle's trace,
  without weight: its posterior is the image of the choices' posterior."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :refer [run-infer]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [deterministic observe sample]]
            [org.replikativ.foerster.moves.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]))

;; μ ~ N(0,1), y = 1 ~ N(μ,1): μ | y ~ N(1/2, 1/2), so 2μ has mean 1.
(defn- model []
  (task
   (let [mu (sample (dist/normal 0.0 1.0) :id :mu)
         twice (deterministic (* 2.0 mu) :id :twice)]
     (observe (dist/normal mu 1.0) 1.0 :id :y)
     twice)))

(defn- site-mean [measure address]
  (:mean (m/measure-stats measure #(m/site-value % address))))

(deftest deterministic-values-follow-the-posterior
  (doseq [[label make] [["importance sampling" #(infer/importance-sampling (fn [] (model)) 3000)]
                        ["SMC" #(infer/smc-infer (fn [] (model)) 3000)]
                        ["particle Gibbs" #(infer/pgibbs-infer (fn [] (model)) 50 40)]
                        ["MH" #(infer/kernel-infer (fn [] (model))
                                                   (k/random-walk-mh-kernel 600 {:step-size 0.8 :samples :all :burn 100})
                                                   4)]]]
    (testing label
      (let [measure (run-infer 4 make)]
        (is (< (Math/abs (- 1.0 (site-mean measure :twice))) 0.12) label)
        (is (every? (fn [[p _]] (= (m/site-value p :twice) (* 2.0 (m/site-value p :mu))))
                    (m/get-particles measure))
            "the recorded value is the one computed")))))

(deftest a-forward-run-returns-the-value
  (let [world (v2/world)]
    (is (number? (binding [v2/*world* world] (v2/value (model) 10000 nil))))))
