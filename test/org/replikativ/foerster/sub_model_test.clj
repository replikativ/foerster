(ns org.replikativ.foerster.sub-model-test
  "Models that await sub-models: the sub-model's sites are the particle's."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.runtime.async :refer [await-op]]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]))

(defn- prior []
  (task (sample (dist/normal 0.0 1.0) :id :mu)))

(defn- model []
  (task
   (let [mu (await-op (prior))]
     (observe (dist/normal mu 1.0) 1.0 :id :y)
     mu)))

(defn- posterior-mean [run]
  (let [world (v2/world)
        result (binding [v2/*world* world]
                 (deref (run) 60000 ::timed-out))]
    (is (not= ::timed-out result) "inference finished")
    (when-not (= ::timed-out result)
      (:mean (infer/query result identity)))))

;; mu ~ N(0,1), y = 1 ~ N(mu,1): the posterior is N(1/2, 1/2).
(deftest sub-model-sites-belong-to-the-particle
  (random/set-seed! 7)
  (doseq [[label run]
          [["importance sampling" #(infer/importance-sampling (fn [] (model)) 2000)]
           ["SMC" #(infer/smc-infer (fn [] (model)) 2000)]
           ["SMC in canonical worlds" #(infer/smc-infer (fn [] (model)) 500 {:world-policy :fork})]]]
    (testing label
      (let [mean (posterior-mean run)]
        (is (some? mean))
        (when mean
          (is (< (Math/abs (- mean 0.5)) 0.1) (str label ": " mean)))))))
