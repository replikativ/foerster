(ns org.replikativ.foerster.learn-test
  "Training data from a steered SMC run: the coin process of steer-test,
  whose tilted law is Binomial(5, e^λ/(1+e^λ))."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.learn :as learn]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.steer :as steer]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private lambda 0.8)

(defn- process []
  (steer/model {:init 0
                :step (fn [heads] (spin (+ heads (if (< (random/uniform01) 0.5) 1 0))))
                :done? (constantly false)
                :max-steps 5
                ;; no twist: the weights carry the whole tilt until the end
                :reward (fn [heads] (* lambda heads))}))

(deftest trajectories-carry-states-rewards-and-weights
  (let [measure (b/run-infer 51 #(smc/smc (process) 200 {:resample-threshold 0.0}))
        ts (learn/trajectories measure)]
    (is (= 200 (count ts)))
    (is (every? #(= 5 (count (:states %))) ts))
    (is (every? #(= (:reward %) (* lambda (peek (:states %)))) ts))
    (is (< (Math/abs (- 1.0 (reduce + (map :weight ts)))) 1e-9))
    (is (every? (fn [{:keys [states]}] (every? #{0 1} (map - (rest states) states))) ts)
        "one coin per step")))

(deftest draws-follow-the-target
  ;; unweighted draws from the tilted law: mean heads 5·e^λ/(1+e^λ)
  (let [measure (b/run-infer 52 #(smc/smc (process) 2000 {:resample-threshold 0.0}))
        ds (learn/draws measure 4000)
        mean (/ (reduce + (map (comp peek :states) ds)) (count ds))
        exact (* 5 (/ (Math/exp lambda) (+ 1 (Math/exp lambda))))]
    (is (= 4000 (count ds)))
    (is (< (Math/abs (- mean exact)) 0.1) (str mean " vs " exact))))
