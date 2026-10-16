(ns org.replikativ.foerster.compiled-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.compiled :as c]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]))

(defn- sigmoid [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))

(def ^:private logistic
  (a/model [xs ns ys]
           (let [a (sample (dist/normal 0.0 1.0) :id :a)
                 b (sample (dist/normal 0.0 1.0) :id :b)]
             (loop [i 0]
               (when (< i (count xs))
                 (observe (dist/binomial (nth ns i) (/ 1.0 (+ 1.0 (Math/exp (- (+ a (* b (nth xs i))))))))
                          (nth ys i) :id [:putt i])
                 (recur (inc i))))
             {:a a :b b})))

(def ^:private data [[2.0 3.0 4.0 5.0] [100 90 80 70] [80 60 40 25]])

(deftest the-compiled-density-is-the-models-log-joint
  (let [{:keys [addresses log-density]} (apply c/compile-model logistic data)]
    (is (= [:a :b] addresses))
    (doseq [[av bv] [[0.3 -0.2] [1.1 -0.6] [-0.5 0.4]]]
      (let [{:keys [weight]} (b/run-infer 1 #(gfi/assess (apply logistic data) {:a av :b bv}))]
        (is (< (Math/abs (- weight (log-density (double-array [av bv])))) 1e-9))))))

(def ^:private normal-mean
  (a/model [ys]
           (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
             (loop [i 0]
               (when (< i (count ys))
                 (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
                 (recur (inc i))))
             mu)))

(deftest rmh-on-the-compiled-density-finds-the-posterior
  (let [ys [0.8 1.4 0.3 1.9 1.1 0.6 1.5 0.9]
        exact (/ (reduce + ys) (inc (count ys)))
        measure (b/run-infer 2 #(c/rmh (c/compile-model normal-mean ys)
                                       {:iterations 20000 :burn 1000 :step-size 0.6 :chains 2}))
        mus (map m/get-value (m/get-contexts measure))]
    (is (= 38000 (count mus)) "every state after :burn, per chain")
    (is (= [19000 19000] (:chain-lengths measure)))
    (is (< (Math/abs (- exact (/ (reduce + mus) (count mus)))) 0.03))))

(deftest infer-rmh-compiled
  (let [ys [0.8 1.4 0.3 1.9 1.1 0.6 1.5 0.9]
        exact (/ (reduce + ys) (inc (count ys)))
        run #(b/run-infer 3 (fn [] (infer/infer (normal-mean ys) {:method :rmh :compiled? true :iterations 8000
                                                                  :burn 500 :chains 2 :step-size 0.6 :seed [1 2]})))
        measure (run)
        mus (map m/get-value (m/get-contexts measure))]
    (is (= [7500 7500] (:chain-lengths measure)))
    (is (< (Math/abs (- exact (/ (reduce + mus) (count mus)))) 0.05))
    (is (= mus (map m/get-value (m/get-contexts (run)))) "a seeded run repeats")
    (is (= (first mus) (m/site-value (first (m/get-contexts measure)) :mu)))))

(deftest a-model-whose-structure-depends-on-a-value-is-refused
  (let [mixture (a/model [y]
                         (let [z (sample (dist/flip 0.5) :id :z)]
                           (if z
                             (observe (dist/normal 1.0 1.0) y :id :a)
                             (observe (dist/normal -1.0 1.0) y :id :b))))]
    (is (= #{:value-dependent-structure :unknown-test :not-continuous}
           (set (map :reason (:refusals (ex-data (try (c/compile-model mixture 0.3)
                                                      (catch clojure.lang.ExceptionInfo e e))))))))))
