(ns org.replikativ.foerster.process-test
  "mem and the Chinese restaurant process, with their state in the
  particle's world."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.process :as process]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- seat-three [alpha]
  (spin
   (let [a (await (process/crp-draw :r alpha :id [:z 0]))
         b (await (process/crp-draw :r alpha :id [:z 1]))
         c (await (process/crp-draw :r alpha :id [:z 2]))]
     [a b c])))

(defn- partition-of [[a b c]]
  (cond (= a b c) :together
        (and (not= a b) (not= a c) (not= b c)) :apart
        :else :pair))

(deftest crp-partitions-are-exact
  ;; Ewens: P(together) = 1/(1+α)·2/(2+α)·… ; for α = 1: 1/3, 1/6, and the
  ;; three pair+single partitions 1/6 each
  (let [measure (b/run-infer 101 #(infer/infer (seat-three 1.0) {:method :enumerate}))
        ws (m/normalize-log-weights (mapv second (m/get-particles measure)))
        p (reduce (fn [acc [[s _] w]] (update acc (partition-of (m/get-value s)) (fnil + 0.0) w))
                  {} (map vector (m/get-particles measure) ws))]
    (is (< (Math/abs (- (:together p) (/ 1.0 3))) 1e-12))
    (is (< (Math/abs (- (:apart p) (/ 1.0 6))) 1e-12))
    (is (< (Math/abs (- (:pair p) 0.5)) 1e-12))
    (is (< (Math/abs (m/log-marginal measure)) 1e-12) "no evidence: the probabilities sum to one")))

(defn- memo-model []
  (spin
   (let [mean-of (process/mem (fn [k] (spin (sample (dist/normal 0.0 1.0) :id [:mean k]))))
         a (await (mean-of 0))
         b (await (mean-of 0))
         c (await (mean-of 1))]
     (observe (dist/normal a 0.5) 1.0 :id :y)
     [a b c])))

(deftest mem-computes-each-argument-once-per-world
  (doseq [opts [{:method :smc :particles 200}
                {:method :mh :iterations 300 :chains 2 :burn 50}]]
    (testing (name (:method opts))
      (let [measure (b/run-infer 102 #(infer/infer (memo-model) opts))
            values (map (comp m/get-value first) (m/get-particles measure))]
        (is (every? (fn [[a b _]] (= a b)) values) "a repeated call returns the first value")
        (is (some (fn [[a _ c]] (not= a c)) values) "another argument draws its own")
        (is (< 1 (count (distinct (map first values)))) "particles differ")))))

(defn- inner []
  ;; z uniform on {0, 1, 2}, a flip of bias z/2 came up heads: P(z) ∝ z
  (spin
   (let [z (sample (dist/uniform-discrete 0 3) :id :z)]
     (observe (dist/flip (/ z 2.0)) true :id :heads)
     z)))

(deftest nested-inference-is-exact-under-enumeration
  ;; the outer agent samples the inner posterior {1: 1/3, 2: 2/3}, then sees
  ;; a signal that favours 2: P(x = 2) = (2/3·0.9) / (2/3·0.9 + 1/3·0.1)
  (let [outer #(spin
                (let [q (await (infer/conditional (inner) {:method :enumerate}))
                      x (sample q :id :x)]
                  (observe (dist/flip (if (= x 2) 0.9 0.1)) true :id :signal)
                  x))
        measure (b/run-infer 103 #(infer/infer (outer) {:method :enumerate}))
        ws (m/normalize-log-weights (mapv second (m/get-particles measure)))
        p2 (reduce + (map (fn [[[s _] w]] (if (= 2 (m/get-value s)) w 0.0))
                          (map vector (m/get-particles measure) ws)))
        exact (/ (* 2/3 0.9) (+ (* 2/3 0.9) (* 1/3 0.1)))]
    (is (< (Math/abs (- p2 exact)) 1e-12) (str p2 " vs " exact))))
