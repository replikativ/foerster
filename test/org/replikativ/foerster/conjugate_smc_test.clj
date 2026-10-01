(ns org.replikativ.foerster.conjugate-smc-test
  "A conjugate parameter inside an SMC program: never sampled until the end,
  so the evidence is exact and the parameter's posterior is the closed form."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.conjugate :as conj]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private ys [1 0 1 1 1 0 1 1 1 0 1 1])

(defn- model []
  (spin (loop [c (conj/beta-bernoulli 1 1) i 0]
          (if (= i (count ys))
            (sample (conj/posterior c) :id :p)
            (do (observe (conj/predictive c) (nth ys i) :id [:y i])
                (recur (conj/update c (nth ys i)) (inc i)))))))

(deftest a-conjugate-parameter-in-smc
  (let [k (count (filter #{1} ys)) n (count ys)
        lbeta (fn [a b] (- (+ (dist/lgamma a) (dist/lgamma b)) (dist/lgamma (+ a b))))
        exact-z (- (lbeta (+ 1 k) (+ 1 (- n k))) (lbeta 1 1))
        measure (b/run-infer 81 #(smc/smc (model) 2000 {}))
        p (b/w-mean identity (b/weighted-values measure))]
    (is (< (Math/abs (- (m/log-marginal measure) exact-z)) 1e-9) "every particle carries the same posterior")
    (is (< (Math/abs (- p (/ (+ 1.0 k) (+ 2 n)))) 0.01) (str p))))
