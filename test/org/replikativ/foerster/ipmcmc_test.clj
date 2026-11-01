(ns org.replikativ.foerster.ipmcmc-test
  "Interacting particle MCMC against exact enumeration."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- sticky-hmm
  "x₁ ~ flip(½), x_t stays x_{t−1} with probability 0.8; y_t ~ flip(x_t ?
  0.9 : 0.1), observed 1 0 1 1."
  []
  (spin
   (loop [t 0 prev nil xs []]
     (if (= t 4)
       xs
       (let [x (sample (dist/flip (cond (nil? prev) 0.5 prev 0.8 :else 0.2)) :id [:x t])]
         (observe (dist/flip (if x 0.9 0.1)) (not= t 1) :id [:y t])
         (recur (inc t) x (conj xs x)))))))

(defn- law [measure]
  (let [ps (m/get-particles measure)]
    (reduce (fn [acc [[s _] w]] (update acc (m/get-value s) (fnil + 0.0) w))
            {} (map vector ps (m/normalize-log-weights (mapv second ps))))))

(defn- total-variation [p q]
  (* 0.5 (reduce + (map #(Math/abs (- (get p % 0.0) (get q % 0.0)))
                        (distinct (concat (keys p) (keys q)))))))

(defn- marginal [law t]
  (reduce + (map (fn [[xs w]] (if (nth xs t) w 0.0)) law)))

(deftest every-node-runs-the-same-smc
  ;; iPMCMC's Gibbs draw of the conditional nodes compares the nodes'
  ;; evidence estimates, so each conditional node must be the conditional of
  ;; the free nodes' SMC (Rainforth et al. 2016). Free nodes that resampled
  ;; only below ESS N/2 — never, with N = 2 — while conditional nodes
  ;; resampled at every barrier left the chain off its target.
  ;;
  ;; N = 2 particles, 4 nodes, 2 conditional, 2000 iterations. Over 120 seeds
  ;; of a reference implementation of the algorithm, the total variation to
  ;; the exact posterior was 0.082–0.151 with mismatched nodes and
  ;; 0.006–0.049 with matched ones, and the error in P(x₂) +0.050…+0.119
  ;; against −0.026…+0.035. This code gave 0.083 and 0.108 with mismatched
  ;; nodes (P(x₂) +0.044, +0.077) and 0.007–0.043 over 15 seeds with matched
  ;; ones (P(x₂) −0.040…+0.026). The review's 3-step HMM separates the two
  ;; only beyond ~20000 iterations.
  (let [exact (law (b/run-infer 1 #(infer/infer (sticky-hmm) {:method :enumerate})))
        chain (law (b/run-infer 3 #(infer/ipmcmc-infer (sticky-hmm) 2 2000
                                                       {:num-nodes 4 :num-csmc-nodes 2 :seed 3})))]
    (is (< (Math/abs (- (marginal exact 1) 0.5248918890412404)) 1e-12) "the reference")
    (is (< (total-variation chain exact) 0.065) (str "TV " (total-variation chain exact)))
    (is (< (Math/abs (- (marginal chain 1) (marginal exact 1))) 0.045)
        (str "P(x₂) " (marginal chain 1) " against " (marginal exact 1)))))

(deftest ipmcmc-always-resamples
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"resample-threshold"
                        (infer/ipmcmc-infer nil 2 1 {:resample-threshold 0.5}))))
