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

(deftest a-trajectory-keeps-what-is-recorded
  (let [measure (b/run-infer 53 #(smc/smc (steer/model {:init {:heads 0 :cache (range 100)}
                                                        :step (fn [s] (spin (update s :heads + (if (< (random/uniform01) 0.5) 1 0))))
                                                        :done? (constantly false)
                                                        :max-steps 3
                                                        :record :heads})
                                          20 {}))]
    (is (every? #(every? number? (:states %)) (learn/trajectories measure)))))

(defn- noisy-mean
  "mu ~ N(0, 1), y ~ N(mu, σ) with σ = exp(:log-sigma) a parameter."
  [{:keys [log-sigma]}]
  (spin
   (let [mu (org.replikativ.foerster.effects/sample (org.replikativ.foerster.dist/normal 0.0 1.0) :id :mu)
         s (Math/exp log-sigma)]
     (loop [[y & more] [0.5 2.1 -0.3 1.7 1.2 0.9 2.6 -0.8] i 0]
       (when y
         (org.replikativ.foerster.effects/observe (org.replikativ.foerster.dist/normal mu s) y :id [:y i])
         (recur more (inc i))))
     mu)))

(deftest maximum-marginal-likelihood
  ;; the exact evidence: y ~ N(0, σ²I + 11ᵀ); its maximizer by a fine grid
  (let [ys [0.5 2.1 -0.3 1.7 1.2 0.9 2.6 -0.8]
        n (count ys) sy (reduce + ys) syy (reduce + (map * ys ys))
        ;; Σ = σ²I + J: det = σ^(2(n-1))·(σ² + n), quadratic form via Sherman–Morrison
        log-z (fn [s] (let [s2 (* s s)]
                        (- (* -0.5 n (Math/log (* 2 Math/PI)))
                           (* 0.5 (+ (* 2 (dec n) (Math/log s)) (Math/log (+ s2 n))))
                           (* 0.5 (- (/ syy s2) (/ (* sy sy) (* s2 (+ s2 n))))))))
        best (apply max-key log-z (range 0.3 4.0 0.0005))
        fit (b/run-infer 54 #(learn/maximize-evidence noisy-mean {:log-sigma (Math/log 0.4)}
                                                      {:method :smc :particles 300}
                                                      {:steps 120 :rate 0.05}))]
    (is (< (Math/abs (- (Math/exp (:log-sigma (:params fit))) best)) 0.06)
        (str (Math/exp (:log-sigma (:params fit))) " vs " best))))
