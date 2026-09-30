(ns org.replikativ.foerster.smcp3-test
  "SMCP3 move-reweight steps against exact answers: a random walk on a
  static parameter, a kernel that revises the previous state of a random
  walk (checked against the Kalman filter), and a data-driven independent
  proposal equal to the posterior, whose weights stay equal."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private ys
  (let [r (java.util.Random. 3)] (vec (repeatedly 30 #(+ 0.7 (.nextGaussian r))))))

(defn- static-model []
  (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
          (loop [i 0]
            (when (< i (count ys))
              (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
              (recur (inc i))))
          mu)))

(def ^:private truth [(/ (reduce + ys) (inc (count ys))) (Math/sqrt (/ 1.0 (inc (count ys))))])

(def ^:private log-evidence
  (let [n (count ys) s (reduce + ys) ss (reduce + (map #(* % %) ys))]
    (- (* -0.5 n (Math/log (* 2 Math/PI))) (* 0.5 (Math/log (+ 1 n))) (* 0.5 (- ss (/ (* s s) (+ 1 n)))))))

;; small against the posterior's spread: move-reweight with a symmetric
;; backward kernel multiplies the weights by π(x')/π(x) at every barrier, and
;; large steps make the evidence estimate heavy-tailed
(def ^:private step (dist/normal 0.0 0.05))

(def ^:private random-walk-kernel
  ;; K: μ' = μ + u, u ~ N(0, .05); back: u' = −u under the same law
  {:forward (fn [{mu :mu} _]
              (let [u (dist/draw step)]
                {:updates {:mu (+ mu u)} :log-q (dist/logpdf step u) :reverse (- u)}))
   :backward (fn [_ u'] (dist/logpdf step u'))})

(deftest move-reweight-keeps-the-posterior-and-the-evidence
  (let [measure (b/run-infer 31 #(smc/smc (static-model) 400
                                          {:anchors #{:mu} :smcp3 random-walk-kernel}))
        [mu sd] (b/w-mean-sd identity (b/weighted-values measure))]
    (is (< (Math/abs (- mu (first truth))) 0.05) (str mu " vs " truth))
    (is (< (Math/abs (- sd (second truth))) 0.04) (str sd))
    (is (< (Math/abs (- (m/log-marginal measure) log-evidence)) 0.3)
        (str (m/log-marginal measure) " vs " log-evidence))))

(defn- posterior-after
  "N(mean, sd) of μ given the observations recorded in `trace`."
  [trace]
  (let [ys' (keep (fn [[a e]] (when (and (vector? a) (= :y (first a))) (:value e)))
                  (:trace/entries trace))
        n (count ys')]
    (dist/normal (/ (reduce + ys') (inc n)) (Math/sqrt (/ 1.0 (inc n))))))

(def ^:private exact-kernel
  ;; K: an independent draw from the exact posterior given the data so far;
  ;; back: the old μ under the same law. Every SMCP3 increment is then 0,
  ;; and after resampling the particles are exact posterior draws.
  {:forward (fn [{mu :mu} trace]
              (let [q (posterior-after trace)
                    mu' (dist/draw q)]
                {:updates {:mu mu'} :log-q (dist/logpdf q mu') :reverse [q mu]}))
   :backward (fn [_ [q mu]] (dist/logpdf q mu))})

(deftest an-exact-data-driven-kernel
  (let [measure (b/run-infer 32 #(smc/smc (static-model) 400
                                          {:anchors #{:mu} :smcp3 exact-kernel :resample-threshold 1.0}))
        [mu sd] (b/w-mean-sd identity (b/weighted-values measure))]
    (is (< (Math/abs (- (m/log-marginal measure) log-evidence)) 0.3)
        (str (m/log-marginal measure) " vs " log-evidence))
    (is (< (Math/abs (- mu (first truth))) 0.05))
    (is (< (Math/abs (- sd (second truth))) 0.03))))

;; x_t = x_{t-1} + N(0,1), y_t ~ N(x_t, 1)
(def ^:private walk-ys [0.8 1.9 1.2 3.0 2.5 2.2 3.4])

(defn- random-walk []
  (spin (loop [t 0 x (sample (dist/normal 0.0 1.0) :id [:x 0])]
          (observe (dist/normal x 1.0) (nth walk-ys t) :id [:y t])
          (if (= t (dec (count walk-ys)))
            x
            (recur (inc t) (sample (dist/normal x 1.0) :id [:x (inc t)]))))))

(def ^:private kalman
  (loop [[y & more] walk-ys pm 0.0 pv 1.0 ll 0.0 out nil]
    (if-not y
      out
      (let [s (+ pv 1.0) k (/ pv s) mean (+ pm (* k (- y pm))) var (* (- 1.0 k) pv)
            ll (+ ll (dist/logpdf (dist/normal pm (Math/sqrt s)) y))]
        (recur more mean (+ var 1.0) ll [mean var ll])))))

(def ^:private revise-previous
  ;; K: at barrier t, move x_{t-1} (the past) by a symmetric random walk
  {:forward (fn [choices _]
              (let [t (dec (count (filter #(and (vector? %) (= :x (first %))) (keys choices))))]
                (if (pos? t)
                  (let [u (dist/draw step)]
                    {:updates {[:x (dec t)] (+ (get choices [:x (dec t)]) u)}
                     :log-q (dist/logpdf step u) :reverse (- u)})
                  {:updates {}})))
   :backward (fn [_ u'] (if (number? u') (dist/logpdf step u') 0.0))})

(deftest a-kernel-may-revise-the-past
  (let [measure (b/run-infer 33 #(smc/smc (random-walk) 2000
                                          {:anchors {:lag 2} :smcp3 revise-previous}))
        [mean var ll] kalman
        [mu sd] (b/w-mean-sd identity (b/weighted-values measure))]
    (is (< (Math/abs (- mu mean)) 0.08) (str mu " vs " mean))
    (is (< (Math/abs (- (* sd sd) var)) 0.08) (str (* sd sd) " vs " var))
    (is (< (Math/abs (- (m/log-marginal measure) ll)) 0.15) (str (m/log-marginal measure) " vs " ll))))

(deftest options-are-checked
  (let [root (context/create-execution-context)
        model (binding [ec/*execution-context* root] (static-model))
        refusal (fn [opts] (try ((smc/smc model 10 opts) (fn [_]) (fn [_])) nil
                                (catch clojure.lang.ExceptionInfo e (:type (ex-data e)))))]
    (is (= ::smc/smcp3-without-anchors (refusal {:smcp3 random-walk-kernel})))
    (is (= ::smc/smcp3-with-retained (refusal {:anchors :all :smcp3 random-walk-kernel :retained {}})))))
