(ns org.replikativ.foerster.tempering-test
  "Tempered SMC against exact answers: a conjugate posterior and its
  evidence, a bimodal posterior no Markov chain crosses, the waste-free
  variant, and seeded reproducibility."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.tempering :as tp]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.engine.executor :as executor]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

;; μ ~ N(0,1), y_i ~ N(μ,1): μ | y ~ N(Σy/(n+1), 1/(n+1)), and the evidence
;; in closed form
(def ^:private ys
  (let [r (java.util.Random. 3)] (vec (repeatedly 50 #(+ 0.7 (.nextGaussian r))))))

(def ^:private truth [(/ (reduce + ys) 51.0) (Math/sqrt (/ 1.0 51.0))])

(def ^:private log-evidence
  (let [n (count ys) s (reduce + ys) ss (reduce + (map #(* % %) ys))]
    (- (* -0.5 n (Math/log (* 2 Math/PI)))
       (* 0.5 (Math/log (+ 1 n)))
       (* 0.5 (- ss (/ (* s s) (+ 1 n)))))))

(defn- conjugate []
  (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
          (loop [i 0]
            (when (< i (count ys))
              (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
              (recur (inc i))))
          mu)))

(deftest a-conjugate-posterior-and-its-evidence
  (doseq [[label opts] [["tempered" {}] ["waste-free" {:waste-free 10}]]]
    (testing label
      (let [measure (b/run-infer 3 #(tp/tempered (conjugate) 400 opts))
            [mu sd] (b/w-mean-sd identity (b/weighted-values measure))]
        (is (< (Math/abs (- mu (first truth))) 0.05) (str "mean " mu))
        (is (< (Math/abs (- sd (second truth))) 0.03) (str "sd " sd))
        (is (< (Math/abs (- (m/log-marginal measure) log-evidence)) 0.4)
            (str "log evidence " (m/log-marginal measure) " vs " log-evidence))
        (is (= 1.0 (peek (:temperatures measure))))
        (is (pos? (:accepted (:rejuvenation measure))))))))

;; x ~ N(0, 2), 4 ~ N(x², 0.5): modes at ±2, equal mass
(defn- bimodal []
  (spin (let [x (sample (dist/normal 0.0 2.0) :id :x)]
          (observe (dist/normal (* x x) 0.5) 4.0 :id :y)
          x)))

(def ^:private bimodal-log-evidence
  (let [h 1e-4
        f (fn [x] (Math/exp (+ (dist/logpdf (dist/normal 0 2) x)
                               (dist/logpdf (dist/normal (* x x) 0.5) 4.0))))]
    (Math/log (* h (reduce + (map f (range -8 8 h)))))))

(deftest both-modes-of-a-bimodal-posterior
  (let [measure (b/run-infer 4 #(infer/tempered-infer (bimodal) 800))
        wv (b/weighted-values measure)]
    (is (< (Math/abs (- 0.5 (b/w-mean #(if (pos? %) 1.0 0.0) wv))) 0.1) "both modes, equal mass")
    (is (< (Math/abs (- 1.98 (b/w-mean #(Math/abs %) wv))) 0.05))
    (is (< (Math/abs (- (m/log-marginal measure) bimodal-log-evidence)) 0.3))))

(deftest seeded-runs-reproduce
  (let [run (fn [] (let [exec (executor/thread-pool-executor {:threads 4})]
                     (try (b/weighted-values (b/run-infer 5 #(tp/tempered (bimodal) 100 {:executor exec})))
                          (finally (.close ^java.lang.AutoCloseable exec)))))]
    (is (= (run) (run)))))

(deftest options-are-checked
  (let [root (context/create-execution-context)
        model (binding [ec/*execution-context* root] (bimodal))
        refusal (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:type (ex-data e)))))]
    (is (= ::tp/invalid-waste-free (refusal #(tp/tempered model 10 {:waste-free 3}))))
    (is (= ::infer/invalid-world-policy (refusal #(infer/tempered-infer model 10 {:world-policy :fork}))))
    (is (= ::infer/unknown-options (refusal #(infer/tempered-infer model 10 {:particles 3}))))))
