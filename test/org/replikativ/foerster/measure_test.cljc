(ns org.replikativ.foerster.measure-test
  "Summaries of a weighted measure respect its weights."
  (:require #?(:clj [clojure.test :refer [deftest is testing]]
               :cljs [cljs.test :refer-macros [deftest is testing]])
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]))

(deftest multinomial-resampling-preserves-categorical-draws
  (doseq [weights [[1.0] [0.0 0.25 0.0 0.75] [0.1 0.2 0.3 0.3999999999999999]]
          n [0 1 49 200]]
    (let [draw (fn [f]
                 (random/with-stream* 42 :resample
                   #(let [indices (f)] [indices (random/uniform01)])))]
      (is (= (draw #(vec (repeatedly n (fn [] (m/sample-categorical weights)))))
             (draw #(m/multinomial-resample weights n)))))))

(deftest quantiles-follow-the-weights
  (let [q (m/weighted-quantiles [3.0 1.0 2.0 4.0] [0.1 0.1 0.7 0.1])]
    (is (= [1.0 1.0 2.0 2.0 3.0 4.0 4.0] (mapv q [0.0 0.05 0.15 0.5 0.85 0.975 1.0]))))
  (testing "equal weights give the order statistics"
    (let [q (m/weighted-quantiles (vec (range 1000)) (vec (repeat 1000 0.001)))]
      (is (= [24 499 974] (mapv q [0.025 0.5 0.975]))))))

(deftest measure-stats-are-weighted
  ;; value 0 carries 90 % of the weight
  (let [measure (m/empirical [[(m/sample-particle 0.0 {}) (Math/log 9.0)]
                              [(m/sample-particle 10.0 {}) 0.0]])
        stats (m/measure-stats measure m/get-value)]
    (is (< (Math/abs (- 1.0 (:mean stats))) 1e-12))
    (is (= {:p025 0.0 :p50 0.0 :p975 10.0} (:quantiles stats)))))

(deftest resampling-schemes-are-unbiased
  ;; E[copies of i] = n·wᵢ for every scheme
  (let [weights [0.5 0.3 0.15 0.05]
        n 10
        reps 3000]
    (doseq [scheme [:systematic :stratified :residual :multinomial]]
      (testing (name scheme)
        (let [counts (reduce (fn [acc indices] (reduce #(update %1 %2 inc) acc indices))
                             (vec (repeat 4 0))
                             (repeatedly reps #(m/resample scheme weights n)))]
          (doseq [[i w] (map-indexed vector weights)]
            (is (< (Math/abs (- (/ (nth counts i) (* reps n)) w)) 0.01)
                (str (name scheme) " " i))))))
    (testing "residual keeps the whole part of n·wᵢ"
      (is (every? (fn [indices] (>= (count (filter zero? indices)) 5))
                  (repeatedly 50 #(m/resample :residual weights n)))))))

(deftest resampling-edge-cases
  (testing "weights summing just below 1 by rounding"
    (let [ws (mapv #(* % (- 1.0 1e-7)) [0.25 0.25 0.25 0.25])]
      (doseq [scheme [:systematic :stratified :residual :multinomial]]
        (is (= 8 (count (m/resample scheme ws 8))) (name scheme)))))
  (testing "residual keeps every particle when the weights are equal"
    (doseq [n [49 98]]
      (is (= (range n) (sort (m/resample :residual (vec (repeat n (/ 1.0 n))) n))))))
  (testing "a single particle"
    (is (= [0 0 0] (m/resample :systematic [1.0] 3)))))
