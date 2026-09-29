(ns org.replikativ.foerster.measure-test
  "Summaries of a weighted measure respect its weights."
  (:require #?(:clj [clojure.test :refer [deftest is testing]]
               :cljs [cljs.test :refer-macros [deftest is testing]])
            [org.replikativ.foerster.measure :as m]))

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
