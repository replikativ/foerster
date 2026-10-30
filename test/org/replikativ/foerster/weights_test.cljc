(ns org.replikativ.foerster.weights-test
  (:require #?(:clj [clojure.test :refer [deftest is]]
               :cljs [cljs.test :refer-macros [deftest is]])
            [org.replikativ.foerster.weights :as w]))

(defn- near? [a b] (< (Math/abs (- a b)) 1e-10))

(deftest corrections-and-resampling-preserve-unnormalized-mass
  (let [initial (w/from-log [(Math/log 2) (Math/log 4)])
        corrected (w/reweight initial [(Math/log 3) (Math/log 2)])
        resampled (w/resampled corrected 7)]
    (is (near? (Math/exp (:log-mass initial)) 3.0))
    (is (near? (Math/exp (:log-mass corrected)) 7.0))
    (is (every? true? (map near? (w/selection corrected) [(/ 3.0 7) (/ 4.0 7)])))
    (is (= (:log-mass corrected) (:log-mass resampled)))
    (is (every? #(near? % (/ 1.0 7)) (w/selection resampled)))))

(deftest finite-importance-law-including-normalizer-scale
  ;; Enumerate all iid two-particle draws under q=[1/4,3/4], gamma=[2,3].
  ;; Check unnormalized state mass, not a biased normalized posterior mean.
  (let [q [0.25 0.75] gamma [2.0 3.0]
        expected (reduce (fn [acc [a b]]
                           (let [xs [a b]
                                 probability (* (q a) (q b))
                                 state (w/from-log (mapv #(Math/log (/ (gamma %) (q %))) xs))
                                 mass (Math/exp (:log-mass state))]
                             (reduce (fn [r [x alpha]] (update r x + (* probability mass alpha)))
                                     acc (map vector xs (w/selection state)))))
                         [0.0 0.0] (for [a (range 2) b (range 2)] [a b]))]
    (is (every? true? (map near? gamma expected)))))

(deftest impossible-populations-never-turn-into-posteriors
  (let [state (w/reweight (w/uniform 2) [##-Inf ##-Inf])]
    (is (= :zero-mass (:status state)))
    (is (= ##-Inf (:log-mass state)))
    (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs js/Error) (w/selection state)))
    (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs js/Error) (w/resampled state 2)))
    (is (= state (w/reweight state [1.0 2.0])))))

(deftest invalid-corrections-fail-before-truncating-a-population
  (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs js/Error) (w/reweight (w/uniform 2) [1.0])))
  (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs js/Error) (w/from-log [##NaN 0.0])))
  (is (thrown? #?(:clj clojure.lang.ExceptionInfo :cljs js/Error) (w/from-log [##Inf 0.0]))))
