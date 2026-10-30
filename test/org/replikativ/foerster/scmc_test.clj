(ns org.replikativ.foerster.scmc-test
  "GIGA against known combinations, and sequential core-set Monte Carlo on a
  conjugate regression against its exact posterior and evidence."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.scmc :as scmc]))

(defn- combine [w ls]
  (let [out (double-array (alength ^doubles (first ls)))]
    (doseq [[wj ^doubles l] (map vector w ls) i (range (alength out))]
      (aset out i (+ (aget out i) (* wj (aget l i)))))
    out))

(defn- rel-error [w ls ^doubles target]
  (let [a (combine w ls)]
    (/ (Math/sqrt (reduce + (map #(let [e (- %1 %2)] (* e e)) a target)))
       (Math/sqrt (reduce + (map #(* % %) target))))))

(deftest giga-recovers-a-sparse-combination
  ;; target = 3·e₀ + 2·e₂ over four orthogonal vectors of different norms
  (let [ls [(double-array [2 0 0 0]) (double-array [0 1 0 0]) (double-array [0 0 5 0]) (double-array [0 0 0 3])]
        target (double-array [6 0 10 0])
        w (scmc/giga ls target 4)]
    (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-9) w [3.0 0.0 2.0 0.0])))))

(deftest giga-is-sparse-and-improves-with-its-budget
  (let [rng (java.util.Random. 1)
        ls (vec (repeatedly 100 (fn [] (double-array (repeatedly 60 #(.nextGaussian rng))))))
        target (combine (repeat 100 1.0) ls)
        errors (for [m [5 20 60]]
                 (let [w (scmc/giga ls target m)]
                   (is (<= (count (filter pos? w)) m))
                   (is (every? #(>= % 0.0) w))
                   (rel-error w ls target)))]
    (is (apply > errors)))
  (testing "a zero target"
    (is (= [0.0 0.0] (scmc/giga [(double-array [1 0]) (double-array [0 1])] (double-array [0 0]) 2)))))

;; y = a + b·x + N(0, 0.5²), (a, b) ~ N(0, I): the posterior and the evidence
;; are Gaussian in closed form
(def ^:private sigma2 0.25)

(defn- regression [n seed]
  (let [rng (java.util.Random. seed)
        xs (vec (repeatedly n #(.nextGaussian rng)))]
    (mapv (fn [x] [x (+ 0.7 (* -1.3 x) (* 0.5 (.nextGaussian rng)))]) xs)))

(defn- exact [data]
  (let [n (count data) xs (map first data) ys (map second data)
        sxx (reduce + (map #(* % %) xs)) sx (reduce + xs)
        syy (reduce + (map #(* % %) ys)) sy (reduce + ys) sxy (reduce + (map (fn [[x y]] (* x y)) data))
        ;; precision I + XᵀX/σ², X = [1 x]
        p00 (+ 1.0 (/ n sigma2)) p01 (/ sx sigma2) p11 (+ 1.0 (/ sxx sigma2))
        det (- (* p00 p11) (* p01 p01))
        r0 (/ sy sigma2) r1 (/ sxy sigma2)]
    {:a (/ (- (* p11 r0) (* p01 r1)) det) :b (/ (- (* p00 r1) (* p01 r0)) det)
     :a-sd (Math/sqrt (/ p11 det)) :b-sd (Math/sqrt (/ p00 det))
     ;; Woodbury and the determinant lemma on N(y; 0, σ²I + XXᵀ)
     :log-z (- (* -0.5 n (Math/log (* 2 Math/PI sigma2)))
               (* 0.5 (Math/log det))
               (* 0.5 (- (/ syy sigma2) (/ (+ (* p11 r0 r0) (* -2.0 p01 r0 r1) (* p00 r1 r1)) det))))}))

(defn- model [data batch]
  {:prior-sample (fn [] [(dist/draw (dist/normal 0.0 1.0)) (dist/draw (dist/normal 0.0 1.0))])
   :prior-log-density (fn [[a b]] (- (* -0.5 (+ (* a a) (* b b))) (* 2.0 0.9189385332046727)))
   :log-lik (fn [[a b] [x y]] (let [r (- y a (* b x))]
                                (- (/ (* -0.5 r r) sigma2) (* 0.5 (Math/log sigma2)) 0.9189385332046727)))
   :batches (partition batch data)})

(defn- moments [qs j]
  (let [v (map #(nth % j) qs) n (count v) mu (/ (reduce + v) n)]
    [mu (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) v)) (dec n)))]))

(deftest scmc-on-a-conjugate-regression
  ;; 2000 points in 100 batches, at most 50 of them remembered
  (random/set-seed! 3)
  (let [data (regression 2000 2)
        {:keys [a b a-sd b-sd log-z]} (exact data)
        r (scmc/scmc (model data 20) {:particles 500 :memory 50 :seed [1 50]})
        [am asd] (moments (mapv (comp m/get-value first) (m/get-particles r)) 0)
        [bm bsd] (moments (mapv (comp m/get-value first) (m/get-particles r)) 1)]
    (is (< (Math/abs (/ (- am a) a-sd)) 0.15))
    (is (< (Math/abs (/ (- bm b) b-sd)) 0.15))
    (is (< (Math/abs (- 1.0 (/ asd a-sd))) 0.1))
    (is (< (Math/abs (- 1.0 (/ bsd b-sd))) 0.1))
    (is (< (Math/abs (- (m/log-marginal r) log-z)) 0.5))
    (testing "the memory bound"
      (is (<= (count (:coreset r)) 50))
      (is (<= (:max-coreset-size r) 70))
      (is (= 100 (:steps r))))))

(deftest with-memory-for-all-data-nothing-is-compressed
  (random/set-seed! 4)
  (let [data (regression 200 5)
        r (scmc/scmc (model data 20) {:particles 300 :memory 200 :seed [2]})]
    (is (= 200 (count (:coreset r))))
    (is (every? #(= 1.0 (first %)) (:coreset r)))
    (is (< (Math/abs (- (m/log-marginal r) (:log-z (exact data)))) 0.5))))

(deftest refusals
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":memory" (scmc/scmc (model [] 1) {:memory 0})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":particles" (scmc/scmc (model [] 1) {:particles 0}))))

(deftest batches-are-consumed-as-they-come
  ;; a lazy stream of batches: each is realized when its step reaches it
  (random/set-seed! 6)
  (let [data (regression 200 7)
        realized (atom 0)
        batches (map (fn [b] (swap! realized inc) b) (partition 20 data))
        seen (atom [])
        m (model data 20)
        log-lik (:log-lik m)
        r (scmc/scmc (assoc m
                            :batches batches
                            :log-lik (fn [q d] (swap! seen conj @realized) (log-lik q d)))
                     {:particles 50 :memory 30 :seed [3]})]
    (is (= 10 @realized))
    (is (= 50 (count (mapv (comp m/get-value first) (m/get-particles r)))))
    (testing "the first batch was scored before the last one was realized"
      (is (< (first @seen) 10)))))

(deftest a-batch-no-particle-explains-is-refused
  (random/set-seed! 7)
  (let [m (assoc (model (regression 40 8) 20)
                 :log-lik (fn [_ [x _]] (if (> x 100.0) 0.0 ##-Inf)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"zero likelihood"
                          (scmc/scmc m {:particles 20 :memory 10 :seed [4]})))))

(deftest settled-result-uses-the-shared-measure-protocol
  (let [result (scmc/scmc {:prior-sample (constantly [2.0])
                           :prior-log-density (constantly 0.0)
                           :log-lik (constantly 0.0)
                           :batches []}
                          {:particles 8 :seed 17})
        particles (m/get-particles result)]
    (is (satisfies? m/PMeasure result))
    (is (= :empirical (m/measure-type result)))
    (is (= (vec (repeat 8 [2.0])) (mapv (comp m/get-value first) particles)))
    (is (every? #(= 0.0 (second %)) particles))
    (is (= 8.0 (m/effective-sample-size result)))
    (is (= 0.0 (m/log-marginal result)))
    (is (= 2.0 (:mean (m/measure-stats result (comp first m/get-value)))))
    (is (= :weighted-population (get-in result [:inference-contract :semantics])))
    (is (= [] (:coreset result)))
    (is (not (contains? result :log-z)))))
