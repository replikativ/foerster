(ns org.replikativ.foerster.analysis-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.conjugate :as conj]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- by-address [g]
  (into {} (for [[n s] (:sites g)] [(:address s) (assoc s :n n)])))

(defn- facts-of [g kind] (filter #(= kind (:fact %)) (a/facts g)))

(deftest normal-mean-with-a-plate
  (let [g (a/analyze
           (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
                   (loop [i 0]
                     (when (< i (count ys))
                       (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
                       (recur (inc i)))))))
        s (by-address g)]
    (is (= #{(:n (s :mu))} (:reads (s '[:y i]))))
    (is (= :plate (:kind (first (vals (:loops g))))))
    (is (= [{:family :normal-mean :children #{(:n (s '[:y i]))} :implemented? true
             :fact :conjugate :prior (:n (s :mu)) :address :mu}]
           (facts-of g :conjugate)))
    (is (= [(:n (s :mu))] (map :site (facts-of g :gradient-candidate))))))

(deftest a-hidden-markov-chain-is-sequential
  (let [g (a/analyze
           (spin (loop [t 0 prev 0 xs []]
                   (if (= t n)
                     xs
                     (let [x (sample (dist/bernoulli (if (= 1 prev) 0.7 0.2)) :id [:x t])]
                       (observe (dist/normal x 1.0) (nth ys t) :id [:y t])
                       (recur (inc t) x (conj xs x)))))))
        s (by-address g)
        x (:n (s '[:x t]))]
    (is (= :sequential (:kind (first (vals (:loops g))))))
    (is (contains? (:reads (s '[:x t])) x) "x_t reads x_{t-1}, carried")
    (is (= [x] (map :site (facts-of g :finite-support))))
    (is (empty? (facts-of g :conjugate)) "x is no conjugate prior")))

(deftest collecting-samples-is-not-a-dependence
  (let [g (a/analyze
           (spin (loop [i 0 zs []]
                   (if (= i 3)
                     zs
                     (let [z (sample (dist/discrete [1 2 3]) :id [:z i])]
                       (observe (dist/normal z 1.0) 0.5 :id [:y i])
                       (recur (inc i) (conj zs z)))))))]
    (is (= :plate (:kind (first (vals (:loops g))))))))

(deftest control-and-blanket
  (let [g (a/analyze
           (spin (let [c (sample (dist/flip 0.3) :id :c)
                       m (sample (dist/normal 0 1) :id :m)]
                   (if c
                     (observe (dist/normal m 1.0) 0.2 :id :y)
                     (observe (dist/normal 0 2.0) 0.2 :id :y2)))))
        s (by-address g)]
    (is (= #{(:n (s :c))} (:control (s :y))))
    (is (= [(:n (s :m))] (map :site (facts-of g :gradient-candidate))))
    (is (= [(:n (s :m))] (map :prior (facts-of g :conjugate)))
        "a child that runs only if c still collapses: c does not depend on m")
    (is (= #{(:n (s :c)) (:n (s :y))}
           (:blanket (first (filter #(= (:n (s :m)) (:site %)) (facts-of g :markov-blanket))))))))

(deftest a-prior-that-decides-a-branch-is-not-collapsed
  (let [g (a/analyze
           (spin (let [p (sample (dist/beta 1 1) :id :p)]
                   (observe (dist/bernoulli p) 1 :id :a)
                   (when (> p 0.5)
                     (observe (dist/normal 0 1) 0.3 :id :b)))))]
    (is (empty? (facts-of g :conjugate)))))

(deftest beta-bernoulli-through-a-transformation-is-not-conjugate
  (let [g (a/analyze
           (spin (let [p (sample (dist/beta 1 1) :id :p)]
                   (observe (dist/bernoulli p) 1 :id :a)
                   (observe (dist/bernoulli (* 0.5 p)) 1 :id :b))))]
    (is (empty? (facts-of g :conjugate)))))

(deftest closures-are-reported
  (let [g (a/analyze
           (spin (let [f (fn [k] (spin (sample (dist/normal 0 1) :id [:mean k])))]
                   (f 1))))]
    (is (= [:closure] (map :reason (facts-of g :unknown))))))

(deftest hand-collapsed-conjugate-stays-opaque
  (let [g (a/analyze
           (spin (loop [c (conj/beta-bernoulli 1 1) i 0]
                   (if (= i 3)
                     (sample (conj/posterior c) :id :p)
                     (do (observe (conj/predictive c) 1 :id [:y i])
                         (recur (conj/update c 1) (inc i)))))))]
    (is (= 2 (count (:sites g))))
    (is (empty? (facts-of g :conjugate)))))
