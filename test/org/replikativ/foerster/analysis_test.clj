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

(defn- conjugate-priors [g] (set (map :prior (facts-of g :conjugate))))

(deftest a-collapsed-parameter-may-only-be-returned
  (testing "returned, alone or in the result's collection: collapses"
    (is (= 1 (count (conjugate-priors
                     (a/analyze (spin (let [p (sample (dist/beta 1 1) :id :p)]
                                        (observe (dist/bernoulli p) 1 :id :a)
                                        p)))))))
    (is (= 1 (count (conjugate-priors
                     (a/analyze (spin (let [p (sample (dist/beta 1 1) :id :p)]
                                        (observe (dist/bernoulli p) 1 :id :a)
                                        {:p p :n 1}))))))))
  (testing "anywhere else, its value is used before its posterior is known"
    (doseq [g [(a/analyze (spin (let [p (sample (dist/beta 1 1) :id :p)]
                                  (swap! seen conj p)
                                  (observe (dist/bernoulli p) 1 :id :a))))
               (a/analyze (spin (let [p (sample (dist/beta 1 1) :id :p)
                                      f (fn [] p)]
                                  (observe (dist/bernoulli p) 1 :id :a)
                                  f)))
               (a/analyze (spin (let [p (sample (dist/beta 1 1) :id :p)]
                                  (loop [i 0 acc []]
                                    (when (< i 2)
                                      (observe (dist/bernoulli p) 1 :id [:a i])
                                      (recur (inc i) (conj acc p)))))))]]
      (is (empty? (conjugate-priors g)))
      (is (seq (:escapes (first (facts-of g :latent))))))))

(deftest link-functions
  (testing "a Normal mean affine in a Normal prior: conjugate, with its link"
    (let [g (a/analyze (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
                               (observe (dist/normal (+ 1.0 (* 2.0 mu)) 0.5) 0.3 :id :y)
                               mu)))
          [f] (facts-of g :conjugate)]
      (is (= :normal-affine (:family f)))
      (is (= {:a 1.0 :b 2.0} (val (first (:links f)))))))
  (testing "a mean another site enters: conjugate given it"
    (let [g (a/analyze (spin (let [m1 (sample (dist/normal 0.0 1.0) :id :m1)
                                   m2 (sample (dist/normal 0.0 1.0) :id :m2)]
                               (observe (dist/normal (+ m1 m2) 1.0) 0.3 :id :y))))
          by-prior (into {} (map (juxt :address identity)) (facts-of g :conjugate))]
      (is (= #{:m1 :m2} (set (keys by-prior))))
      (is (= #{1} (:given (by-prior :m1))))))
  (testing "not affine: no collapse"
    (is (empty? (facts-of (a/analyze (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
                                             (observe (dist/normal (* mu mu) 1.0) 0.3 :id :y))))
                          :conjugate))))
  (testing "a derived value that leaves the model escapes"
    (is (empty? (facts-of (a/analyze (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
                                             (swap! seen conj (* 2 mu))
                                             (observe (dist/normal mu 1.0) 0.3 :id :y))))
                          :conjugate))))
  (testing "affine"
    (is (= [0 1] (a/affine {:site 3} 3)))
    (is (= [-4 3] (a/affine '(clojure.core/- (clojure.core/* 3 {:site 0}) 4) 0)))
    (is (nil? (a/affine '(clojure.core/* {:site 0} {:site 0}) 0)))))

(def ^:private chain
  (a/model [n ys]
           (loop [t 0 prev 0]
             (when (< t n)
               (let [x (sample (dist/bernoulli (if (= 1 prev) 0.9 0.1)) :id [:x t])]
                 (observe (dist/normal x 1.0) (nth ys t) :id [:y t])
                 (recur (inc t) x))))))

(def ^:private indicators
  (a/model [ys]
           (loop [i 0]
             (when (< i (count ys))
               (let [z (sample (dist/flip 0.5) :id [:z i])]
                 (observe (dist/normal (if z 1.0 -1.0) 1.0) (nth ys i) :id [:y i])
                 (recur (inc i)))))))

(deftest specialized-to-data
  (testing "a sequential loop with known bounds unrolls"
    (let [g (a/specialize chain 3 [0.1 0.9 0.4])
          s (by-address g)
          [l] (vals (:loops g))]
      (is (= #{[:x 0] [:x 1] [:x 2] [:y 0] [:y 1] [:y 2]} (set (keys s))))
      (is (= #{(:n (s [:x 0]))} (:reads (s [:x 1]))) "x_1 reads x_0 only")
      (is (= #{(:n (s [:x 1]))} (:reads (s [:y 1]))))
      (is (= 4 (:unrolled l)) "three steps and the pass that ends the loop")
      (is (= #{(:n (s [:x 0])) (:n (s [:x 1]))} (:carried l)))))
  (testing "a plate stays symbolic"
    (let [g (a/specialize indicators [0.3 -1.2 0.8])]
      (is (= 2 (count (:sites g))))
      (is (nil? (:unrolled (first (vals (:loops g))))))
      (is (= :plate (:kind (first (vals (:loops g))))))))
  (testing "a known test takes one branch; constants fold"
    (let [g (a/specialize (a/model [prior? k]
                                   (if prior?
                                     (sample (dist/normal 0.0 1.0) :id :a)
                                     (observe (dist/normal (* 2.0 k) 1.0) 0.5 :id :b)))
                          false 1.5)
          [site] (vals (:sites g))]
      (is (= 1 (count (:sites g))))
      (is (= :b (:address site)))
      (is (= 3.0 (:value (first (get-in site [:dist :args])))))))
  (testing "an unknown test is the site's path condition"
    (let [g (a/analyze (spin (let [c (sample (dist/flip 0.5) :id :c)]
                               (when c (observe (dist/normal 0 1) 0.2 :id :y)))))
          y (:path ((by-address g) :y))]
      (is (= [[{:site 0} true]] y)))))

(deftest linear-gaussian-blocks
  (testing "a regression's intercept and slope form one block"
    (let [g (a/analyze (spin (let [a (sample (dist/normal 0.0 5.0) :id :a)
                                   b (sample (dist/normal 0.0 5.0) :id :b)]
                               (loop [i 0]
                                 (when (< i (count xs))
                                   (observe (dist/normal (+ a (* b (nth xs i))) 0.3) (nth ys i) :id [:y i])
                                   (recur (inc i))))
                               [a b])))
          s (by-address g)]
      (is (= [{:latents #{(:n (s :a)) (:n (s :b))} :observations #{(:n (s '[:y i]))}}]
             (map #(dissoc % :fact) (facts-of g :linear-gaussian))))))
  (testing "a Gaussian random walk, unrolled, is one block (a Kalman filter's)"
    (let [g (a/specialize (a/model [n ys]
                                   (loop [t 0 x 0.0]
                                     (when (< t n)
                                       (let [x' (sample (dist/normal x 1.0) :id [:x t])]
                                         (observe (dist/normal x' 0.5) (nth ys t) :id [:y t])
                                         (recur (inc t) x')))))
                          3 [0.2 0.1 0.5])
          [b] (facts-of g :linear-gaussian)]
      (is (= 3 (count (:latents b))))
      (is (= 3 (count (:observations b))))))
  (testing "a square is not linear"
    (is (empty? (facts-of (a/analyze (spin (let [a (sample (dist/normal 0.0 1.0) :id :a)]
                                             (observe (dist/normal (* a a) 1.0) 0.3 :id :y))))
                          :linear-gaussian))))
  (testing "linear"
    (is (= {:const 1.0 :coefs {0 2.0 1 -1}}
           (a/linear '(clojure.core/- (clojure.core/+ 1.0 (clojure.core/* 2.0 {:site 0})) {:site 1}) #{0 1})))))
