(ns org.replikativ.foerster.derive-test
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.derive :as derive]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.smc :as smc]))

(def ^:private weights [0.2 0.5 0.3])
(def ^:private means [-2.0 0.0 3.0])
(def ^:private ys [-1.7 0.4 2.2 3.5 -0.2])

(def ^:private mixture
  (a/model [ys]
           (loop [i 0 zs []]
             (if (< i (count ys))
               (let [z (sample (dist/discrete [0.2 0.5 0.3]) :id [:z i])]
                 (observe (dist/normal (nth [-2.0 0.0 3.0] z) 1.0) (nth ys i) :id [:y i])
                 (recur (inc i) (conj zs z)))
               zs))))

(defn- component-logs [y]
  (map (fn [w mu] (+ (Math/log w) (dist/logpdf (dist/normal mu 1.0) y))) weights means))

(deftest a-mixture-indicator-is-enumerated
  (let [{:keys [enumerate refused]} (derive/plan (a/graph-of mixture))]
    (is (= ['[:z i]] (map :address (vals enumerate))))
    (is (empty? refused))
    (is (= 1 (count (:group (first (vals enumerate))))))))

(deftest summed-out-indicators-leave-the-exact-evidence
  ;; nothing random is left once every z_i is summed out: every particle
  ;; carries the exact marginal, whatever their number
  (let [exact (reduce + (map #(m/log-sum-exp (component-logs %)) ys))
        plan (derive/plan (a/graph-of mixture))]
    (doseq [n [1 7]]
      (let [measure (b/run-infer 3 #(smc/smc (mixture ys) n {:enumerate plan}))]
        (is (< (Math/abs (- exact (m/log-marginal measure))) 1e-9)
            (str (m/log-marginal measure) " vs " exact))))
    (testing "each z_i is an exact draw from its conditional"
      (let [measure (b/run-infer 4 #(smc/smc (mixture ys) 3000 {:enumerate plan}))
            zs (map (comp first m/get-value) (m/get-contexts measure))
            freq (frequencies zs)
            post (m/normalize-log-weights (vec (component-logs (first ys))))]
        (doseq [k (range 3)]
          (is (< (Math/abs (- (nth post k) (/ (get freq k 0) 3000.0))) 0.03)
              (str "z_0 = " k)))))
    (testing "seeded runs repeat"
      (is (= (map m/get-value (m/get-contexts (b/run-infer 5 #(smc/smc (mixture ys) 20 {:enumerate plan}))))
             (map m/get-value (m/get-contexts (b/run-infer 5 #(smc/smc (mixture ys) 20 {:enumerate plan})))))))))

(def ^:private branching
  (a/model []
           (let [z (sample (dist/flip 0.3) :id :z)]
             (if z
               (observe (dist/normal 1.0 1.0) 0.5 :id :a)
               (observe (dist/normal -1.0 2.0) 0.5 :id :b))
             z)))

(deftest a-branch-decided-by-the-summed-out-site
  (let [exact (m/log-sum-exp [(+ (Math/log 0.3) (dist/logpdf (dist/normal 1.0 1.0) 0.5))
                              (+ (Math/log 0.7) (dist/logpdf (dist/normal -1.0 2.0) 0.5))])
        measure (b/run-infer 6 #(smc/smc (branching) 5 {:enumerate (derive/plan (a/graph-of branching))}))]
    (is (< (Math/abs (- exact (m/log-marginal measure))) 1e-9))))

(def ^:private one-impossible
  (a/model []
           (let [z (sample (dist/discrete [0.5 0.5]) :id :z)]
             (observe (dist/uniform (if (= 0 z) 0.0 5.0) (if (= 0 z) 1.0 6.0)) 0.5 :id :y)
             z)))

(deftest an-impossible-value-is-never-drawn
  (let [measure (b/run-infer 7 #(smc/smc (one-impossible) 50 {:enumerate (derive/plan (a/graph-of one-impossible))}))]
    (is (< (Math/abs (- (Math/log 0.5) (m/log-marginal measure))) 1e-9))
    (is (every? zero? (map m/get-value (m/get-contexts measure))))))

(deftest refused-with-options-it-does-not-support
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":enumerate is not supported with :batch"
                        (b/run-infer 8 #(smc/smc (branching) 5 {:enumerate {} :batch 2})))))

(deftest what-is-not-enumerated
  (testing "a chain: z is read in the next iteration"
    (is (= [:carried]
           (vals (:refused (derive/plan
                            (a/graph-of
                             (a/model [n]
                                      (loop [t 0 prev 0]
                                        (when (< t n)
                                          (let [x (sample (dist/bernoulli (if (= 1 prev) 0.9 0.1)) :id [:x t])]
                                            (observe (dist/normal x 1.0) 0.3 :id [:y t])
                                            (recur (inc t) x))))))))))))
  (testing "a latent child"
    (is (= [:latent-child]
           (vals (:refused (derive/plan
                            (a/graph-of
                             (a/model []
                                      (let [z (sample (dist/flip 0.5) :id :z)
                                            m (sample (dist/normal (if z 1.0 -1.0) 1.0) :id :m)]
                                        (observe (dist/normal m 1.0) 0.2 :id :y))))))))))
  (testing "an unrelated site between z and its last child"
    (is (= [:not-contiguous]
           (vals (:refused (derive/plan
                            (a/graph-of
                             (a/model []
                                      (let [z (sample (dist/flip 0.5) :id :z)]
                                        (observe (dist/normal (if z 1.0 0.0) 1.0) 0.2 :id :a)
                                        (sample (dist/normal 0 1) :id :w)
                                        (observe (dist/normal (if z 1.0 0.0) 1.0) 0.4 :id :b)))))))))))

(deftest a-branch-decided-by-z-is-its-group
  (let [{:keys [enumerate refused]}
        (derive/plan (a/graph-of (a/model []
                                          (let [z (sample (dist/flip 0.3) :id :z)]
                                            (if z
                                              (observe (dist/normal 1.0 1.0) 0.5 :id :a)
                                              (observe (dist/normal -1.0 2.0) 0.5 :id :b))))))]
    (is (empty? refused))
    (is (= 2 (count (:group (first (vals enumerate))))))))
