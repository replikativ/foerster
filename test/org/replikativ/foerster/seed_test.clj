(ns org.replikativ.foerster.seed-test
  "`:seed` fixes an iterated particle method's run, and its sweeps still
  draw afresh."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- model []
  (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
          (observe (dist/normal x 1.0) 0.5 :id :y)
          x)))

(defn- values [opts process-seed]
  (mapv (comp m/get-value first)
        (m/get-particles (b/run-infer process-seed #(infer/infer (model) opts)))))

(deftest seeded-sweeps-draw-afresh
  (doseq [opts [{:method :pimh :particles 5 :iterations 4 :seed 99}
                {:method :pgibbs :particles 5 :iterations 4 :seed 99}
                {:method :ipmcmc :particles 5 :iterations 2 :num-nodes 4 :seed 99}]]
    (testing (name (:method opts))
      (let [vs (values opts 1)]
        (is (< (* 2 (count (distinct (take 5 vs)))) (count (distinct vs))) "the sweeps differ")
        (is (= vs (values opts 2)) ":seed fixes the run, whatever the process generator")))))

(deftest seeded-tempering-is-fixed
  (let [opts {:method :tempered :particles 8 :seed 99}
        vs (values opts 1)]
    (is (= vs (values opts 2)) ":seed fixes the run, whatever the process generator")
    (is (not= vs (values (assoc opts :seed 100) 1)) "another seed, another run")))

(deftest impossible-and-empty-runs-say-so
  (testing "a population whose particles are all impossible has no posterior"
    (let [measure (b/run-infer 3 #(infer/infer (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                                                       (observe (dist/uniform 5.0 6.0) x :id :y)
                                                       x))
                                               {:method :smc :particles 20}))]
      (is (= ##-Inf (m/log-marginal measure)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probability zero"
                            (m/measure-stats measure identity)))))
  (testing "particle MCMC on a model that is impossible everywhere reports it"
    ;; conditional SMC resamples at every barrier, which must not make an
    ;; impossible population possible
    (doseq [wp [:fresh :fork]
            opts [{:method :pimh :particles 5 :iterations 3}
                  {:method :pgibbs :particles 5 :iterations 3}
                  {:method :pgas :particles 5 :iterations 3}
                  {:method :ipmcmc :particles 5 :iterations 2 :num-nodes 4}]]
      (testing [(:method opts) wp]
        (let [measure (b/run-infer 3 #(infer/infer (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                                                           (observe (dist/uniform 5.0 6.0) x :id :y)
                                                           x))
                                                   (assoc opts :world-policy wp)))]
          (is (nil? (m/log-marginal measure)) "pooled chains do not estimate evidence")
          (is (every? #(= ##-Inf (second %)) (m/get-particles measure))
              "resampling must not give impossible particles positive weight")
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probability zero"
                                (m/measure-stats measure m/get-value)))))))
  (testing "no particles, no measure"
    (is (thrown? Throwable (b/run-infer 4 #(infer/infer (model) {:method :smc :particles 0}))))
    (is (thrown? AssertionError (m/empirical [])))
    (is (thrown? clojure.lang.ExceptionInfo (m/normalize-log-weights [0.0 ##NaN]))))
  (testing "infinite weights share the mass"
    (is (= [0.5 0.0 0.5] (m/normalize-log-weights [##Inf 0.0 ##Inf])))))

(deftest a-burn-that-leaves-no-draws-is-refused
  (is (thrown-with-msg? Throwable #":burn leaves no draws"
                        (b/run-infer 5 #(infer/infer (model) {:method :mh :iterations 10 :burn 10})))))
