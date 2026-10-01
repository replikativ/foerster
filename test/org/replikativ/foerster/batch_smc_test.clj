(ns org.replikativ.foerster.batch-smc-test
  "Arrival-batched SMC: particles resample in batches of B as they arrive at
  each observation, without waiting for the slowest. The evidence stays
  unbiased and the posterior exact, whatever the arrival order."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.spindel.engine.executor :as executor]))

(deftest batches-keep-the-evidence-and-the-posterior
  (doseq [batch [1 8 1000]]
    (testing (str "B = " batch)
      (let [measure (b/run-infer 61 #(smc/smc (b/conjugate-model) 1000
                                              {:batch batch :resample-threshold 1.0}))]
        (is (< (Math/abs (- (m/log-marginal measure) b/conjugate-log-evidence)) 0.25)
            (str (m/log-marginal measure) " vs " b/conjugate-log-evidence))))))

(deftest batches-on-a-thread-pool-follow-the-hmm
  ;; arrival order is the executor's; the marginals stay correct
  (let [exec (executor/thread-pool-executor {:threads 4})]
    (try
      (let [measure (b/run-infer 62 #(smc/smc (apply b/hmm-model b/hmm-args) 600
                                              {:batch 16 :resample-threshold 1.0 :executor exec}))
            history (:history measure)]
        (is (< (b/hmm-error (b/weighted-values measure)) 0.08))
        (is (every? #(<= (:size %) 16) history))
        (is (= (* 16 600) (reduce + (map :size history))) "every particle passes every stage once"))
      (finally (.close ^java.lang.AutoCloseable exec)))))

(deftest a-stage-closes-when-the-rest-have-finished
  ;; b = true adds an observation: particles with b = false finish before the
  ;; second stage, whose last batch must close without them
  (let [measure (b/run-infer 63 #(smc/smc (b/varlen-model) 2000 {:batch 64 :resample-threshold 1.0}))
        p (b/w-mean #(if % 1.0 0.0) (b/weighted-values measure))]
    (is (< (Math/abs (- p b/varlen-truth)) 0.05) (str p " vs " b/varlen-truth))))
