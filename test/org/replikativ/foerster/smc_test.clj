(ns org.replikativ.foerster.smc-test
  "SMC's resampling schemes and its per-barrier history."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.smc :as smc]))

(deftest every-resampling-scheme-estimates-the-evidence
  (doseq [scheme [:systematic :stratified :residual :multinomial]]
    (testing (name scheme)
      (let [measure (b/run-infer 21 #(smc/smc (b/conjugate-model) 1000 {:resampling scheme
                                                                        :resample-threshold 1.0}))]
        (is (< (Math/abs (- (m/log-marginal measure) b/conjugate-log-evidence)) 0.2)
            (str (m/log-marginal measure) " vs " b/conjugate-log-evidence))))))

(deftest the-history-records-every-barrier
  (let [n 50
        measure (b/run-infer 22 #(smc/smc (apply b/hmm-model b/hmm-args) n
                                          {:resample-threshold 0.5 :genealogy? true}))
        history (:history measure)]
    (is (= 16 (count history)) "one entry per observation")
    (is (every? #(<= 1.0 (:ess %) n) history))
    (is (some :resampled? history))
    (is (every? (fn [{:keys [resampled? ancestors]}]
                  (if resampled?
                    (and (= n (count ancestors)) (every? #(< -1 % n) ancestors))
                    (nil? ancestors)))
                history))
    (testing "the evidence is the sum of the barriers' factors and the final mean weight"
      (let [final (m/log-mean-exp (m/get-log-weights measure))
            factors (reduce + (keep :log-mean-weight history))]
        (is (< (Math/abs (- (m/log-marginal measure) (+ factors final))) 1e-9))))))
