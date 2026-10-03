(ns org.replikativ.foerster.cascade-test
  "The particle cascade against exact answers: the evidence of a conjugate
  model (with room to branch, and with a cap that collapses copies into
  multiplicities) and the HMM's marginals on a thread pool."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.cascade :as cascade]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.engine.executor :as executor]))

(deftest the-cascade-estimates-the-evidence
  (doseq [[label opts] [["room to branch" {}] ["a tight cap" {:cap 300}]]]
    (testing label
      (let [measure (b/run-infer 71 #(cascade/cascade (b/conjugate-model) 300 opts))
            {:keys [peak collapsed]} (:cascade measure)]
        ;; over 8 runs per setting the log error stayed within ±0.25
        (is (< (Math/abs (- (m/log-marginal measure) b/conjugate-log-evidence)) 0.4)
            (str (m/log-marginal measure) " vs " b/conjugate-log-evidence))
        (is (<= peak (or (:cap opts) 1200)))))))

(deftest the-cascade-follows-the-hmm-on-a-thread-pool
  (let [exec (executor/thread-pool-executor {:threads 4})]
    (try
      (doseq [[label n opts] [["room to branch" 400 {}] ["a little room" 300 {:cap 400}]]]
        (testing label
          (let [measure (b/run-infer 72 #(cascade/cascade (apply b/hmm-model b/hmm-args) n
                                                          (assoc opts :executor exec)))]
            ;; arrival order on the pool is not seeded: over 20 runs per
            ;; setting the error stayed below 0.1
            (is (< (b/hmm-error (b/weighted-values measure)) 0.12))
            (is (<= (:peak (:cascade measure)) (or (:cap opts) 1600))))))
      (finally (.close ^java.lang.AutoCloseable exec)))))

(def ^:private hmm-log-evidence
  ;; the forward algorithm
  (let [[ys init trans obs] b/hmm-args
        p-init (mapv #(Math/exp (dist/logpdf init %)) (range 3))
        step (fn [alpha y]
               (mapv (fn [j]
                       (* (Math/exp (dist/logpdf (get obs j) y))
                          (reduce + (map (fn [i a] (* a (Math/exp (dist/logpdf (get trans i) j)))) (range 3) alpha))))
                     (range 3)))]
    (Math/log (reduce + (reduce step p-init ys)))))

(deftest collapsed-copies-keep-the-evidence
  ;; no room at all: most copies fold into multiplicities, and the evidence
  ;; must still come out right
  (let [measure (b/run-infer 73 #(cascade/cascade (apply b/hmm-model b/hmm-args) 400 {:cap 400}))
        {:keys [peak collapsed]} (:cascade measure)]
    (is (<= peak 400))
    (is (pos? collapsed))
    ;; arrival order is not seeded on the default executor: over 20 runs
    ;; the log evidence erred by −0.08 on average, sd 0.3, at most 0.56
    (is (< (Math/abs (- (m/log-marginal measure) hmm-log-evidence)) 0.9)
        (str (m/log-marginal measure) " vs " hmm-log-evidence))))
