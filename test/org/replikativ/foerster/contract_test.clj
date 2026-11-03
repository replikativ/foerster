(ns org.replikativ.foerster.contract-test
  "What results vouch for (`foerster.contract`): each method's normalizer
  status, the closed vocabulary, and the claim behind :unbiased checked
  against the exact evidence."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.contract :as contract]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private ys [0.4 -0.9 1.3 0.2 -0.5 1.1])

(defn- hmm
  "A two-state hidden Markov model over `ys`: 64 paths, so enumeration gives
  the evidence exactly."
  []
  (spin
   (loop [t 0 prev false]
     (if (= t (count ys))
       prev
       (let [x (sample (dist/flip (if prev 0.8 0.3)) :id [:x t])]
         (observe (dist/normal (if x 1.0 -1.0) 1.0) (nth ys t) :id [:y t])
         (recur (inc t) x))))))

(defn- status [seed opts] (contract/normalizer-status (b/run-infer seed #(infer/infer (hmm) opts))))

(deftest every-method-states-its-normalizer
  (is (= :exact (status 1 {:method :enumerate})))
  (is (= :unbiased (status 2 {:method :smc :particles 20})) "adaptive resampling, prior proposals")
  (is (= :unbiased (status 3 {:method :importance :particles 20})))
  (is (= :unverified (status 4 {:method :smc :particles 20 :policy (itrace/policy {})}))
      "a caller's policy carries the correction obligations")
  (is (= :unavailable (status 5 {:method :mh :iterations 50 :chains 2})))
  (testing "tempered SMC chooses its schedule from its population"
    (let [r (b/run-infer 6 #(infer/infer (hmm) {:method :tempered :particles 20}))]
      (is (= :unverified (contract/normalizer-status r)))
      (is (= :population (get-in r [:inference-contract :adaptation])))))
  (is (contract/unbiased-evidence? (b/run-infer 7 #(infer/infer (hmm) {:method :smc :particles 20}))))
  (is (not (contract/unbiased-evidence? (b/run-infer 8 #(infer/infer (hmm) {:method :mh :iterations 50 :chains 2}))))))

(deftest the-vocabulary-is-closed
  (is (= {:semantics :weighted-population :normalizer {:status :unverified}} (contract/weighted-population)))
  (is (= {:semantics :markov-chain :normalizer {:status :estimate :estimator :stepping-stone}}
         (contract/markov-chain {:estimator :stepping-stone})))
  (doseq [bad [{:normalizer :certified} {:semantics :posterior} {:adaptation :schedule}
               {:normalizer :estimate}]]
    (is (= ::contract/unknown-term
           (try (contract/weighted-population bad) nil
                (catch clojure.lang.ExceptionInfo e (:type (ex-data e)))))
        (pr-str bad))))

(deftest smc-evidence-is-unbiased
  ;; Ẑ/Z over independent runs of 8 particles with the default adaptive
  ;; resampling: its mean is 1 (the claim behind :unbiased), within 4
  ;; standard errors estimated from the runs themselves
  (let [z (Math/exp (m/log-marginal (b/run-infer 1 #(infer/infer (hmm) {:method :enumerate}))))
        ratios (vec (for [seed (range 400)]
                      (/ (Math/exp (m/log-marginal (b/run-infer (+ 1000 seed)
                                                                #(infer/infer (hmm) {:method :smc :particles 8}))))
                         z)))
        n (count ratios)
        mean (/ (reduce + ratios) n)
        se (Math/sqrt (/ (reduce + (map #(let [d (- % mean)] (* d d)) ratios)) (* n (dec n))))]
    (is (< (Math/abs (- mean 1.0)) (* 4 se)) (str "mean Ẑ/Z " mean " ± " se))
    (is (< se 0.05) "the check has power: the runs pin the mean to a few percent")))
