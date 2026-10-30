(ns org.replikativ.foerster.reactive-telemetry-test
  "Numerical and reactive admission checks for the reviewed telemetry fixture.
  Run explicitly with experiments and test on the classpath."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.reactive-telemetry :as telemetry]))

(def receipt
  (delay
    (let [output "/tmp/foerster-reactive-telemetry-validated.edn"]
      (telemetry/-main output)
      (edn/read-string (slurp output)))))

(defn accurate? [errors]
  (and (every? #(< % 0.25) (:mean-in-sd errors))
       (every? #(< % 0.1) (:relative-sd errors))
       (< (:risk-absolute errors) 0.01)))

(deftest quadrature-is-resolved-and-the-posterior-is-nontrivial
  (let [r @receipt]
    (doseq [error (:grid-check r)]
      (is (every? #(< % 0.005) (:mean-in-sd error)))
      (is (every? #(< % 0.005) (:relative-sd error))))
    (doseq [ref (:references r)] (is (< (:boundary-mass ref) 1e-7)))
    ;; The covariate shift changes both identification and posterior geometry.
    (is (> (:correlation (first (:references r))) 0.6))
    (is (< (:correlation (nth (:references r) 2)) -0.6))
    (is (> (/ (get-in r [:references 0 :sd 1]) (get-in r [:references 2 :sd 1])) 4.0))))

(deftest accuracy-and-cost-are-checked-independently-of-ess
  (let [fits (:fits @receipt)
        by-key (into {} (map #(vector [(:memory %) (:seed %)] %) fits))]
    (doseq [seed [31 32 33] memory [128 300]]
      (let [fit (by-key [memory seed])]
        (is (accurate? (:errors fit)))
        (is (<= (:coreset-size fit) memory))
        (is (<= (:max-coreset-size fit) (+ memory telemetry/batch-size)))))
    (doseq [seed [31 32 33]]
      (is (< (:calls (by-key [128 seed])) (* 0.65 (:calls (by-key [300 seed]))))))
    (let [poor (filter #(= 8 (:memory %)) (:fits @receipt))]
      (doseq [fit poor]
        (is (not (accurate? (:errors fit))))
        (is (> (apply max (get-in fit [:errors :mean-in-sd])) 0.8))
        (is (> (get-in fit [:last-update :ess-fraction]) 0.7))))))

(deftest revisions-corrections-and-speculation
  (let [r (:reactive @receipt)]
    (is (= [:append :append :append :replay] (mapv :operation (:updates r))))
    (is (= [1 2 3 4] (mapv :revision (:updates r))))
    (doseq [update (:updates r)]
      (is (accurate? (:errors update)))
      (is (= :not-assessed (:posterior-accuracy update)))
      (is (<= (:coreset-size update) 128)))
    (is (= [:stale-base :scorer-changed :invalid-observations] (mapv :reason (:refusals r))))
    (is (:unchanged-after-refusals? r))
    (is (:duplicate-is-noop? r))
    (is (:fork-isolated? r))
    (is (= 5 (get-in r [:branch-result :revision])))
    (is (pos? (get-in r [:updates 3 :batch-splits])))
    (is (= :needs-recovery (get-in r [:stress-result :status])))
    (is (= :low-ess (get-in r [:stress-result :reason])))
    (is (= 4 (get-in r [:stress-result :revision])))
    (is (true? (get-in r [:stress-result :unchanged?])))
    (is (= :approximate-population (get-in r [:updates 3 :inference-contract :semantics])))))

(deftest selected-budget-on-independent-simulated-streams
  (doseq [seed [20261031 20261032]]
    (let [rows (telemetry/observations seed)
          ref (telemetry/reference rows 0.05)
          fit (telemetry/fit rows {:particles 800 :memory 128 :seed 31})]
      (is (accurate? (telemetry/errors (:summary fit) ref))))))
