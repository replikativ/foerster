(ns org.replikativ.foerster.reactive-telemetry-long-test
  "A smoke run of the long-stream experiment (E1) on a short stream. Run
  explicitly with experiments and test on the classpath."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.reactive-telemetry-long :as long]))

(deftest a-short-long-stream-run
  (let [config {:windows 150 :data-seeds [7] :seeds [1] :particles 40 :checkpoints 3 :grid 41
                :arms [[:full] [:scmc 16] [:scmc 16 {:projection-inflation 3}] [:reservoir 16]]}
        {:keys [runs]} (long/experiment config)
        by-arm (into {} (map (juxt :arm identity)) runs)]
    (is (= 4 (count runs)))
    (testing "the stream covers low, mid and high load"
      (let [rows (long/observations (merge long/defaults config) 7)]
        (is (= 150 (count rows)))
        (is (every? #(<= -1.0 (:load %) 0.0) (subvec rows 0 30)))
        (is (every? #(<= 0.0 (:load %) 1.0) (subvec rows 90)))
        (is (< 4 (count (distinct (map :load rows)))))))
    (doseq [{:keys [updates checkpoints references]} runs]
      (is (= 10 (count updates)))
      (is (= [60 105 150] (mapv :t checkpoints)))
      (is (every? #(< (:boundary-mass %) 1e-6) (vals references)))
      (is (every? #(Double/isFinite %) (mapcat #(get-in % [:errors :mean-in-sd]) checkpoints))))
    (testing "SCMC's per-update cost is bounded by its memory, the full history's grows"
      (let [bound (+ (* 40 15) (* 2 40 (+ 16 15)))]
        (is (every? #(<= (:calls %) bound) (:updates (by-arm [:scmc 16]))))
        (is (every? #(<= (:retained %) 16) (:updates (by-arm [:scmc 16]))))
        (is (every? #(<= (:retained %) 16) (:updates (by-arm [:reservoir 16]))))
        (is (> (:calls (peek (:updates (by-arm [:full])))) bound))
        (is (= (+ (* 40 15) (* 2 40 150)) (:calls (peek (:updates (by-arm [:full]))))))))
    (testing "the inflated projection pays for its points on compression steps"
      (is (some #(pos? (get-in % [:evaluations :projection])) (:updates (by-arm [:scmc 16 {:projection-inflation 3}]))))
      (is (every? #(zero? (get-in % [:evaluations :projection])) (:updates (by-arm [:scmc 16])))))))
