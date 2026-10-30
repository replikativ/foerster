(ns org.replikativ.foerster.gsdm-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.gsdm :as experiment]
            [org.replikativ.foerster.dist :as dist]))

(deftest analytic-posterior-agrees-with-independent-quadrature
  (doseq [evidence [{:x 1.0 :y 2.5} {:x -1.0 :y -3.0} {:x 2.0 :y 6.0}]]
    (let [{:keys [x y]} evidence
          points (mapv #(+ -9.0 (* % 0.003)) (range 6001))
          densities (mapv #(Math/exp (+ (dist/logpdf (dist/normal 0 1) %)
                                        (dist/logpdf (dist/normal % 1) x)
                                        (dist/logpdf (dist/normal (+ (* 2 x) %) 1) y))) points)
          mass (reduce + densities)
          mu (/ (reduce + (map * densities points)) mass)
          variance (/ (reduce + (map (fn [z p] (* p (Math/pow (- z mu) 2))) points densities)) mass)
          oracle (experiment/oracle evidence)]
      (is (< (Math/abs (- mu (:z-mean oracle))) 1e-9))
      (is (< (Math/abs (- variance (:z-variance oracle))) 1e-9)))))

(deftest v-training-target-and-noised-state-reconstruct-the-original
  (let [system {:config {:prediction :v} :betas (double-array 4)
                :alpha-bars (double-array [0.99 0.7 0.3 0.001])}
        rng (java.util.Random. 19)]
    (dotimes [_ 20]
      (let [{:keys [values target z t]} (experiment/training-example system rng)
            a (aget ^doubles (:alpha-bars system) t)
            reconstructed (- (* (Math/sqrt a) (aget ^doubles values 0))
                             (* (Math/sqrt (- 1 a)) (aget ^doubles target 0)))]
        (is (< (Math/abs (- z reconstructed)) 1e-12))))))

(deftest abduction-retains-the-factual-observation
  (doseq [z [-2.0 0.0 1.5]]
    (let [evidence {:x 1.0 :y 2.5} {:keys [noise factual]} (experiment/abduction evidence z)]
      (is (= (:x evidence) (+ (:z noise) (:x noise))))
      (is (= (:y evidence) (+ (* 2 (:x factual)) (:z noise) (:y noise)))))))

(deftest oracle-ddpm-draws-exercise-the-real-causal-world-api
  ;; No compiled network is needed: the analytic denoiser isolates the replay
  ;; contract from approximation and training errors in GSDM.
  (let [system {:oracle-denoiser? true :betas (double-array [0.1 0.3 0.9])
                :alpha-bars (double-array [0.9 0.63 0.063])}
        report (experiment/check-causal-query! system {:x 1.0 :y 2.5} 8 12)]
    (is (:complete? report))
    (is (< (:max-replay-error report) 1e-12))
    (is (< (Math/abs (- 0.5 (get-in report [:y-do-x-zero :mean]))) 1e-12))
    (is (< (get-in report [:y-do-x-zero :variance]) 1e-24))))
