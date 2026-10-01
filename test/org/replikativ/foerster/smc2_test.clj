(ns org.replikativ.foerster.smc2-test
  "PMMH and SMC² against exact answers: the observation noise s of a
  Gaussian random walk, whose likelihood the Kalman filter gives exactly, so
  its posterior and the evidence follow by quadrature over s."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc2 :as smc2]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

;; x_t = x_{t-1} + N(0,1), y_t ~ N(x_t, s), s ~ U(0.3, 2.0)
(def ^:private ys [0.4 -0.6 1.9 0.2 2.8 1.1 3.5 2.0])

(defn- model [stream?]
  (spin (let [s (sample (dist/uniform 0.3 2.0) :id :s)]
          (loop [t 0 x 0.0]
            (if (= t (count ys))
              s
              (let [x' (sample (dist/normal x 1.0) :id [:x t])]
                (if stream?
                  (sample (dist/normal x' s) :id [:y t] :stream true)
                  (observe (dist/normal x' s) (nth ys t) :id [:y t]))
                (recur (inc t) x')))))))

(defn- kalman-log-lik [s ys]
  (loop [[y & more] ys pm 0.0 pv 0.0 ll 0.0]
    (if-not y
      ll
      (let [pv (+ pv 1.0) sv (+ pv (* s s)) k (/ pv sv)]
        (recur more (+ pm (* k (- y pm))) (* (- 1.0 k) pv)
               (+ ll (dist/logpdf (dist/normal pm (Math/sqrt sv)) y)))))))

(defn- exact [ys]
  (let [h 1e-3 ss (range 0.3 2.0 h)
        ls (map #(Math/exp (kalman-log-lik % ys)) ss)
        z (* h (reduce + ls))]
    {:mean (/ (* h (reduce + (map * ss ls))) z)
     :log-evidence (Math/log (/ z 1.7))}))

(deftest pmmh-finds-the-noise-posterior
  (let [measure (b/run-infer 51 #(smc2/pmmh (model false) 40 150
                                            {:params #{:s} :scale 0.3 :burn 30}))
        thetas (map :s (:thetas measure))
        mean (/ (reduce + thetas) (count thetas))]
    (is (< 0.15 (:acceptance measure) 0.9) (str (:acceptance measure)))
    (is (< (Math/abs (- mean (:mean (exact ys)))) 0.2) (str mean " vs " (:mean (exact ys))))))

(defn- await-cps [operation]
  (let [p (promise)]
    (operation #(deliver p [:ok %]) #(deliver p [:error %]))
    (let [[k v] (deref p 300000 [:error (ex-info "timed out" {})])]
      (if (= :ok k) v (throw v)))))

(deftest smc2-tracks-the-noise-posterior-online
  (random/set-seed! 52)
  (let [root (context/create-execution-context)
        step0 (await-cps (smc2/smc2 (binding [ec/*execution-context* root] (model true))
                                    {:params #{:s} :n-theta 40 :n-x 40 :ess-target 0.9}))
        final (reduce (fn [step y] (await-cps ((:push step) y))) step0 ys)
        measure (:measure final)
        ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))
        mean (reduce + (map (fn [[p _] w] (* w (:s (m/get-value p)))) ps ws))
        {exact-mean :mean exact-z :log-evidence} (exact ys)]
    (is (:done? final))
    (is (pos? (:moves measure)) "the θ-particles were moved")
    (is (< (Math/abs (- mean exact-mean)) 0.2) (str mean " vs " exact-mean))
    (is (< (Math/abs (- (m/log-marginal measure) exact-z)) 0.4)
        (str (m/log-marginal measure) " vs " exact-z))
    ((:close final))))

(deftest seeded-smc2-reproduces
  (let [run (fn []
              (random/set-seed! 53)
              (let [root (context/create-execution-context)
                    step0 (await-cps (smc2/smc2 (binding [ec/*execution-context* root] (model true))
                                                {:params #{:s} :n-theta 10 :n-x 10 :ess-target 0.9}))
                    final (reduce (fn [step y] (await-cps ((:push step) y))) step0 (take 4 ys))
                    out [(m/log-marginal (:measure final))
                         (mapv (comp :s m/get-value first) (m/get-particles (:measure final)))]]
                ((:close final))
                out))]
    (is (= (run) (run)))))
