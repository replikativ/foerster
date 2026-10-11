(ns org.replikativ.foerster.coupling-invariants-test
  "Invariants that multilevel inference and exact enumeration rely on, and
  that performance work must keep: a draw's randomness is named by what is
  drawn (seed and address), and the common families use a fixed number of
  uniforms whatever their parameters, so two representations of one model
  share their draws site by site."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- next-after
  "The uniform drawn right after one draw of `d` on a fresh stream: equal for
  two laws iff they consumed the same number of uniforms."
  [d]
  (random/with-stream* [1 2] ::k #(do (dist/draw d) (random/uniform01))))

(deftest common-families-draw-a-fixed-number-of-uniforms
  (doseq [[label a b] [["normal" (dist/normal 0.0 1.0) (dist/normal 5.0 3.0)]
                       ["log-normal" (dist/log-normal 0.0 1.0) (dist/log-normal 2.0 0.5)]
                       ["half-normal" (dist/half-normal 1.0) (dist/half-normal 4.0)]
                       ["exponential" (dist/exponential 1.0) (dist/exponential 7.0)]
                       ["uniform" (dist/uniform 0.0 1.0) (dist/uniform -3.0 9.0)]
                       ["bernoulli" (dist/bernoulli 0.1) (dist/bernoulli 0.9)]]]
    (testing label
      (is (= (next-after a) (next-after b))))))

(defn- coarse [] (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)
                             y (sample (dist/normal x 1.0) :id :y)]
                         [x y])))

(defn- fine [] (spin (let [x (sample (dist/normal 0.0 2.0) :id :x)
                           _ (sample (dist/bernoulli 0.5) :id :extra)
                           y (sample (dist/normal (* 0.5 x) 3.0) :id :y)]
                       [x y])))

(defn- simulated [model seed]
  (itrace/choices (b/run-infer 1 #(gfi/simulate (model) {:seed seed}))))

(deftest named-sites-couple-across-representations
  ;; same session seed, same addresses: the Gaussian sites see the same
  ;; standard normal noise, so they are synchronously coupled whatever the
  ;; parameters, and a site only one representation has does not shift them
  (let [c (simulated coarse [3 4])
        f (simulated fine [3 4])
        z (fn [v mu sd] (/ (- v mu) sd))]
    (is (< (Math/abs (- (z (:x c) 0.0 1.0) (z (:x f) 0.0 2.0))) 1e-12))
    (is (< (Math/abs (- (z (:y c) (:x c) 1.0) (z (:y f) (* 0.5 (:x f)) 3.0))) 1e-12))
    (testing "and a different seed draws afresh"
      (is (not= (:x c) (:x (simulated coarse [5 6])))))))
