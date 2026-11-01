(ns org.replikativ.foerster.chain-warmup-test
  "A component kernel's warm-up inside `cycle`/`mixture` is not emitted as
  posterior draws, and a chain that ends during it is refused."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- model []
  (spin
   (let [z (sample (dist/bernoulli 0.3) :id :z)
         mu (sample (dist/normal (* 2.0 z) 1.0) :id :mu)]
     (observe (dist/normal mu 0.5) 1.5 :id :y)
     [z mu])))

(deftest nested-warm-up-is-not-emitted
  (testing "an adaptive leaf with :burn 100 in a cycle of 300 rounds: 200 draws"
    (let [kernel (k/cycle 300 [(k/random-walk-mh-kernel 1 {:step-size 0.5 :burn 100})
                               (k/single-site-mh-kernel 1)]
                          {:samples :all})
          measure (b/run-infer 81 #(infer/kernel-infer (model) kernel 1))]
      (is (= 200 (count (m/get-particles measure))))))
  (testing "the outer :burn and the leaf's warm-up: whichever ends later"
    (let [kernel (k/cycle 300 [(k/random-walk-mh-kernel 1 {:step-size 0.5 :burn 100})
                               (k/single-site-mh-kernel 1)]
                          {:samples :all :burn 150})
          measure (b/run-infer 82 #(infer/kernel-infer (model) kernel 1))]
      (is (= 150 (count (m/get-particles measure))))))
  (testing "a non-adaptive component's :burn is not a warm-up"
    (let [kernel (k/cycle 300 [(k/single-site-mh-kernel 1 {:burn 100})
                               (k/single-site-mh-kernel 1)]
                          {:samples :all})
          measure (b/run-infer 84 #(infer/kernel-infer (model) kernel 1))]
      (is (= 300 (count (m/get-particles measure))))))
  (testing "a chain that ends during a component's warm-up is refused"
    (let [kernel (k/cycle 50 [(k/random-walk-mh-kernel 1 {:step-size 0.5 :burn 100})
                              (k/single-site-mh-kernel 1)])
          outcome (try (b/run-infer 83 #(infer/kernel-infer (model) kernel 1))
                       (catch Exception e e))]
      (is (= :org.replikativ.foerster.core/warm-up-unfinished
             (some (comp :type ex-data) (take-while some? (iterate ex-cause outcome))))))))
