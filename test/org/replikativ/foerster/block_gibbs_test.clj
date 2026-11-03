(ns org.replikativ.foerster.block-gibbs-test
  "Block Gibbs against an exact posterior, and the conditions it needs:
  moves that keep the block's membership, symmetric block kernels and
  real-valued random-walk targets."
  (:require [clojure.test :refer [deftest is]]
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

;; p(z | y) ∝ p(z)·N(1.5; 2z, √1.25); E[mu | z, y] = (2z + 1.5·4)/5
(def ^:private exact
  (let [lik (fn [z] (Math/exp (dist/logpdf (dist/normal (* 2.0 z) (Math/sqrt 1.25)) 1.5)))
        w1 (* 0.3 (lik 1)) w0 (* 0.7 (lik 0))
        p1 (/ w1 (+ w0 w1))]
    {:p-z p1 :mean-mu (+ (* p1 (/ 8.0 5)) (* (- 1 p1) (/ 6.0 5)))}))

(deftest block-gibbs-reaches-the-posterior
  (let [kernel (k/block-gibbs-kernel 4000 (k/round-robin-selector [:cont :disc])
                                     {:cont (k/random-walk-block-kernel {:step-size 0.6})
                                      :disc (k/prior-block-kernel)}
                                     (fn [address _] (if (= address :mu) :cont :disc))
                                     {:samples :all :burn 500})
        wv (b/weighted-values (b/run-infer 101 #(infer/kernel-infer (model) kernel 4)))
        p-z (b/w-mean first wv)
        mean-mu (b/w-mean second wv)]
    (is (< (Math/abs (- p-z (:p-z exact))) 0.05) (str "P(z) " p-z " vs " (:p-z exact)))
    (is (< (Math/abs (- mean-mu (:mean-mu exact))) 0.07) (str "E[mu] " mean-mu " vs " (:mean-mu exact)))))

(defn- signed []
  (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
          (observe (dist/normal x 1.0) 0.0 :id :y)
          x)))

(deftest a-move-that-changes-the-block-is-rejected
  ;; x belongs to block :pos while positive: a move across zero would leave
  ;; the block, and its reverse would move another block, so it is rejected
  (let [kernel (k/block-gibbs-kernel 400 (k/round-robin-selector [:pos])
                                     {:pos (k/random-walk-block-kernel {:step-size 2.0})}
                                     (fn [_ entry] (if (pos? (:value entry)) :pos :neg))
                                     {:samples :all})
        start #(spin (let [x (sample (dist/normal 0.0 1.0) :id :x :init 0.5)]
                       (observe (dist/normal x 1.0) 0.0 :id :y)
                       x))
        xs (map m/get-value (map first (m/get-particles (b/run-infer 102 #(infer/kernel-infer (start) kernel 1)))))]
    (is (seq xs))
    (is (every? pos? xs) "the chain never leaves the block it moves")))

(defrecord ^:private Lopsided []
  k/PBlockKernel
  (propose-block [_ trace addresses]
    (into {} (map (fn [a] [a (+ (:value (get trace a)) 0.1)])) addresses)))

(deftest block-kernels-must-be-symmetric-or-the-prior
  (is (= ::k/asymmetric-block-kernel
         (:type (ex-data (try (k/block-gibbs-kernel 10 (k/round-robin-selector [:a])
                                                    {:a (->Lopsided)} (constantly :a))
                              (catch Exception e e)))))))

(deftest a-random-walk-block-refuses-discrete-sites
  (let [kernel (k/block-gibbs-kernel 10 (k/round-robin-selector [:all])
                                     {:all (k/random-walk-block-kernel {:step-size 0.5})}
                                     (constantly :all))
        outcome (try (b/run-infer 103 #(infer/kernel-infer (model) kernel 1))
                     (catch Exception e e))]
    (is (= ::k/not-real-valued
           (some (comp :type ex-data) (take-while some? (iterate ex-cause outcome)))))))

(deftest signed-model-is-well-formed
  ;; guard for the model above: a plain chain crosses zero
  (let [xs (map m/get-value (map first (m/get-particles
                                        (b/run-infer 104 #(infer/kernel-infer (signed) (k/random-walk-mh-kernel 400 {:step-size 2.0 :samples :all}) 1)))))]
    (is (some neg? xs))))
