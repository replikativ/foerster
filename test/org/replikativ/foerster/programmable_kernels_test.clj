(ns org.replikativ.foerster.programmable-kernels-test
  "Involutive and proposal-program moves as Markov-chain kernels: they nest
  in `cycle` with the built-in kernels and reach an exact posterior; a
  population program refuses them."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.program :as ip]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private y 2.0)

(defn- switch
  "b ~ flip(1/2); v0 ~ N(0,1) drawn from its site proposal N(3,1) when b,
  else v1 ~ N(5,1); y = 2 ~ N(v,1)."
  []
  (spin (let [b (sample (dist/flip 0.5) :id :b)
              v (if b
                  (sample (dist/normal 0.0 1.0) :id :v0 :proposal (dist/normal 3.0 1.0))
                  (sample (dist/normal 5.0 1.0) :id :v1))]
          (observe (dist/normal v 1.0) y :id :y)
          b)))

(def ^:private exact-p-b
  (let [a (Math/exp (dist/logpdf (dist/normal 0.0 (Math/sqrt 2.0)) y))
        c (Math/exp (dist/logpdf (dist/normal 5.0 (Math/sqrt 2.0)) y))]
    (/ a (+ a c))))

(def ^:private flip-move
  {:propose (fn [_] {:aux nil :log-q 0.0})
   :log-q (fn [_ _] 0.0)
   :involution (fn [{b :b} _] {:choices {:b (not b)} :aux nil :log-jacobian 0.0})})

(defn- flip-b [{b :b}]
  (spin (sample (dist/flip (if b 0.0 1.0)) :id :b)))

(defn- p-b [kernel seed]
  (let [wv (b/weighted-values (b/run-infer seed #(infer/kernel-infer (switch) kernel 2)))]
    (b/w-mean #(if % 1.0 0.0) wv)))

(deftest programmable-moves-compose-with-chain-kernels
  (testing "an involutive move cycled with single-site MH"
    (let [p (p-b (k/cycle 4000 [(k/involutive-kernel 1 flip-move) (k/single-site-mh-kernel 1)]
                          {:samples :all :burn 300})
                 121)]
      (is (< (Math/abs (- p exact-p-b)) 0.04) (str "P(b) " p " exact " exact-p-b))))
  (testing "a proposal program in a mixture with single-site MH"
    (let [p (p-b (k/mixture 8000 [[1 (k/proposal-mh-kernel 1 flip-b)] [1 (k/single-site-mh-kernel 1)]]
                            {:samples :all :burn 300})
                 122)]
      (is (< (Math/abs (- p exact-p-b)) 0.04) (str "P(b) " p " exact " exact-p-b)))))

(deftest populations-refuse-moves-that-give-back-their-state
  (is (= ::ip/unretaining-population-kernel
         (some (comp :type ex-data)
               (take-while some?
                           (iterate ex-cause
                                    (try (b/run-infer 123 #(ip/run {:seed 123}
                                                                   (fn [scope]
                                                                     (spin
                                                                      (let [p (await (ip/initialize scope (ip/trace-target (switch)) 8))
                                                                            p (await (ip/move scope p (k/involutive-kernel 1 flip-move)))]
                                                                        (ip/finish scope p))))))
                                         (catch Exception e e))))))))

(deftest a-refused-move-leaves-the-population-usable
  (let [result (b/run-infer 124 #(ip/run {:seed 124}
                                         (fn [scope]
                                           (spin
                                            (let [p (await (ip/initialize scope (ip/trace-target (switch)) 8))
                                                  refused (try (await (ip/move scope p (k/involutive-kernel 1 flip-move)))
                                                               nil
                                                               (catch Exception e (:type (ex-data e))))
                                                  p (await (ip/move scope p (k/single-site-mh-kernel 1)))]
                                              (assoc (ip/finish scope p) ::refused refused))))))]
    (is (= ::ip/unretaining-population-kernel (::refused result)))
    (is (= 8 (count (m/get-particles result))))))

(deftest proposal-kernel-chains-reproduce-under-a-seed
  (let [kernel (k/proposal-mh-kernel 50 flip-b {:samples :all})
        run (fn [process-seed]
              (mapv (comp m/get-value first)
                    (m/get-particles (b/run-infer process-seed #(infer/kernel-infer (switch) kernel 2 {:seed 7})))))]
    (is (= (run 1) (run 2)))))
