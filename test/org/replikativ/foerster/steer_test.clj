(ns org.replikativ.foerster.steer-test
  "Steering a stochastic process by SMC against exact answers: five coin
  steps (randomness without a sample site, like a language model's output)
  tilted by exp(λ·heads). Twisting with the partial reward and plain
  best-of-N weighting reach the same exact posterior and evidence, with a
  barrier at every step."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.steer :as steer]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private lambda 0.8)
(def ^:private steps 5)

(defn- coin-step
  "A 'model' step: a fair coin drawn outside any sample site."
  [heads]
  (spin (+ heads (if (< (random/uniform01) 0.5) 1 0))))

(defn- process [twisted?]
  (steer/model (cond-> {:init 0
                        :step coin-step
                        :done? (constantly false)
                        :max-steps steps
                        :reward (fn [heads] (* lambda heads))}
                 twisted? (assoc :value (fn [heads] (* lambda heads))))))

;; heads ~ Binomial(5, ½) tilted by e^{λ·heads}: Binomial(5, e^λ/(1+e^λ)),
;; evidence ((1 + e^λ)/2)^5
(def ^:private p (/ (Math/exp lambda) (+ 1 (Math/exp lambda))))
(def ^:private exact-mean (* steps p))
(def ^:private exact-log-evidence (* steps (Math/log (/ (+ 1 (Math/exp lambda)) 2))))

(deftest twisted-and-untwisted-steering-agree-with-the-tilted-law
  (doseq [twisted? [true false]]
    (testing (if twisted? "twisted" "best-of-N")
      (let [measure (b/run-infer 41 #(smc/smc (process twisted?) 2000 {:resampling :stratified}))
            mean (b/w-mean identity (b/weighted-values measure))]
        (is (< (Math/abs (- mean exact-mean)) 0.1) (str mean " vs " exact-mean))
        (is (< (Math/abs (- (m/log-marginal measure) exact-log-evidence)) 0.05)
            (str (m/log-marginal measure) " vs " exact-log-evidence))
        (is (= (dec steps) (count (:history measure))) "a barrier at every step but the last")))))

(deftest the-trajectory-is-in-the-trace
  (let [measure (b/run-infer 42 #(smc/smc (process true) 20 {}))
        [particle] (first (m/get-particles measure))]
    (is (= (m/get-value particle) (m/site-value particle [:steer/state (dec steps)])))
    (is (every? #(some? (m/site-value particle [:steer/state %])) (range steps)))))
