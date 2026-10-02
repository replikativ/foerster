(ns org.replikativ.foerster.reversible-jump-test
  "Reversible jump by involutive MCMC: one component or two, against the
  exact posterior over the number of components.

    k ~ uniform{1, 2};  k = 1: μ ~ N(0, 1), y ~ N(μ, 1)
                        k = 2: μ₁, μ₂ ~ N(0, 1), y ~ N(μ₁ + μ₂, 1)
  so y | k=1 ~ N(0, √2), y | k=2 ~ N(0, √3). Split: μ₁ = μ/2 + u,
  μ₂ = μ/2 − u with u ~ N(0, 1); merge: μ = μ₁ + μ₂, u = (μ₁ − μ₂)/2; the
  Jacobian of the split is 1 (|det [[½ 1] [½ −1]]| = 1), and of the merge
  its inverse."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.involutive :as involutive]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private y 2.5)

(defn- model []
  (spin
   (let [k (sample (dist/uniform-discrete 1 3) :id :k)]
     (if (= k 1)
       (observe (dist/normal (sample (dist/normal 0.0 1.0) :id :mu) 1.0) y :id :y)
       (observe (dist/normal (+ (sample (dist/normal 0.0 1.0) :id [:mu 1])
                                (sample (dist/normal 0.0 1.0) :id [:mu 2]))
                             1.0)
                y :id :y))
     k)))

(def ^:private split-merge
  {:propose (fn [x] (if (= 1 (:k x))
                      (let [u (dist/draw (dist/normal 0.0 1.0))] {:aux u :log-q (dist/logpdf (dist/normal 0.0 1.0) u)})
                      {:aux nil :log-q 0.0}))
   :log-q (fn [x u] (if (= 1 (:k x)) (dist/logpdf (dist/normal 0.0 1.0) u) 0.0))
   :involution (fn [x u]
                 (if (= 1 (:k x))
                   {:choices {:k 2 [:mu 1] (+ (/ (:mu x) 2) u) [:mu 2] (- (/ (:mu x) 2) u)}
                    :aux nil :log-jacobian 0.0 :removed #{:mu}}
                   (let [m1 (get x [:mu 1]) m2 (get x [:mu 2])]
                     {:choices {:k 1 :mu (+ m1 m2)}
                      :aux (/ (- m1 m2) 2) :log-jacobian 0.0 :removed #{[:mu 1] [:mu 2]}})))})

(defn- await* [op]
  (let [p (promise)] (op #(deliver p %) #(deliver p %))
       (let [v (deref p 60000 ::timeout)] (if (instance? Throwable v) (throw v) v))))

(deftest split-and-merge-reach-the-model-posterior
  (random/set-seed! 131)
  (let [root (ctx/create-execution-context)
        exact (let [l1 (Math/exp (dist/logpdf (dist/normal 0.0 (Math/sqrt 2.0)) y))
                    l2 (Math/exp (dist/logpdf (dist/normal 0.0 (Math/sqrt 3.0)) y))]
                (/ l2 (+ l1 l2)))]
    (try
      (binding [rtc/*execution-context* root]
        (let [t0 (:trace (await* (gfi/generate (model) {:k 1 :mu 0.0})))
              n 6000 burn 500
              ks (loop [i 0 t t0 out [] jumps 0]
                   (if (= i n)
                     (do (is (< 300 jumps) (str jumps " jumps")) out)
                     (let [{t1 :trace jumped? :accepted?} (await* (involutive/step t split-merge))
                           ;; and a within-model move of every continuous site
                           {t2 :trace} (await* (itrace/mh-chain t1 2 {:propose (itrace/random-walk-proposal 0.8) :first-iteration (* 2 i)}))]
                       (recur (inc i) t2 (if (>= i burn) (conj out (:trace/result t2)) out)
                              (if jumped? (inc jumps) jumps)))))
              p2 (/ (count (filter #(= 2 %) ks)) (double (count ks)))]
          (is (< (Math/abs (- p2 exact)) 0.04) (str "P(k = 2) " p2 " vs " exact))))
      (finally (ctx/stop-context! root)))))
