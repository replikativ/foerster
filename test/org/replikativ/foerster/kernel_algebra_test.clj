(ns org.replikativ.foerster.kernel-algebra-test
  "Composed Markov-chain kernels against an exact posterior: z ~ Bernoulli(0.3),
  mu ~ N(2z, 1), y = 1.5 ~ N(mu, 0.5). Cycled or mixed with single-site
  moves, and nested, the random walk reaches the posterior of both sites."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.nuts :as nuts]
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

(defn- posterior [kernel seed]
  (let [wv (b/weighted-values (b/run-infer seed #(infer/kernel-infer (model) kernel 4)))]
    {:p-z (b/w-mean first wv) :mean-mu (b/w-mean second wv)}))

(deftest composed-kernels-reach-the-posterior
  (doseq [[label kernel] [["cycle" (k/cycle 1500 [(k/random-walk-mh-kernel 2 {:step-size 0.5})
                                                  (k/single-site-mh-kernel 1)]
                                            {:samples :all :burn 200})]
                          ["mixture" (k/mixture 3000 [[2 (k/random-walk-mh-kernel 1 {:step-size 0.5})]
                                                      [1 (k/single-site-mh-kernel 1)]]
                                                {:samples :all :burn 300})]
                          ["nested" (k/cycle 1000 [(k/mixture 1 [[1 (k/random-walk-mh-kernel 1 {:step-size 0.5})]
                                                                 [1 (k/single-site-mh-kernel 1)]])
                                                   (k/single-site-mh-kernel 1)]
                                             {:samples :all :burn 200})]]]
    (testing label
      (let [{:keys [p-z mean-mu]} (posterior kernel 71)]
        (is (< (Math/abs (- p-z (:p-z exact))) 0.06) (str "P(z) " p-z " vs " (:p-z exact)))
        (is (< (Math/abs (- mean-mu (:mean-mu exact))) 0.08) (str "E[mu] " mean-mu " vs " (:mean-mu exact)))))))

(deftest only-chain-kernels-compose
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Only Markov-chain kernels compose"
                        (k/cycle 10 [(k/prior-kernel)]))))

(deftest a-component-keeps-its-state-across-iterations
  ;; NUTS adapts as it runs: in a cycle its step is made once per chain, not
  ;; once per iteration of the cycle
  (let [within-gibbs nuts/within-gibbs
        made (fn [iterations]
               (let [n (atom 0)]
                 (with-redefs [nuts/within-gibbs (fn [opts] (swap! n inc) (within-gibbs opts))]
                   (b/run-infer 72 #(infer/kernel-infer (model) (k/cycle iterations [(k/nuts-kernel 1 {:burn 10})
                                                                                     (k/single-site-mh-kernel 1)])
                                                        2)))
                 @n))]
    (is (= (made 10) (made 30)))))
