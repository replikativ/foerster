(ns org.replikativ.foerster.open-kernels-test
  "Markov kernels are open and go wherever moves are made: a kernel of your
  own (`k/kernel`) runs as chains and composes with the built-in ones, and
  any fixed kernel rejuvenates SMC."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.contract :as contract]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.inference.smc :as smc]
            [org.replikativ.foerster.moves.kernel :as k]
            [org.replikativ.foerster.proposal :as proposal]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private ys [1.4 0.6 1.1 0.9 1.7 1.2 0.8 1.3])

(defn- static-model
  "mu ~ N(0, 1), y_i ~ N(mu, 1): the posterior is N(Σy/(n+1), 1/(n+1))."
  []
  (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
          (doseq [[i y] (map-indexed vector ys)]
            (observe (dist/normal mu 1.0) y :id [:y i]))
          mu)))

(def ^:private exact-mean (/ (reduce + ys) (inc (count ys))))

(defn- weighted-mean [measure]
  (b/w-mean identity (b/weighted-values measure)))

(defn- random-walk-step
  "A move of my own: MH with a random walk, honouring the context."
  [t context]
  (itrace/mh-step t (assoc context :propose (proposal/random-walk 0.5))))

(defn- error-types [f]
  (try (f) []
       (catch Throwable e
         (vec (keep #(:type (ex-data %)) (take-while some? (iterate ex-cause e)))))))

(deftest a-kernel-of-your-own-runs-as-chains-and-composes
  (let [mine (k/kernel 3000 random-walk-step {:samples :all :burn 300 :retains? true})
        chains (b/run-infer 11 #(infer/kernel-infer (static-model) mine 2))]
    (is (< (Math/abs (- (weighted-mean chains) exact-mean)) 0.1)))
  (let [mixed (k/cycle 1500 [(k/kernel 1 random-walk-step) (k/single-site-mh-kernel 1)]
                       {:samples :all :burn 150})]
    (is (< (Math/abs (- (weighted-mean (b/run-infer 12 #(infer/kernel-infer (static-model) mixed 2))) exact-mean))
           0.1))))

(deftest any-fixed-kernel-rejuvenates-smc
  (testing "a cycle of built-in kernels"
    (let [r (b/run-infer 13 #(infer/smc-infer (static-model) 400
                                              {:anchors :all
                                               :rejuvenate {:moves 2
                                                            :kernel (k/cycle 1 [(k/single-site-mh-kernel 1)
                                                                                (k/random-walk-mh-kernel 1 {:step-size 0.4})])}}))]
      (is (< (Math/abs (- (weighted-mean r) exact-mean)) 0.1))
      (is (pos? (get-in r [:rejuvenation :moves])))
      (is (= :unbiased (contract/normalizer-status r)) "built-in kernels keep the evidence unbiased")))
  (testing "a kernel of your own is your obligation"
    (let [r (b/run-infer 14 #(infer/smc-infer (static-model) 200
                                              {:anchors :all
                                               :rejuvenate {:kernel (k/kernel 1 random-walk-step {:retains? true})}}))]
      (is (< (Math/abs (- (weighted-mean r) exact-mean)) 0.15))
      (is (= :unverified (contract/normalizer-status r))))))

(deftest a-rejuvenation-kernel-is-refused-when-it-cannot-be-exact
  (let [run #(b/run-infer 15 (fn [] (infer/smc-infer (static-model) 20 (merge {:anchors :all} %))))]
    (is (some #{::smc/invalid-rejuvenation}
              (error-types #(run {:rejuvenate {:kernel (k/random-walk-mh-kernel 1 {:burn 10})}})))
        "a kernel that adapts")
    (is (some #{::smc/invalid-rejuvenation}
              (error-types #(run {:rejuvenate {:kernel (k/single-site-mh-kernel 1) :propose proposal/prior}})))
        "a kernel and a proposal")
    (is (some #{::smc/invalid-rejuvenation}
              (error-types #(run {:rejuvenate {:kernel (k/kernel 1 random-walk-step)}})))
        "a custom kernel that does not declare it shares anchors")
    (is (some #{::smc/invalid-rejuvenation}
              (error-types #(b/run-infer 16 (fn [] (infer/pgibbs-infer (static-model) 10 2
                                                                       {:anchors :all
                                                                        :rejuvenate {:kernel (k/single-site-mh-kernel 1)}})))))
        "conditional SMC needs reversible moves")))
