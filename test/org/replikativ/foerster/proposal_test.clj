(ns org.replikativ.foerster.proposal-test
  "A sample site's `:proposal`: draws come from the guide, weights restore
  the prior. mu ~ N(0, 1), y = 1 ~ N(mu, 0.5): the posterior is N(0.8, 0.2),
  the evidence N(1; 0, √1.25)."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- model [proposal]
  (spin
   (let [mu (if proposal
              (sample (dist/normal 0.0 1.0) :id :mu :proposal proposal)
              (sample (dist/normal 0.0 1.0) :id :mu))]
     (observe (dist/normal mu 0.5) 1.0 :id :y)
     mu)))

(def ^:private log-evidence (dist/logpdf (dist/normal 0.0 (Math/sqrt 1.25)) 1.0))

(deftest a-guide-keeps-the-posterior-and-the-evidence
  (let [prior (b/run-infer 61 #(infer/importance-sampling (model nil) 2000))
        guided (b/run-infer 61 #(infer/importance-sampling (model (dist/normal 0.8 0.6)) 2000))]
    (doseq [[label measure] [["prior" prior] ["guide" guided]]]
      (testing label
        (let [[mean sd] (b/w-mean-sd identity (b/weighted-values measure))]
          (is (< (Math/abs (- mean 0.8)) 0.05) (str mean))
          (is (< (Math/abs (- sd (Math/sqrt 0.2))) 0.05) (str sd))
          (is (< (Math/abs (- (m/log-marginal measure) log-evidence)) 0.05)
              (str (m/log-marginal measure) " vs " log-evidence)))))
    (is (> (m/effective-sample-size guided) (* 2 (m/effective-sample-size prior)))
        (str "ESS " (m/effective-sample-size guided) " vs " (m/effective-sample-size prior)))))

(deftest a-markov-chain-over-a-guided-site
  (let [measure (b/run-infer 62 #(infer/kernel-infer (model (dist/normal 2.0 0.3))
                                                     (k/single-site-mh-kernel 400 {:samples :all :burn 100})
                                                     4))
        [mean sd] (b/w-mean-sd identity (b/weighted-values measure))]
    (is (< (Math/abs (- mean 0.8)) 0.1) (str mean))
    (is (< (Math/abs (- sd (Math/sqrt 0.2))) 0.1) (str sd))))
