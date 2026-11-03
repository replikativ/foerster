(ns org.replikativ.foerster.consistency-test
  "Options are checked where they are given, and results' contracts are
  read where they matter: site options, proposals, PMMH's options and its
  filter's evidence, nested inference."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.contract :as contract]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :as effects :refer [observe sample]]
            [org.replikativ.foerster.inference.smc2 :as smc2]
            [org.replikativ.foerster.proposal :as proposal]
            [org.replikativ.foerster.runtime.trace :as itrace]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- line []
  (spin (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
          (observe (dist/normal mu 0.5) 0.8 :id :y)
          mu)))

(defn- coin []
  (spin (let [c (sample (dist/flip 0.3) :id :c)]
          (observe (dist/normal (if c 1.0 0.0) 1.0) 0.5 :id :y)
          c)))

(defn- error-types
  "The `:type`s along an error's cause chain."
  [f]
  (try (f) []
       (catch Throwable e
         (vec (keep #(:type (ex-data %)) (take-while some? (iterate ex-cause e)))))))

(deftest a-misspelt-site-option-is-refused
  (is (some #{::effects/invalid-site-options}
            (error-types #(b/run-infer 1 (fn [] (infer/infer (spin (sample (dist/normal 0.0 1.0) :id :x
                                                                           :proposl (dist/normal 0.0 1.0)))
                                                             {:method :smc :particles 2}))))))
  (is (some #{::effects/invalid-site-options}
            (error-types #(b/run-infer 1 (fn [] (infer/infer (spin (observe (dist/normal 0.0 1.0) 0.1 :idd :y))
                                                             {:method :smc :particles 2}))))))
  (is (= :ok (do (b/run-infer 1 #(infer/infer (spin (sample (dist/normal 0.0 1.0) :id :x
                                                            :proposal (dist/normal 0.1 1.0)))
                                              {:method :smc :particles 2}))
                 :ok))))

(deftest a-policy-that-only-fixes-sites-keeps-smc-unbiased
  (let [status #(contract/normalizer-status (b/run-infer 2 (fn [] (infer/smc-infer (line) 10 {:policy %}))))]
    (is (= :unbiased (status (itrace/policy {:constraints {:mu 0.3}}))))
    (is (= :unbiased (status (itrace/policy {:interventions {:mu {:do 0.3}}}))))
    (is (= :unverified (status (itrace/policy {:draw (fn [_ _] nil)}))) "a policy that proposes")
    (is (= :unverified (status (let [p (itrace/policy {:constraints {:mu 0.3}})] (fn [sp old] (p sp old)))))
        "a policy function whose options are unknown")))

(deftest proposals-go-where-sites-are-proposed
  (let [q (proposal/distribution (dist/normal 0.5 1.0))
        status #(contract/normalizer-status (b/run-infer 5 (fn [] (infer/smc-infer (line) 10 %))))]
    (testing "a built-in proposal keeps the evidence unbiased; a function proposal is the caller's"
      (is (= :unbiased (status {:proposal q})))
      (is (= :unbiased (status {:proposal q :policy (itrace/policy {:constraints {:y 0.8}})})))
      (is (= :unverified (status {:proposal (proposal/from-fn (fn [{d :dist}]
                                                                (let [v (dist/draw d)] {:value v :log-q (dist/logpdf d v)})))}))))
    (testing "one proposer per run"
      (is (some #{::itrace/invalid-proposal}
                (error-types #(b/run-infer 5 (fn [] (infer/smc-infer (line) 4 {:proposal q
                                                                               :policy (itrace/policy {:draw (fn [_ _] nil)})})))))))
    (testing "a proposal is not a Markov kernel"
      (is (= [::infer/not-a-markov-kernel] (error-types #(b/run-infer 5 (fn [] (infer/kernel-infer (line) q 4))))))
      (is (= [::infer/unknown-options] (error-types #(b/run-infer 5 (fn [] (infer/infer (line) {:method :mh :iterations 5
                                                                                                :proposal q})))))
          "chains take their proposals in their kernels"))))

(deftest pmmh-checks-its-options-and-its-filter
  (let [run #(b/run-infer 3 (fn [] (infer/infer (line) (merge {:method :pmmh :params #{:mu} :particles 8 :iterations 4} %))))]
    (testing "a misspelt option"
      (is (= [::infer/unknown-options] (error-types #(run {:scael 0.2})))))
    (testing "a filter whose evidence is not unbiased"
      (is (some #{::smc2/unverified-evidence}
                (error-types #(run {:policy (itrace/policy {:draw (fn [_ _] nil)})}))))
      (is (= 4 (count (:particles (run {:policy (itrace/policy {:draw (fn [_ _] nil)})
                                        :allow-unverified-evidence? true}))))
          "accepted when the caller says so"))
    (is (= 4 (count (:particles (run {})))) "θ's constraints keep the default filter unbiased")))

(deftest nested-inference-keeps-the-inner-contract
  (let [d (b/run-infer 4 #(infer/conditional (coin) {:method :enumerate}))]
    (is (= :exact (get-in (meta d) [:inference/nesting :contract :normalizer :status])))))
