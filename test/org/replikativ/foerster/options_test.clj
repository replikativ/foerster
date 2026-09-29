(ns org.replikativ.foerster.options-test
  "An option an inference would ignore is refused: a misspelled one, one the
  method does not take, and a canonical-world option in fresh worlds."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private world (context/create-execution-context))

(defn- model [] (binding [ec/*execution-context* world] (spin :done)))

(defn- refusal [f]
  (try (f) nil
       (catch clojure.lang.ExceptionInfo e (:type (ex-data e)))))

(deftest unknown-options-are-refused
  (let [model (model)]
    (is (= ::infer/unknown-options
           (refusal #(infer/smc-infer model 10 {:resample-treshold 0.3}))) "misspelled")
    (is (= ::infer/unknown-options
           (refusal #(infer/importance-sampling model 10 {:resample-threshold 0.3})))
        "importance sampling never resamples")
    (is (= ::infer/unknown-options
           (refusal #(infer/bbvi-infer model 10 1 {:policy nil}))) "BBVI decides its sites")
    (is (= ::infer/unknown-options
           (refusal #(infer/kernel-infer model (k/random-walk-mh-kernel 10 {}) 1
                                         {:resample-threshold 0.5})))
        "a Markov chain does not resample")))

(deftest canonical-world-options-need-canonical-worlds
  (let [model (model)]
    (doseq [opts [{:authority :some-authority} {:grant {:tokens 1}} {:world-opts {}}
                  {:world-policy :fresh :grant {:tokens 1}}]]
      (is (= ::infer/fork-only-options (refusal #(infer/smc-infer model 10 opts))) (str opts)))))
