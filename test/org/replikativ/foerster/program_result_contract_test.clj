(ns org.replikativ.foerster.program-result-contract-test
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.program :as program]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(deftest a-custom-program-returns-a-queryable-measure-directly
  (let [root (context/create-execution-context)]
    (try
      (binding [engine/*execution-context* root]
        (let [prior (d/categorical {2.0 1.0})
              target (program/value-target #(d/logpdf prior %) #(d/draw prior) #(d/logpdf prior %))
              result (promise)]
          ((program/run {:seed 29}
                        (fn [scope]
                          (spin (let [population (await (program/initialize scope target 4))]
                                  (program/finish scope population)))))
           #(deliver result [:ok %]) #(deliver result [:error %]))
          (let [[status measure] (deref result 10000 [::timeout nil])]
            (is (= :ok status) (str measure))
            (when (= :ok status)
              (is (satisfies? m/PMeasure measure))
              (is (nil? (:measure measure)) "the standard result is the measure itself")
              (is (= 4 (count (m/get-particles measure))))
              (is (= 2.0 (:mean (infer/query measure identity))))))))
      (finally (context/stop-context! root)))))
