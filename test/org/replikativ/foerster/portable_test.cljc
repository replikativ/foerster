(ns org.replikativ.foerster.portable-test
  "Inference runs in both runtimes: savepoint SMC on a conjugate model."
  (:refer-clojure :exclude [await])
  (:require #?(:clj [clojure.test :refer [deftest is]]
               :cljs [cljs.test :refer-macros [deftest is async]])
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]])
  #?(:cljs (:require-macros [org.replikativ.spindel.spin.cps :refer [spin]])))

(defn- run-spin
  "Run `task` (a spin) in a fresh context; `check` gets its result."
  [task-fn check done]
  (let [context (ctx/create-execution-context)
        finish (fn [outcome] (check outcome) (done))]
    (binding [ec/*execution-context* context]
      ((task-fn) finish finish))))

(defn- conjugate-model []
  ;; μ ~ N(0, 1), y ~ N(μ, 1), y = 1: posterior mean 0.5
  (spin
   (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
     (observe (dist/normal mu 1.0) 1.0 :id :y)
     mu)))

(defn- check-posterior [measure]
  (is (= 200 (count (m/get-particles measure))))
  (is (< (Math/abs (- 0.5 (:mean (infer/query measure identity)))) 0.2)))

(deftest smc-on-a-conjugate-model
  #?(:clj (let [p (promise)]
            (run-spin #(infer/smc-infer (conjugate-model) 200) #(deliver p %) (fn []))
            (check-posterior (deref p 30000 ::timeout)))
     :cljs (async done
                  (run-spin #(infer/smc-infer (conjugate-model) 200) check-posterior done))))

(deftest measures-are-portable
  (let [measure (m/empirical [[(m/sample-particle :a {}) (Math/log 1.0)]
                              [(m/sample-particle :b {}) (Math/log 3.0)]])]
    (is (= [0.25 0.75] (mapv #(/ (Math/round (* 100 %)) 100.0)
                             (m/normalize-log-weights (mapv second (m/get-particles measure))))))
    (is (< (Math/abs (- (m/effective-sample-size measure) 1.6)) 1e-9))))
