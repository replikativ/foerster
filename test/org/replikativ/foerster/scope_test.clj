(ns org.replikativ.foerster.scope-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.scope :as scope]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- completion [operation]
  (let [p (promise)]
    (operation #(deliver p [:ok %]) #(deliver p [:error %]))
    p))

(deftest close-waits-for-an-acquisition-already-in-flight
  (let [s (scope/create) entered (promise) proceed (promise) released (atom [])
        acquisition (future (scope/acquire! s #(do (deliver entered true) @proceed :resource)
                                            (fn [r] (fn [resolve _] (swap! released conj r) (resolve nil)))))]
    (is (= true (deref entered 5000 ::timeout)))
    (let [closed (completion (scope/close! s))]
      (is (= ::waiting (deref closed 10 ::waiting)))
      (is (thrown? clojure.lang.ExceptionInfo (scope/acquire! s (constantly :late) identity)))
      (deliver proceed true)
      (is (= :resource (deref acquisition 5000 ::timeout)))
      (is (= [:ok nil] (deref closed 5000 ::timeout)))
      (is (= [:resource] @released))
      (is (= [:ok nil] @(completion (scope/close! s)))))))

(deftest all-resources-are-released-after-a-cleanup-failure
  (let [s (scope/create) released (atom []) bad (ex-info "bad release" {})]
    (doseq [r [:first :second :third]]
      (scope/acquire! s (constantly r)
                      (fn [x] (fn [resolve reject]
                                (swap! released conj x)
                                (if (= x :second) (reject bad) (resolve nil))))))
    (let [[tag error] @(completion (scope/close! s))]
      (is (= :error tag))
      (is (= [:third :second :first] @released))
      (is (= [bad] (:errors (ex-data error))))
      (is (identical? error (second @(completion (scope/close! s))))))))

(deftest inference-failure-remains-the-cause-if-cleanup-fails
  (binding [ec/*execution-context* (ctx/create-execution-context)]
    (let [original (ex-info "model failure" {}) cleanup (ex-info "cleanup failure" {})
          [_ error] @(completion
                      (scope/run (fn [s]
                                   (scope/acquire! s (constantly :resource)
                                                   (fn [_] (fn [_ reject] (reject cleanup))))
                                   (spin (throw original)))))]
      (is (identical? original (ex-cause error)))
      (is (= ::scope/cleanup-failed (:type (ex-data error)))))))

(deftest cancellation-awaits-asynchronous-resource-finalization
  (let [root (ctx/create-execution-context)
        entered (promise) releasing (promise) proceed (promise)
        task* (atom nil)
        outcome (future
                  (binding [ec/*execution-context* root]
                    (let [task (scope/run
                                (fn [s]
                                  (scope/acquire! s (constantly :resource)
                                                  (fn [_]
                                                    (fn [resolve _]
                                                      (deliver releasing true)
                                                      (future
                                                        @proceed
                                                        (binding [is.simm.partial-cps.async/*in-trampoline* false]
                                                          (resolve nil))))))
                                  (spin
                                   (deliver entered true)
                                   (org.replikativ.spindel.effects.await/await (fn [_ _] nil)))))]
                      (reset! task* task)
                      (try @task (catch Throwable e e)))))]
    (try
      (is (= true (deref entered 5000 ::timeout)))
      (binding [ec/*execution-context* root]
        (org.replikativ.spindel.spin.core/cancel-spin! @task*))
      (is (= true (deref releasing 5000 ::timeout)))
      (is (= ::waiting (deref outcome 20 ::waiting)))
      (deliver proceed true)
      (let [error (deref outcome 5000 ::timeout)]
        (is (instance? Throwable error))
        (is (= org.replikativ.spindel.spin.core/spin-cancelled (:type (ex-data error)))))
      (finally (deliver proceed true) (ctx/close-context! root)))))

(deftest thousands-of-synchronous-releases-are-stack-safe-and-exactly-once
  (let [s (scope/create) released (atom [])]
    (dotimes [i 10000]
      (scope/acquire! s (constantly i)
                      (fn [x] (fn [resolve _] (swap! released conj x) (resolve nil)))))
    (is (= [:ok nil] (deref (completion (scope/close! s)) 10000 ::timeout)))
    (is (= (vec (reverse (range 10000))) @released))))
