(ns org.replikativ.foerster.inference-failure-test
  "Regression test: a particle's spin failure must resolve the inference
  (re-throwing the failure to the caller) — NOT hang forever waiting for the
  other particles. Observed in production (on the retired coordinator) as
  14k leaked threads and parked dispatch threads."
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.spindel.v2.core :as v2 :refer [task]]
            [org.replikativ.foerster.runtime.async :refer [await-op]]
            
            ))

(defn clean-context-fixture [f]
  (binding [v2/*world* nil]
    (f)))

(use-fixtures :each clean-context-fixture)

(deftest test-particle-failure-resolves-not-hangs
  (testing "When every particle's spin throws, smc-infer must propagate
            the failure to the caller within a bounded time — not hang
            forever waiting for completion notifications that will
            never arrive."
    (let [context (v2/world)]
      (try
        (binding [v2/*world* context]
          (let [;; A model whose spin body throws — every particle fails
            ;; identically. This mirrors the production failure mode
            ;; (a broken model where `sample` runs outside spin context
            ;; and throws on every particle).
                failing-model (fn [] (task
                                      (throw (ex-info "model went boom"
                                                      {:reason :regression-test}))))

                outer (task
                       (try
                         (let [m (await-op (infer/smc-infer failing-model 3))]
                        ;; Should never reach here — the inference must
                        ;; throw the InferenceFailure rather than return
                        ;; a measure when every particle errored.
                           {:status :unexpected-success :measure m})
                         (catch Throwable t
                           {:status :threw
                            :message (ex-message t)
                            :cause-message (some-> t ex-cause ex-message)})))

        ;; Bounded wall-clock: a regression that re-introduces the hang
        ;; would block on the inference forever. 5s is generous — the
        ;; expected path is "fail fast, the moment any particle throws"
        ;; (well under 1s with 3 particles).
                result (v2/value outer 5000 ::timed-out)]
            (is (not= ::timed-out result)
                "smc-infer must NOT hang when particles fail")
            (when (not= ::timed-out result)
              (is (= :threw (:status result))
                  (str "outer spin must propagate the inference failure, got: "
                       (pr-str result)))
              (is (re-find #"Inference failed" (str (:message result)))
                  "thrown error message identifies it as an inference failure")
              (is (re-find #"boom" (str (:cause-message result)))
                  "the original model error is preserved as the cause"))))
        (finally (v2/release-world! context))))))
