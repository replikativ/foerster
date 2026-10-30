(ns org.replikativ.foerster.execution
  "Shared execution of independent CPS operations. Fail-fast rejection does
  not cancel/join children; their owning inference scope supplies that lifetime."
  (:require [is.simm.partial-cps.runtime :as cps-runtime]
            [org.replikativ.spindel.engine.executor :as ex]))

(defn parallel
  "Run (start i resolve reject), preserving index order. Reject on first error,
  including a synchronous start exception; skip work not started after failure.
  executor is optional. An owner must cancel/join any already-started children."
  [executor n start]
  (when-not (nat-int? n)
    (throw (ex-info "Operation count must be nonnegative" {:type ::invalid-count :count n})))
  (fn [resolve reject]
    (if (zero? n)
      (resolve [])
      (let [out (object-array n)
            remaining (atom n)
            failed? (atom false)
            launch (fn [i]
                     (when-not @failed?
                       (let [settled? (atom false)
                             accept (fn [v]
                                      (when (compare-and-set! settled? false true)
                                        (aset out i v)
                                        (when (and (zero? (swap! remaining dec)) (not @failed?))
                                          (resolve (vec out)))))
                             fail (fn [e]
                                    (when (and (compare-and-set! settled? false true)
                                               (compare-and-set! failed? false true))
                                      (reject e)))]
                         (try
                           ;; A branch may return a Thunk under an enclosing
                           ;; trampoline; drive it to its first real suspension.
                           (loop [result (start i accept fail)]
                             (when (cps-runtime/thunk? result)
                               (recur (cps-runtime/force-thunk result))))
                           (catch #?(:clj Throwable :cljs :default) e (fail e))))))]
        (if executor (ex/spread! executor n launch)
            (dotimes [i n] (launch i)))))))
