(ns org.replikativ.foerster.transport-test
  (:refer-clojure :exclude [await])
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.program :as ip]
            [org.replikativ.foerster.transport :as transport]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(deftest affine-transport-includes-its-jacobian-and-preserves-inputs
  (let [normal (d/normal 0.0 1.0)
        wider (d/normal 0.0 2.0)
        source (ip/value-target #(d/logpdf normal (aget ^doubles % 0))
                                #(double-array [(d/draw normal)])
                                #(d/logpdf normal (aget ^doubles % 0))
                                {:copy aclone :project #(aget ^doubles % 0)})
        destination (ip/retarget source :wider #(d/logpdf wider (aget ^doubles % 0)))
        kernel (ip/transport-kernel source destination
                                    (fn [x] (aset-double x 0 (* 2.0 (aget ^doubles x 0)))
                                      {:value x :log-q 0.0 :reverse nil :log-jacobian (Math/log 2.0)})
                                    (fn [_ _] 0.0))
        original (atom nil) snapshot (atom nil)
        result (b/run-infer 91 #(ip/run {:seed 91}
                                        (fn [scope]
                                          (spin
                                           (let [p (await (ip/initialize scope source 16))
                                                 _ (reset! original (:particles p))
                                                 _ (reset! snapshot (mapv vec (:particles p)))
                                                 p (await (ip/transport scope p destination kernel))]
                                             (ip/finish scope p))))))]
    (is (= @snapshot (mapv vec @original)))
    (is (= (mapv (comp (partial * 2.0) first) @snapshot)
           (mapv (comp m/get-value first) (m/get-particles result))))
    (is (< (Math/abs (m/log-marginal result)) 1e-12))
    (is (every? #(< (Math/abs (second %)) 1e-12) (m/get-particles result)))))

(defn- coin []
  (spin (let [h (sample (d/flip 0.3) :id :h)]
          (observe (d/flip (if h 0.9 0.2)) true :id :y)
          h)))

(deftest trace-transport-corrects-the-proposal-and-the-temperature-bridge
  (let [before (atom nil)
        after (atom nil)
        result (b/run-infer 92 #(ip/run {:seed 92}
                                        (fn [scope]
                                          (spin
                                           (let [source (ip/trace-target (coin))
                                                 destination (ip/temperature source 1.0)
                                                 kernel (ip/transport-kernel
                                                         source destination
                                                         (fn [choices _] {:updates {:h (not (:h choices))}
                                                                          :log-q 0.0 :reverse nil})
                                                         (fn [_ _] 0.0))
                                                 p (await (ip/initialize scope source 1024))
                                                 _ (reset! before (mapv :trace/result (:particles p)))
                                                 p (await (ip/transport scope p destination kernel))
                                                 _ (reset! after (mapv :trace/result (:particles p)))]
                                             (ip/finish scope p))))))
        weighted (b/weighted-values result)
        posterior (reduce + (map (fn [[x w]] (if x w 0.0)) weighted))]
    (is (= (mapv not @before) @after))
    (is (< (Math/abs (- (Math/exp (m/log-marginal result)) 0.41)) 0.035))
    (is (< (Math/abs (- posterior (/ 0.27 0.41))) 0.04))))

(deftest indeterminate-transport-scores-are-refused
  (is (thrown? clojure.lang.ExceptionInfo (transport/correction 0.0 ##NaN 0.0 0.0)))
  (is (thrown? clojure.lang.ExceptionInfo (transport/correction ##Inf ##Inf 0.0 0.0)))
  (is (= ##-Inf (transport/correction ##-Inf 0.0 0.0 nil))))

(deftest absent-trace-updates-cannot-be-silently-ignored
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"absent sites"
       (b/run-infer 93 #(ip/run {:seed 93}
                                (fn [scope]
                                  (spin
                                   (let [source (ip/trace-target (coin))
                                         destination (ip/temperature source 1.0)
                                         kernel (ip/transport-kernel source destination
                                                                     (fn [_ _] {:updates {:typo true} :log-q 0.0})
                                                                     (fn [_ _] 0.0))
                                         p (await (ip/initialize scope source 2))
                                         p (await (ip/transport scope p destination kernel))]
                                     (ip/finish scope p)))))))))

(deftest transport-cannot-implicitly-convert-values-into-execution-traces
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"compatible particle representations"
       (b/run-infer 94 #(ip/run {:seed 94}
                                (fn [scope]
                                  (spin
                                   (let [source (ip/value-target (constantly 0.0) (constantly 0.0) (constantly 0.0))
                                         destination (ip/trace-target (coin))
                                         kernel (ip/transport-kernel source destination
                                                                     (fn [x] {:value x :log-q 0.0})
                                                                     (fn [_ _] 0.0))
                                         p (await (ip/initialize scope source 1))
                                         p (await (ip/transport scope p destination kernel))]
                                     (ip/finish scope p)))))))))
