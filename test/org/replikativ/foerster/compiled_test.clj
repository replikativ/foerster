(ns org.replikativ.foerster.compiled-test
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.analysis :as a]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.inference.compiled :as c]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.runtime.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.engine.executor :as sched]))

(def ^:private logistic
  (a/model [xs ns ys]
           (let [a (sample (dist/normal 0.0 1.0) :id :a)
                 b (sample (dist/normal 0.0 1.0) :id :b)]
             (loop [i 0]
               (when (< i (count xs))
                 (observe (dist/binomial (nth ns i) (/ 1.0 (+ 1.0 (Math/exp (- (+ a (* b (nth xs i))))))))
                          (nth ys i) :id [:putt i])
                 (recur (inc i))))
             {:a a :b b})))

(def ^:private data [[2.0 3.0 4.0 5.0] [100 90 80 70] [80 60 40 25]])

(deftest the-compiled-density-is-the-models-log-joint
  (let [{:keys [addresses log-density]} (apply c/compile-model logistic data)]
    (is (= [:a :b] addresses))
    (doseq [[av bv] [[0.3 -0.2] [1.1 -0.6] [-0.5 0.4]]]
      (let [{:keys [weight]} (b/run-infer 1 #(gfi/assess (apply logistic data) {:a av :b bv}))]
        (is (< (Math/abs (- weight (log-density (double-array [av bv])))) 1e-9))))))

(def ^:private normal-mean
  (a/model [ys]
           (let [mu (sample (dist/normal 0.0 1.0) :id :mu)]
             (loop [i 0]
               (when (< i (count ys))
                 (observe (dist/normal mu 1.0) (nth ys i) :id [:y i])
                 (recur (inc i))))
             mu)))

(deftest rmh-on-the-compiled-density-finds-the-posterior
  (let [ys [0.8 1.4 0.3 1.9 1.1 0.6 1.5 0.9]
        exact (/ (reduce + ys) (inc (count ys)))
        measure (b/run-infer 2 #(c/rmh (c/compile-model normal-mean ys)
                                       {:iterations 20000 :burn 1000 :step-size 0.6 :chains 2}))
        mus (map m/get-value (m/get-contexts measure))]
    (is (= 38000 (count mus)) "every state after :burn, per chain")
    (is (= [19000 19000] (:chain-lengths measure)))
    (is (< (Math/abs (- exact (/ (reduce + mus) (count mus)))) 0.03))))

(deftest infer-rmh-compiled
  (let [ys [0.8 1.4 0.3 1.9 1.1 0.6 1.5 0.9]
        exact (/ (reduce + ys) (inc (count ys)))
        run #(b/run-infer 3 (fn [] (infer/infer (normal-mean ys) {:method :rmh :compiled? true :iterations 8000
                                                                  :burn 500 :chains 2 :step-size 0.6 :seed [1 2]})))
        measure (run)
        mus (map m/get-value (m/get-contexts measure))]
    (is (= [7500 7500] (:chain-lengths measure)))
    (is (< (Math/abs (- exact (/ (reduce + mus) (count mus)))) 0.05))
    (is (= mus (map m/get-value (m/get-contexts (run)))) "a seeded run repeats")
    (is (= (first mus) (m/site-value (first (m/get-contexts measure)) :mu)))))

(deftest a-model-whose-structure-depends-on-a-value-is-refused
  (let [mixture (a/model [y]
                         (let [z (sample (dist/flip 0.5) :id :z)]
                           (if z
                             (observe (dist/normal 1.0 1.0) y :id :a)
                             (observe (dist/normal -1.0 1.0) y :id :b))))]
    (is (= #{:value-dependent-structure :unknown-test :not-continuous}
           (set (map :reason (:refusals (ex-data (try (c/compile-model mixture 0.3)
                                                      (catch clojure.lang.ExceptionInfo e e))))))))))

(defn- counting-executor
  "An executor that runs its tasks on `inner`, counting them in `n`."
  [inner n]
  (reify sched/PExecutor
    (execute! [_ f] (swap! n inc) (sched/execute! inner f))
    (execute-after! [_ ms f] (swap! n inc) (sched/execute-after! inner ms f))))

(deftest compiled-chains-run-on-the-executor
  ;; no thread of their own: the chains are tasks of :executor, the same
  ;; draws whatever executes them
  (let [ys [0.8 1.4 0.3 1.9]
        run (fn [executor]
              (with-redefs [clojure.core/future-call (fn [_] (throw (ex-info "a chain on a future" {})))]
                (b/run-infer 4 (fn [] (infer/infer (normal-mean ys) {:method :rmh :compiled? true :iterations 300
                                                                     :chains 3 :step-size 0.6 :seed [3 4]
                                                                     :executor executor})))))
        tasks (atom 0)
        pool (sched/thread-pool-executor {:threads 2})]
    (try
      (let [pooled (run (counting-executor pool tasks))
            inline (run (sched/synchronous-executor))]
        (is (pos? @tasks))
        (is (= [300 300 300] (:chain-lengths pooled)))
        (is (= (map m/get-value (m/get-contexts pooled)) (map m/get-value (m/get-contexts inline)))))
      (finally (.close ^java.lang.AutoCloseable pool)))))

(deftest a-failing-chain-is-one-clean-error
  ;; one chain of eight throws: the run rejects once, with that error, never
  ;; resolves, and its own pool is not shut down under the other workers
  ;; (no RejectedExecutionException from them)
  (let [compiled (c/compile-model normal-mean [0.8 1.4])
        calls (atom 0)
        outcomes (atom [])
        with-stream* @#'random/with-stream*
        done (promise)]
    (with-redefs [random/with-stream* (fn [seed key f]
                                        (if (= 2 (swap! calls inc))
                                          (throw (ex-info "chain failed" {:type ::chain-failed}))
                                          (with-stream* seed key f)))]
      ((c/rmh compiled {:iterations 5 :chains 8 :seed [5 6]})
       #(do (swap! outcomes conj [:resolve %]) (deliver done true))
       #(do (swap! outcomes conj [:reject %]) (deliver done true)))
      (deref done 10000 nil)
      ;; let the other chains finish before looking
      (Thread/sleep 500))
    (is (= 1 (count @outcomes)) (str (mapv first @outcomes)))
    (is (= :reject (ffirst @outcomes)))
    (is (= ::chain-failed (:type (ex-data (second (first @outcomes))))))))
