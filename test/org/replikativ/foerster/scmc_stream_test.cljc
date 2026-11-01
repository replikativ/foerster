(ns org.replikativ.foerster.scmc-stream-test
  (:require #?(:clj [clojure.test :refer [deftest is]]
               :cljs [cljs.test :refer-macros [deftest is]])
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.scmc :as scmc]))

(def model
  {:prior-sample #(vector (d/draw (d/normal 0.0 1.0)))
   :prior-log-density #(d/logpdf (d/normal 0.0 1.0) (first %))
   :log-lik (fn [[mu] y] (d/logpdf (d/normal mu 1.0) y))})

(deftest pause-resume-replay-and-snapshot-isolation
  (let [opts {:particles 64 :memory 4 :seed 41}
        initial (scmc/start model opts)
        paused (scmc/append-batch initial [0.0 1.0 0.5])
        before (scmc/result paused)
        resumed (scmc/append-batch paused [2.0 1.0 -0.5])
        continued (scmc/append-batch resumed [0.3 0.7])
        replay (scmc/scmc (assoc model :batches [[0.0 1.0 0.5] [2.0 1.0 -0.5] [0.3 0.7]]) opts)]
    (is (= replay (scmc/result continued)))
    (is (= before (scmc/result paused)))
    (is (= (scmc/append-batch paused [2.0 1.0 -0.5]) resumed))
    (is (= 0 (:steps initial)))
    (is (= 1 (:steps paused)))
    (is (= 3 (:steps continued)))
    (is (<= (count (:coreset continued)) 4))
    (is (<= (:max-coreset-size continued) 7))
    (is (= (:max-coreset-size continued) (:max-transient-coreset-size continued)))
    (is (<= (:max-retained-coreset-size continued) 4))
    (is (= :approximate-population (get-in replay [:inference-contract :semantics])))
    (is (= :unverified (get-in replay [:inference-contract :normalizer :status])))
    (is (<= 1.0 (get-in continued [:last-update :ess]) 64.000001))
    (is (= 64 (count (m/get-particles replay))))))

(deftest compression-only-affects-the-posterior-when-used-in-a-move
  (let [first-step (scmc/append-batch (scmc/start model {:particles 32 :memory 2 :seed 42})
                                      [0.0 1.0 2.0])
        next-step (scmc/append-batch first-step [0.5])]
    (is (true? (:compressed? first-step)))
    (is (= :weighted-population (get-in (scmc/result first-step) [:inference-contract :semantics])))
    (is (= :approximate-population (get-in (scmc/result next-step) [:inference-contract :semantics])))))

(deftest an-empty-append-is-not-an-inference-step
  (let [state (scmc/start model {:particles 16 :seed 43})]
    (is (thrown-with-msg? #?(:clj clojure.lang.ExceptionInfo :cljs js/Error)
                          #"non-empty batch" (scmc/append-batch state [])))
    (is (= 0 (:steps state)))))

(deftest a-run-without-a-seed-replays-from-its-recorded-seed
  (let [batches [[0.0 1.0 0.5] [2.0 1.0 -0.5] [0.3 0.7]]
        run (reduce scmc/append-batch (scmc/start model {:particles 32 :memory 3}) batches)
        options (scmc/options run)]
    (is (some? (:seed run)))
    (is (= {:particles 32 :memory 3 :scale 2.38 :seed (:seed run) :refit :nnls} options))
    (is (= run (reduce scmc/append-batch (scmc/initial run) batches)))
    (is (= (dissoc run :model)
           (dissoc (reduce scmc/append-batch (scmc/start model options) batches) :model)))
    (is (not= (:particles run)
              (:particles (reduce scmc/append-batch (scmc/start model {:particles 32 :memory 3}) batches))))))

(deftest retained-and-transient-core-set-sizes
  ;; memory 4, batches of 3: the core-set kept between steps never exceeds 4,
  ;; while a step moves against up to 4 + 3
  (let [states (reductions scmc/append-batch (scmc/start model {:particles 16 :memory 4 :seed 44})
                           (repeat 5 [0.1 0.4 -0.2]))
        last-state (last states)]
    (is (= (reduce max (map (comp count :coreset) states)) (:max-retained-coreset-size last-state)))
    (is (<= (:max-retained-coreset-size last-state) 4))
    (is (= 7 (:max-transient-coreset-size last-state) (:max-coreset-size last-state)))
    (is (= (select-keys last-state [:max-retained-coreset-size :max-transient-coreset-size :max-coreset-size])
           (select-keys (scmc/result last-state) [:max-retained-coreset-size :max-transient-coreset-size :max-coreset-size])))))
