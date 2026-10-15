(ns org.replikativ.foerster.parallel-tempering-test
  "Non-reversible parallel tempering on blocks whose posterior and evidence
  are known in closed form."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.parallel-tempering :as pt]
            [org.replikativ.foerster.population :as pop]
            [org.replikativ.foerster.tempering :as tempering]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- lnorm [x mu s] (dist/logpdf (dist/normal mu s) x))

;; the reference: (a, b) ~ N(0, 3²) each
(def ^:private reference
  {:sample (fn [_] [(* 3.0 (dist/draw (dist/normal 0.0 1.0))) (* 3.0 (dist/draw (dist/normal 0.0 1.0)))])
   :sample-log-density (fn [th _] (let [[a b] (vec th)] (+ (lnorm a 0.0 3.0) (lnorm b 0.0 3.0))))})

(def ^:private log-z -5.0)

(defn- block-model
  "A model that is one block whose target is exp(log-z) times the normalized
  density `lp` of θ."
  [lp]
  (let [blk (block/block {:block/id :pt :block/target :complete-conditional
                          :block/latents [{:name :a :shape []} {:name :b :shape []}]}
                         (merge {:log-density (fn [th _] (+ log-z (lp (vec th))))
                                 ;; required of a block; parallel tempering never asks
                                 :value+grad (fn [th _]
                                               (let [th (vec th) f (+ log-z (lp th)) h 1e-6]
                                                 [f (double-array (map (fn [i] (/ (- (lp (update th i + h)) (lp th)) h))
                                                                       (range (count th))))]))}
                                reference))]
    (spin (let [[a b] (sample (block/block-dist blk nil) :id :theta :init [0.0 0.0])]
            {:a a :b b}))))

(deftest a-seed-fixes-the-run
  (let [model #(block-model (fn [[a b]] (+ (lnorm a 1.0 0.3) (lnorm b -0.5 0.3))))
        values (fn [process-seed make]
                 (mapv (comp m/get-value first) (m/get-particles (b/run-infer process-seed make))))]
    (testing "parallel tempering"
      (let [run #(pt/parallel-tempering (model) {:chains 4 :rounds 4 :seed %})]
        (is (= (values 1 #(run 7)) (values 2 #(run 7))))
        (is (not= (values 1 #(run 7)) (values 1 #(run 8))))))
    (testing "tempered SMC on a block"
      (let [run #(tempering/tempered (model) 16 {:seed %})]
        (is (= (values 1 #(run 7)) (values 2 #(run 7))))
        (is (not= (values 1 #(run 7)) (values 1 #(run 8))))))))

(deftest gaussian-posterior-and-evidence
  (let [res (b/run-infer 5 #(pt/parallel-tempering
                             (block-model (fn [[a b]] (+ (lnorm a 1.0 0.3) (lnorm b -0.5 0.3))))
                             {:chains 8 :rounds 11}))
        a (d/summary res :a)]
    (is (< (Math/abs (- (:mean a) 1.0)) 0.05))
    (is (< (Math/abs (- (:sd a) 0.3)) 0.05))
    (is (< (Math/abs (- (m/log-marginal res) log-z)) 0.3))
    (testing "the diagnostics"
      (is (= 1024 (count (m/get-particles res))))
      (is (= 0.0 (first (:temperatures res))))
      (is (= 1.0 (peek (:temperatures res))))
      (is (apply < (:temperatures res)))
      (is (pos? (:barrier res)))
      (is (pos? (:round-trips res))))))

(deftest bimodal-mass-split
  ;; a ~ 0.3·N(−4, 0.4²) + 0.7·N(4, 0.4²): a single random-walk chain stays
  ;; in the mode it starts in; the tempered chains carry states between them
  (let [res (b/run-infer 9 #(pt/parallel-tempering
                             (block-model (fn [[a b]]
                                            (+ (pop/log-sum-exp [(+ (Math/log 0.3) (lnorm a -4.0 0.4))
                                                                 (+ (Math/log 0.7) (lnorm a 4.0 0.4))])
                                               (lnorm b 0.0 0.4))))
                             {:chains 12 :rounds 11}))
        right (/ (count (filter #(pos? (:a (:result %))) (m/get-contexts res)))
                 (double (count (m/get-particles res))))]
    (is (< (Math/abs (- right 0.7)) 0.08))
    (is (< (Math/abs (- (m/log-marginal res) log-z)) 0.3))
    (is (>= (:round-trips res) 20))))

(deftest refusals
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":chains"
                        (pt/parallel-tempering nil {:chains 1})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #":rounds"
                        (pt/parallel-tempering nil {:rounds 0})))
  (testing "a model that is more than one block"
    (let [e (try (b/run-infer 3 #(pt/parallel-tempering
                                  (spin (let [x (sample (dist/normal 0.0 1.0) :id :x)]
                                          (observe (dist/normal x 1.0) 0.5 :id :y)
                                          x))))
                 nil
                 (catch Exception e e))]
      (is (= ::pt/not-one-block (:type (ex-data (or (ex-cause e) e))))))))

(deftest invalid-steps
  (doseq [steps [0 -1 1.5]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #":steps" (pt/parallel-tempering nil {:steps steps})))))

(deftest a-program-that-fails-on-a-final-state
  ;; the program after the block throws for a < 0, which the posterior
  ;; reaches: the run must fail rather than report nil results
  (let [blk (block/block {:block/id :pt :block/target :complete-conditional
                          :block/latents [{:name :a :shape []} {:name :b :shape []}]}
                         (merge {:log-density (fn [th _] (let [[a b] (vec th)] (+ (lnorm a 0.0 1.0) (lnorm b 0.0 1.0))))
                                 :value+grad (fn [th _] (let [[a b] (vec th)]
                                                          [(+ (lnorm a 0.0 1.0) (lnorm b 0.0 1.0)) (double-array [(- a) (- b)])]))}
                                reference))
        ;; the first run is the template, which must get past the block
        model (fn [] (let [runs (atom 0)]
                       (spin (let [[a _] (sample (block/block-dist blk nil) :id :theta :init [0.5 0.0])]
                               (when (and (< 1 (swap! runs inc)) (neg? a))
                                 (throw (ex-info "negative" {})))
                               a))))
        failure (fn [run] (let [e (try (b/run-infer 3 run) nil (catch Exception e e))]
                            (some #(:type (ex-data %)) (take-while some? (iterate ex-cause e)))))]
    (is (= ::pop/model-failed (failure #(pt/parallel-tempering (model) {:chains 4 :rounds 4}))))
    (testing "tempered SMC's θ path writes back the same way"
      (is (= ::pop/model-failed (failure #(tempering/tempered (model) 64)))))))

(deftest round-trips-start-at-the-reference
  ;; a replica first seen at the target and then at the reference has not
  ;; completed a round trip
  (let [visit @#'pt/end-visit]
    (is (= 0 (:trips (-> {:last nil :trips 0} (visit 3 3) (visit 0 3)))))
    (is (= 1 (:trips (-> {:last nil :trips 0} (visit 0 3) (visit 3 3) (visit 0 3)))))
    (is (= 1 (:trips (-> {:last :reference :trips 0} (visit 2 3) (visit 3 3) (visit 1 3) (visit 0 3)))))))
