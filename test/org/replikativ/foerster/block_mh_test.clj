(ns org.replikativ.foerster.block-mh-test
  "Random-walk MH and tempered SMC on a block site: the golf putting logistic
  regression as one block, against its exact posterior and evidence (a 2-D
  grid; experiments/comparison/data/workloads.edn)."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def ^:private golf
  {:x [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]
   :n [1443 694 455 353 272 256 240 217 200 237 202 192 174 167 201 195 191 147 152]
   :y [1346 577 337 208 149 136 111 69 67 75 52 46 54 28 27 31 33 20 24]})

(def ^:private exact
  {:a 2.224729440047498 :a-sd 0.058283119499046576
   :b -0.2551149806173231 :log-evidence -191.93219950514396})

(defn- sigmoid [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))

(defn- prior-lp [a b] (+ (dist/logpdf (dist/normal 0.0 1.0) a) (dist/logpdf (dist/normal 0.0 1.0) b)))

(defn- golf-lp [th {:keys [n x y]}]
  (let [[a b] (vec th)]
    (reduce + (prior-lp a b)
            (map (fn [ni xi yi] (dist/logpdf (dist/binomial ni (sigmoid (+ a (* b xi)))) yi)) n x y))))

(defn- golf-block [capabilities]
  (block/block {:block/id :golf :block/target :complete-conditional
                :block/latents [{:name :a :shape []} {:name :b :shape []}]}
               (merge {:log-density golf-lp
                       :value+grad (fn [th {:keys [n x y] :as inputs}]
                                     (let [[a b] (vec th)
                                           rs (map (fn [ni xi yi] (- yi (* ni (sigmoid (+ a (* b xi)))))) n x y)]
                                       [(golf-lp th inputs)
                                        (double-array [(- (reduce + rs) a)
                                                       (- (reduce + (map * rs x)) b)])]))}
                      capabilities)))

;; drawn from the prior, so tempering runs from the prior to the posterior
(def ^:private prior-sampled
  {:sample (fn [_] [(dist/draw (dist/normal 0.0 1.0)) (dist/draw (dist/normal 0.0 1.0))])
   :sample-log-density (fn [th _] (let [[a b] (vec th)] (prior-lp a b)))})

(defn- model [blk] (spin (let [[a b] (sample (block/block-dist blk golf) :id :theta :init [0.0 0.0])]
                           {:a a :b b})))

(deftest random-walk-mh-moves-a-block-on-its-own-density
  (let [measure (b/run-infer 7 #(infer/infer (model (golf-block prior-sampled))
                                             {:method :rmh :iterations 3000 :burn 1000 :chains 4
                                              :step-size 0.02}))
        a (d/summary measure :a)]
    (is (< (:rhat a) 1.05))
    (is (< (Math/abs (- (:mean a) (:a exact))) (* 0.25 (:a-sd exact))) (str a))
    (is (< (Math/abs (- (:mean (d/summary measure :b)) (:b exact))) 0.005))))

(deftest tempered-smc-tempers-a-block-from-its-draws-to-its-target
  (let [measure (b/run-infer 11 #(infer/infer (model (golf-block prior-sampled))
                                              {:method :tempered :particles 2000}))]
    (testing "the evidence of the block's target"
      (is (< (Math/abs (- (m/log-marginal measure) (:log-evidence exact))) 0.5)
          (str (m/log-marginal measure))))
    (testing "the posterior, not the draws' law"
      (is (< (Math/abs (- (:mean (d/summary measure :a)) (:a exact))) (* 0.25 (:a-sd exact)))))))

(deftest a-block-without-a-draw-density-cannot-be-tempered
  (let [error (try (b/run-infer 3 #(infer/infer (model (golf-block (dissoc prior-sampled :sample-log-density)))
                                                {:method :tempered :particles 20}))
                   nil
                   (catch Exception e e))]
    (is (some #(= ::block/no-sample-density (:type (ex-data %)))
              (take-while some? (iterate ex-cause error)))
        (str error))))
