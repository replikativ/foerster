(ns org.replikativ.foerster.site-kernel-trace-test
  "A site kernel sees the particle's trace so far, kept up to date
  incrementally in the particle's world (`trace/world-legacy-trace`)."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.kernel :as k]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- chain-model [n]
  (spin
   (loop [i 0 x 0.0]
     (if (= i n)
       x
       (let [x' (sample (dist/normal x 1.0) :id [:x i])]
         (observe (dist/normal x' 1.0) 0.0 :id [:y i])
         (recur (inc i) x'))))))

(deftest a-site-kernel-sees-the-trace-so-far
  (let [seen (atom [])
        checking (reify k/PInferenceKernel
                   (kernel-id [_] :checking)
                   (step [_ world checkpoint trace]
                     (swap! seen conj [(:address checkpoint)
                                       (= trace (itrace/legacy-trace (rtp/get-state world [:savepoint/trace])))
                                       (count trace)])
                     {:value (dist/draw (:source checkpoint))}))]
    (b/run-infer 111 #(infer/kernel-infer (chain-model 25) checking 3))
    (is (= (* 3 25) (count @seen)))
    (is (every? second @seen) "the incremental view equals the full conversion")
    (is (= (set (range 0 50 2)) (set (map #(nth % 2) @seen)))
        "site i sees its i latent and i observed predecessors")))
