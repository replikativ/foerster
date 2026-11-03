(ns org.replikativ.foerster.site-kernel-trace-test
  "A site kernel sees the particle's trace so far, kept up to date
  incrementally in the particle's world (`trace/world-legacy-trace`)."
  (:require [clojure.test :refer [deftest is]]
            [org.replikativ.foerster.proposal :as proposal]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn- chain-model [n]
  (spin
   (loop [i 0 x 0.0]
     (if (= i n)
       x
       (let [x' (sample (dist/normal x 1.0) :id [:x i])]
         (observe (dist/normal x' 1.0) 0.0 :id [:y i])
         (recur (inc i) x'))))))

(deftest a-proposal-sees-the-trace-so-far
  (let [seen (atom [])
        checking (proposal/from-fn
                  (fn [{:keys [address dist trace]}]
                    (swap! seen conj [address (count @trace)])
                    (let [v (dist/draw dist)] {:value v :log-q (dist/logpdf dist v)})))]
    (b/run-infer 111 #(infer/smc-infer (chain-model 25) 3 {:proposal checking}))
    (is (= (* 3 25) (count @seen)))
    (is (= (set (range 0 50 2)) (set (map second @seen)))
        "site i sees its i latent and i observed predecessors")))
