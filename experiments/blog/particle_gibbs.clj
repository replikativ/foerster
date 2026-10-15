;; The fix for the demand model's Metropolis–Hastings failure in the simm.is post:
;; particle Gibbs, four chains, checked with the same R-hat and bulk ESS.
;; Usage, from the repository root: clojure -M experiments/blog/particle_gibbs.clj out.edn
(ns demand-fix
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.diagnostics :as d]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.core :as sp]))

(def weeks [110 108 106 115 114 97 118 97 135 114 105 128 105 101 107 115 112 103 103 98])

(defn demand [weeks drift]
  (sp/spin
   (loop [t 0
          level (sample (dist/normal 100.0 20.0) :id [:level 0])]
     (observe (dist/normal level 8.0) (nth weeks t) :id [:y t])
     (if (= (inc t) (count weeks))
       level
       (recur (inc t)
              (sample (dist/normal level drift) :id [:level (inc t)]))))))

(def world (sp/create-execution-context))

(defn sweep-means
  "The weighted mean of `f` within each sweep of `particles` particles."
  [measure particles f burn]
  (let [ps (m/get-particles measure)]
    (->> (partition particles ps)
         (drop burn)
         (mapv (fn [sweep]
                 (let [ws (m/normalize-log-weights (mapv second sweep))]
                   (reduce + (map (fn [[p _] w] (* w (double (f p)))) sweep ws))))))))

(defn chains [method particles sweeps burn]
  (vec (for [seed [1 2 3 4]]
         (do (random/set-seed! seed)
             (let [t0 (System/nanoTime)
                   measure (sp/with-context world
                             @(infer/infer (demand weeks 4.0) {:method method :particles particles :iterations sweeps}))]
               {:final (sweep-means measure particles m/get-value burn)
                :week0 (sweep-means measure particles #(m/site-value % [:level 0]) burn)
                :s (/ (- (System/nanoTime) t0) 1e9)})))))

(defn report [cs]
  (let [final (mapv :final cs) week0 (mapv :week0 cs)
        all (apply concat final)]
    {:mean (/ (reduce + all) (count all))
     :rhat (d/rhat final) :ess-bulk (d/ess-bulk final)
     :week0-rhat (d/rhat week0) :week0-ess (d/ess-bulk week0)
     :draws (count all) :s (reduce + (map :s cs))}))

(defn -main [out]
  (let [r {:pgibbs (report (chains :pgibbs 50 600 100))}]
    (spit out (pr-str r)) (prn r)
    (shutdown-agents) (System/exit 0)))

(apply -main *command-line-args*)
