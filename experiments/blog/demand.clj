;; Reruns the demand numbers in the simm.is post "Probabilistic programming for
;; Clojure programmers": SMC, importance sampling, MH and PMMH against the exact
;; answers.  Usage, from the repository root: clojure -M experiments/blog/demand.clj out.edn
(ns demand-post
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

(defn demand-unknown [weeks]
  (sp/spin
   (let [drift (sample (dist/gamma 2.0 2.5) :id :drift)]
     (loop [t 0
            level (sample (dist/normal 100.0 20.0) :id [:level 0])]
       (observe (dist/normal level 8.0) (nth weeks t) :id [:y t])
       (if (= (inc t) (count weeks))
         drift
         (recur (inc t)
                (sample (dist/normal level drift) :id [:level (inc t)])))))))

(def world (sp/create-execution-context))

(defn run [seed make]
  (random/set-seed! seed)
  (let [t0 (System/nanoTime)
        measure (sp/with-context world @(make))]
    {:measure measure :s (/ (- (System/nanoTime) t0) 1e9)}))

(defn mean-sd [xs]
  (let [n (count xs) mu (/ (reduce + xs) n)]
    [mu (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) (dec n)))]))

(defn particle-runs [method]
  (vec (for [seed (range 1 11)]
         (let [{:keys [measure s]} (run seed #(infer/infer (demand weeks 4.0) {:method method :particles 1000}))
               {:keys [mean ess]} (d/summary measure identity)]
           {:mean mean :log-z (m/log-marginal measure) :ess ess
            :histories (d/distinct-count measure [:level 0]) :s s}))))

(defn summarize [runs]
  (into {} (for [k [:mean :log-z :ess :histories :s]]
             [k (let [[mu sd] (mean-sd (mapv k runs))]
                  {:mean mu :sd sd :min (apply min (map k runs)) :max (apply max (map k runs))})])))

(defn band []
  (vec (for [t (range 1 21)]
         (let [{:keys [measure]} (run (+ 100 t) #(infer/infer (demand (subvec weeks 0 t) 4.0) {:method :smc :particles 1000}))
               {:keys [mean quantiles]} (d/summary measure identity)]
           {:week t :mean mean :quantiles quantiles}))))

(defn -main [out]
  (let [smc (particle-runs :smc)
        is (particle-runs :importance)
        mh (run 7 #(infer/infer (demand weeks 4.0) {:method :mh :iterations 4000 :chains 4 :burn 1000}))
        mh-s (d/summary (:measure mh) identity)
        pmmh (run 11 #(infer/infer (demand-unknown weeks)
                                   {:method :pmmh :particles 100 :iterations 2000
                                    :params #{:drift} :scale 3.0 :burn 200}))
        drift-draws (mapv (comp double m/get-value first) (m/get-particles (:measure pmmh)))
        pmmh-s (d/summary (:measure pmmh) identity)
        result {:smc (summarize smc) :importance (summarize is)
                :mh {:mean (:mean mh-s) :rhat (:rhat mh-s) :ess-bulk (:ess-bulk mh-s)
                     :draws (:draws mh-s) :s (:s mh)}
                :pmmh {:summary (dissoc pmmh-s :chains) :n (count drift-draws) :s (:s pmmh)
                       :q05 (nth (sort drift-draws) (long (* 0.05 (count drift-draws))))
                       :q95 (nth (sort drift-draws) (long (* 0.95 (count drift-draws))))
                       :draws drift-draws}
                :band (band)}]
    (spit out (pr-str result))
    (println (pr-str (-> result (update :pmmh dissoc :draws) (dissoc :band))))
    (shutdown-agents)
    (System/exit 0)))

(apply -main *command-line-args*)
