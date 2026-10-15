;; The two-dice examples in the simm.is post, under exact enumeration: a hard
;; constraint (the sum is 7) and a soft one (a noisy reading of 7).
;; Usage, from the repository root: clojure -M experiments/blog/dice.clj
(require '[org.replikativ.foerster.core :as infer]
         '[org.replikativ.foerster.dist :as dist]
         '[org.replikativ.foerster.effects :refer [sample observe factor]]
         '[org.replikativ.foerster.measure :as m]
         '[org.replikativ.spindel.core :as sp]
         '[org.replikativ.spindel.spin.cps :refer [spin]])
(def world (sp/create-execution-context))
(defn posterior [make]
  (let [ms (sp/with-context world @(infer/infer (make) {:method :enumerate}))
        ps (m/get-particles ms) ws (m/normalize-log-weights (mapv second ps))]
    (->> (map (fn [[p _] w] [(m/get-value p) w]) ps ws)
         (reduce (fn [acc [v w]] (update acc v (fnil + 0.0) w)) {})
         (filter (comp pos? second)) (sort-by first) (mapv (fn [[v w]] [v (/ (Math/round (* 1000 w)) 1000.0)])))))
;; hard constraint: the dice sum to 7
(defn dice-hard []
  (spin (let [a (sample (dist/uniform-discrete 1 7) :id :a)
              b (sample (dist/uniform-discrete 1 7) :id :b)]
          (factor (if (= 7 (+ a b)) 0.0 ##-Inf))
          [a b])))
;; soft constraint: a noisy sensor read 7
(defn dice-soft []
  (spin (let [a (sample (dist/uniform-discrete 1 7) :id :a)
              b (sample (dist/uniform-discrete 1 7) :id :b)]
          (observe (dist/normal (+ a b) 1.0) 7.0 :id :reading)
          (+ a b))))
(defn dice-soft-a []
  (spin (let [a (sample (dist/uniform-discrete 1 7) :id :a)
              b (sample (dist/uniform-discrete 1 7) :id :b)]
          (observe (dist/normal (+ a b) 1.0) 7.0 :id :reading)
          a)))
(prn :hard (posterior dice-hard))
(prn :soft-sum (posterior dice-soft))
(prn :soft-a (posterior dice-soft-a))
(shutdown-agents) (System/exit 0)
