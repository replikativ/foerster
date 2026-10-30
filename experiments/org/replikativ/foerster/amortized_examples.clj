(ns org.replikativ.foerster.amortized-examples
  "Source-to-GSDM examples: no manual schemas or attention masks."
  (:refer-clojure :exclude [await])
  (:require [clojure.java.io :as io]
            [org.replikativ.foerster.amortized :as a]
            [org.replikativ.foerster.amortized.compiler :as compiler]
            [org.replikativ.foerster-raster.amortized :as gsdm]
            [org.replikativ.foerster.dist :as d]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.counterfactual :as cf]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.engine.context :as context]
            [org.replikativ.spindel.engine.core :as engine]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(defn gaussian-simulator []
  (spin
   (let [z (sample (d/normal 0 1) :id :z)
         x (sample (d/normal z 1) :id :x)
         y (sample (d/normal (+ (* 2 x) z) 1) :id :y)]
     {:z z :x x :y y})))

(defn intervention-simulator []
  (spin
   (let [intervention (sample (d/flip 0.5) :id :intervention)
         ;; An explicit prior over two mechanism regimes: baseline versus a
         ;; shift intervention on the root x. This is not selection of an
         ;; arbitrary 'cause' without a declared intervention family/prior.
         x (sample (d/normal (if intervention 2.0 0.0) 0.5) :id :x)
         y (sample (d/normal (* 3 x) 0.5) :id :y)]
     {:intervention intervention :x x :y y})))

(defn intervention-probability [y]
  ;; Marginal y | I is Normal(6I, sqrt(2.5)); prior odds are one.
  (let [log-odds (/ (- (* y y) (* (- y 6) (- y 6))) 5.0)]
    (/ 1.0 (+ 1.0 (Math/exp (- log-odds))))))

(defn gaussian-checks [artifact]
  (spin
   (let [joint-conditioned (await (a/infer artifact {:x 1.0 :y 2.5} {:targets [:z] :draws 512 :seed 812}))
         partial (await (a/infer artifact {:x 1.0} {:targets [:z] :draws 512 :seed 813}))
         prior (await (a/infer artifact {} {:targets [:z] :draws 512 :seed 814}))
         worlds (await (a/worlds artifact joint-conditioned {:x 1.0 :y 2.5} [[:z-zero {:z {:do 0.0}}]]))]
     {:x-y {:reference {:mean 0.5 :variance (/ 1.0 3)}
            :learned (select-keys (m/measure-stats joint-conditioned #(get (m/get-value %) :z)) [:mean :variance :std-dev :quantiles])}
      :x-only {:reference {:mean 0.5 :variance 0.5}
               :learned (select-keys (m/measure-stats partial #(get (m/get-value %) :z)) [:mean :variance :std-dev :quantiles])}
      :prior {:reference {:mean 0.0 :variance 1.0}
              :learned (select-keys (m/measure-stats prior #(get (m/get-value %) :z)) [:mean :variance :std-dev :quantiles])}
      :counterfactual {:reference-mean 1.0
                       :mean (cf/expectation worlds #(get-in % [:worlds :z-zero :y]))
                       :complete? (every? #(= :complete (:status %)) worlds)}})))

(defn intervention-checks [artifact]
  (spin
   (loop [remaining [0.0 2.0 3.0 4.0 6.0] reports []]
     (if (seq remaining)
       (let [y (first remaining)
             posterior (await (a/infer artifact {:y y} {:targets [:intervention] :draws 512 :seed 911}))
             stats (m/measure-stats posterior #(if (get (m/get-value %) :intervention) 1.0 0.0))]
         (recur (rest remaining) (conj reports {:y y :reference (intervention-probability y)
                                                :learned (:mean stats)}))) reports))))

(defn save! [path artifact report]
  (io/make-parents path)
  (spit path (pr-str {:artifact (a/freeze (assoc artifact :validation (:checks report))) :report report})))

(defn -main [& [kind output]]
  (let [interventions? (= kind "interventions")
        focused? (= kind "gaussian-z")
        task (compiler/compile (if interventions? #'intervention-simulator #'gaussian-simulator)
                               (cond-> {:id (cond interventions? :intervention-demo/v1 focused? :gaussian-z/v1 :else :gaussian-demo/v1)
                                        :causal? true}
                                 focused? (assoc :targets [:z])))
        backend (gsdm/backend)]
    (binding [engine/*execution-context* (context/create-execution-context)]
      (let [artifact @(a/train task backend (if interventions?
                                              {:simulations 4096 :updates 150000 :rate 0.0001 :seed 911}
                                              {:simulations 4096 :seed 811
                                               :stages [{:updates 100000 :rate 0.0003}
                                                        {:updates 100000 :rate 0.00005}]}))
            checks @(if interventions? (intervention-checks artifact) (gaussian-checks artifact))
            report {:compilation (:compilation task) :training (get-in artifact [:payload :training]) :checks checks}]
        (save! (or output (str "experiments/results/amortized/" (or kind "gaussian") ".edn")) artifact report)
        (prn report)))
    (shutdown-agents)))
