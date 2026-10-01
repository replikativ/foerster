(ns org.replikativ.foerster.learn
  "Training data from inference: the trajectories SMC drew, with their
  rewards and weights, for learning the value estimates (twists) and
  proposals that make the next search cheaper.

  A steered program (`foerster.steer/model`) records its states under
  `[:steer/state t]` and its reward under `:steer/reward`. `trajectories`
  reads them back from a measure's particles with their normalized weights;
  `draws` resamples them by weight into unweighted draws from the target
  p · exp(reward), which is what a model trained on plain examples needs.
  Training itself (value heads, proposals over a language model's hidden
  states) lives with the models: finetune-rstr's typed-decision records take
  a state and the probability of success as a soft target.

    (learn/draws (await (infer/smc-infer (steer/model …) 64)) 256)"
  (:require [org.replikativ.foerster.measure :as m]))

(defn trajectory
  "A particle's steered trajectory: {:states [s₀ s₁ …] :reward r}, the states
  in step order; nil when the particle has no steered states."
  [particle]
  (let [trace (m/get-trace particle)
        states (into [] (map #(get-in trace [[:steer/state %] :value]))
                     (take-while #(contains? trace [:steer/state %]) (range)))]
    (when (seq states)
      {:states states
       :reward (get-in trace [:steer/reward :value])})))

(defn trajectories
  "Every particle's trajectory of `measure` with its normalized `:weight`
  and `:log-weight`, in particle order."
  [measure]
  (let [ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))]
    (into []
          (keep (fn [[[particle log-w] w]]
                  (some-> (trajectory particle) (assoc :weight w :log-weight log-w))))
          (map vector ps ws))))

(defn draws
  "`n` trajectories of `measure` drawn by weight (with replacement): unweighted
  draws from the target. Each keeps its particle's `:index`."
  [measure n]
  (let [ts (trajectories measure)
        ws (mapv :weight ts)]
    (vec (repeatedly n #(let [i (m/sample-categorical ws)]
                          (assoc (nth ts i) :index i))))))
