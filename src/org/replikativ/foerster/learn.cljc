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
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]])
  #?(:cljs (:require-macros [org.replikativ.spindel.spin.cps :refer [spin]])))

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

(defn maximize-evidence
  "Fit a model's parameters by maximum marginal likelihood (empirical Bayes;
  what Gen's `train!` does for a model's parameters): `model-fn` maps a
  parameter map {name value} to a spin, and the parameters climb the
  evidence log Ẑ(θ) that `infer-opts` (a particle method of `infer/infer`)
  estimates. The gradient is central differences of step `:h`, both sides
  run with the same seed (common random numbers), so their difference is
  not drowned in Monte Carlo noise; Adam takes the steps. Parameters are
  unconstrained reals: give a scale as its log.

  Options: `:steps` (100), `:rate` (0.05), `:h` (0.02), `:seed`. Resolves
  {:params θ :history [{:params :log-evidence} …]}. A run costs 2·d+1
  inferences per step for d parameters: for many parameters, or a guide
  network, train with gradients in finetune-rstr / raster instead."
  [model-fn params infer-opts & [{:keys [steps rate h seed] :or {steps 100 rate 0.05 h 0.02}}]]
  (spin
   (let [names (vec (keys params))
         base-seed (or seed 1)]
     (loop [t 1 theta params m (zipmap names (repeat 0.0)) v (zipmap names (repeat 0.0)) history []]
       (if (> t steps)
         {:params theta :history history}
         (let [opts (assoc infer-opts :seed (+ (* 1000003 base-seed) t))
               lz (m/log-marginal (await (infer/infer (model-fn theta) opts)))
               grad (loop [[k & more] names g {}]
                      (if-not k
                        g
                        (let [up (m/log-marginal (await (infer/infer (model-fn (update theta k + h)) opts)))
                              down (m/log-marginal (await (infer/infer (model-fn (update theta k - h)) opts)))]
                          (recur more (assoc g k (/ (- up down) (* 2.0 h)))))))
               ;; Adam (Kingma & Ba 2015), ascending
               m (into {} (map (fn [k] [k (+ (* 0.9 (m k)) (* 0.1 (grad k)))]) names))
               v (into {} (map (fn [k] [k (+ (* 0.999 (v k)) (* 0.001 (grad k) (grad k)))]) names))
               theta' (into {} (map (fn [k]
                                      (let [mh (/ (m k) (- 1.0 (Math/pow 0.9 t)))
                                            vh (/ (v k) (- 1.0 (Math/pow 0.999 t)))]
                                        [k (+ (theta k) (/ (* rate mh) (+ (Math/sqrt vh) 1e-8)))]))
                                    names))]
           ;; the evidence was estimated at θ, before the step
           (recur (inc t) theta' m v (conj history {:params theta :log-evidence lz}))))))))
