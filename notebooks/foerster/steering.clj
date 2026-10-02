;; # Steering a Process

;; Some processes are random without a `sample` site: a language model
;; writing a sentence, a simulator taking a step. We cannot score their
;; draws, but we can still steer them — run many copies, and keep the ones a
;; reward favours. `foerster.steer` turns such a process into a program for
;; SMC whose target is the process's own law tilted by the reward,
;; p(trajectory) · exp(reward). `foerster.learn` reads the trajectories back
;; as training data.

(ns foerster.steering
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.learn :as learn]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.steer :as steer]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]))

(def world (sp/create-execution-context))

(defn run
  "Run the inference `(make)` returns, seeded."
  [seed make]
  (random/set-seed! seed)
  (sp/with-context world @(make)))

(defn mean
  "Posterior mean of `f` of the program's value."
  [measure f]
  (let [ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))]
    (reduce + (map (fn [[particle _] w] (* w (double (f (m/get-value particle))))) ps ws))))

;; ## A process and a reward
;;
;; The process tosses a fair coin five times and counts the heads. The coin
;; is drawn with `random/uniform01` — no sample site, as a language model's
;; next token would be. The reward λ·heads tilts the law towards heads:
;; the target is Binomial(5, ½) · e^{λ·heads}, which is Binomial(5, p) with
;; p = e^λ / (1 + e^λ), and its evidence is ((1 + e^λ)/2)⁵.

(def lambda 0.8)
(def steps 5)

(defn coin-step [heads]
  (spin (+ heads (if (< (random/uniform01) 0.5) 1 0))))

(def exact
  (let [p (/ (Math/exp lambda) (+ 1 (Math/exp lambda)))]
    {:mean (* steps p)
     :log-evidence (* steps (Math/log (/ (+ 1 (Math/exp lambda)) 2)))}))

exact

;; `steer/model` takes the initial state, the `:step` to the next state, the
;; `:reward` at the end and `:done?`. With a `:value` — an estimate of the
;; reward to come, here the reward so far — SMC resamples at every step on
;; the change of the estimate (twisted SMC); without one, it weighs whole
;; trajectories by their reward at the end (best-of-N).

(defn process [twisted?]
  (steer/model (cond-> {:init 0
                        :step coin-step
                        :done? (constantly false)
                        :max-steps steps
                        :reward (fn [heads] (* lambda heads))}
                 twisted? (assoc :value (fn [heads] (* lambda heads))))))

(def steered
  (array-map
   "twisted" (run 41 #(infer/smc-infer (process true) 1000 {:resampling :stratified}))
   "best-of-N" (run 41 #(infer/smc-infer (process false) 1000 {:resampling :stratified}))))

(kind/table {:column-names ["steering" "mean heads" "log evidence"]
             :row-vectors (into [["exact" (:mean exact) (:log-evidence exact)]]
                                (for [[label measure] steered]
                                  [label (mean measure identity) (m/log-marginal measure)]))})

;; Both reach the tilted law and its evidence. The twist pays off when the
;; value estimate is informative and trajectories are long: SMC drops the
;; unpromising ones early instead of finishing them.

;; ## A biased proposal, weighted back
;;
;; A step may come from another process than the one we target — a learned
;; proposal, a smaller model. It then returns `(steer/weighted state
;; log-w)` with log-w = log p(state) − log q(state), and the weight restores
;; the target. Here the steps come from a coin with heads 0.8:

(defn biased-step [heads]
  (spin (if (< (random/uniform01) 0.8)
          (steer/weighted (inc heads) (Math/log (/ 0.5 0.8)))
          (steer/weighted heads (Math/log (/ 0.5 0.2))))))

(def biased
  (run 43 #(infer/smc-infer (steer/model {:init 0
                                          :step biased-step
                                          :done? (constantly false)
                                          :max-steps steps
                                          :value (fn [heads] (* lambda heads))
                                          :reward (fn [heads] (* lambda heads))})
                            1000 {:resampling :stratified})))

{:mean-heads (mean biased identity)
 :log-evidence (m/log-marginal biased)}

;; The same target, from draws of a different coin.

;; ## Trajectories as training data
;;
;; Every step records its state under `[:steer/state t]` and the end its
;; reward under `:steer/reward`, so each particle carries its trajectory.
;; `learn/trajectories` reads them back with their normalized weights — the
;; data for learning a value estimate or a proposal that makes the next
;; search cheaper:

(def trajectories (learn/trajectories (get steered "best-of-N")))

(kind/table {:column-names ["states" "reward" "weight"]
             :row-vectors (mapv (juxt :states :reward :weight) (take 5 trajectories))})

;; A learner that takes plain, unweighted examples wants draws from the
;; target instead: `learn/draws` resamples the trajectories by weight.

(def draws (learn/draws (get steered "best-of-N") 2000))

{:mean-final-heads (/ (reduce + (map (comp peek :states) draws)) (double (count draws)))
 :exact (:mean exact)}
