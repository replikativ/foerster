(ns org.replikativ.foerster.steer
  "Steering a sequential process by SMC: a program that takes steps — an
  agent's turns, a model's sentences — whose proposals come from the process
  itself (a language model, a simulator: randomness without a sample site,
  so its density cancels as a prior draw's does), scored by a value estimate
  at each step and by a reward at the end.

  The target is p(trajectory) · exp(reward): the process's own law tilted by
  the reward. A value estimate ψ (a verifier, a process reward model, a
  judge) twists the intermediate targets without changing the final one
  (twisted SMC; Whiteley & Lee 2014, Zhao et al. 2024): step t adds
  log ψ_t − log ψ_{t−1} as a barrier factor, so SMC resamples on it, and the
  end adds reward − log ψ_T. With no value estimate every intermediate
  factor is 0 and SMC is best-of-N weighted by the reward.

    (infer/smc-infer (steer/model {:init s0
                                   :step (fn [state] (spin …))   ; the next state
                                   :value (fn [state] (spin …))  ; log ψ
                                   :reward (fn [state] (spin …)) ; log potential
                                   :done? (fn [state] …)})
                     16 {:resampling :stratified})

  `:step`, `:value` and `:reward` may return a spin or a plain value."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.effects :refer [factor deterministic]]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.spindel.effects.savepoint :refer [savepoint]]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]])
  #?(:cljs (:require-macros [org.replikativ.spindel.spin.cps :refer [spin]])))

(defn- value-of
  "`x` or, when it is a spin, a spin resolving its value."
  [x]
  (if (satisfies? spin-core/PSpin x) x (spin x)))

(defn model
  "A program for SMC that steers `:step` by `:value` and `:reward` (see the
  namespace). Options:

    :init       the initial state
    :step       (fn [state]) → the next state (a spin or a value)
    :done?      (fn [state]) → whether the trajectory ends at `state`
    :value      (fn [state]) → log ψ(state), an estimate of the reward to come
                (default: none, 0)
    :reward     (fn [state]) → the final log potential (default 0)
    :max-steps  end after this many steps (default 100)

  Each step records the state under `[:steer/state t]`
  (`foerster.effects/deterministic`), so the trajectory is in every
  particle's trace. The program's value is the final state. It starts at
  `smc/start-site`: the steps are random without a sample site, so every
  particle must take them itself rather than share a prefix."
  [{:keys [init step done? value reward max-steps] :or {max-steps 100}}]
  (spin
   (savepoint smc/start-site nil)
   (loop [t 0 state init psi 0.0]
     (let [state' (await (value-of (step state)))
           _ (deterministic state' :id [:steer/state t])
           end? (or (done? state') (>= (inc t) max-steps))
           psi' (if (and value (not end?)) (await (value-of (value state'))) 0.0)]
       (if end?
         (let [r (if reward (await (value-of (reward state'))) 0.0)]
           ;; untwist: the final target is the reward, whatever ψ said
           (factor (- r psi))
           state')
         (do (factor (- psi' psi) :barrier true)
             (recur (inc t) state' psi')))))))
