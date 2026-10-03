;; # Streaming SMC

;; Some data arrives over time: sensor readings, messages, a market. A
;; particle filter updates its posterior with each new observation, without
;; re-running what it has already seen. In foerster the model marks the
;; sites whose values arrive from outside with `:stream true`, and
;; `smc/stream` pushes observations into them one at a time.

(ns foerster.streaming
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]))

(def world (sp/create-execution-context))
(random/set-seed! 5)

;; ## A tracking model
;;
;; A position that drifts as a random walk, x_t = x_{t−1} + N(0, 1), seen
;; through noisy measurements y_t ~ N(x_t, 1). Each `[:y t]` site is a
;; stream site: its value is the next measurement, pushed from outside.

(def measurements [0.8 1.9 1.2 3.0 2.5 3.9 4.1 3.6 5.2 6.0])

(defn tracker []
  (spin
   (loop [t 0
          x (sample (dist/normal 0.0 1.0) :id [:x 0])]
     (sample (dist/normal x 1.0) :id [:y t] :stream true)
     (if (= t (dec (count measurements)))
       x
       (recur (inc t) (sample (dist/normal x 1.0) :id [:x (inc t)]))))))

;; ## Pushing observations
;;
;; `smc/stream` is a CPS operation: it resolves a *step* once every particle
;; waits at its next stream site. A step carries the current `:measure`, a
;; `:push` function taking the next value, and `:done?`. Inside a spin
;; `await` takes a CPS operation; at the REPL we deref such a spin.

(defn run [operation]
  (sp/with-context world @(spin (await operation))))

(defn filtered
  "Posterior mean and standard deviation of x_t."
  [measure t]
  (let [ps (m/get-particles measure)
        ws (m/normalize-log-weights (mapv second ps))
        xs (mapv (fn [[s _]] (:value (get (m/get-trace s) [:x t]))) ps)
        mu (reduce + (map * ws xs))]
    [mu (Math/sqrt (reduce + (map (fn [w x] (* w (let [d (- x mu)] (* d d)))) ws xs)))]))

(def steps
  (sp/with-context world
    (loop [step (run (smc/stream (tracker) 2000))
           t 0
           out []]
      (if (= t (count measurements))
        out
        (let [step' (run ((:push step) (nth measurements t)))]
          (recur step' (inc t) (conj out (filtered (:measure step') t))))))))

;; ## Against the exact answer
;;
;; For a linear Gaussian model the filtering distribution is known exactly —
;; it is the Kalman filter:

(defn kalman [ys]
  (loop [[y & more] ys, mean 0.0, var 1.0, out []]
    (if-not y
      out
      (let [k (/ var (+ var 1.0))
            mean' (+ mean (* k (- y mean)))
            var' (* (- 1.0 k) var)]
        (recur more mean' (+ var' 1.0) (conj out [mean' (Math/sqrt var')]))))))

(def comparison
  (tc/dataset (map (fn [t y [pm ps] [km ks]]
                     {:t t :y y :particles pm :particles-sd ps :kalman km :kalman-sd ks})
                   (range) measurements steps (kalman measurements))))

(kind/table comparison)

(-> comparison
    (tc/pivot->longer [:y :particles :kalman] {:target-columns :series :value-column-name :value})
    (plotly/layer-line {:=x :t :=y :value :=color :series}))

;; ## What happens at a push
;;
;; Each push scores every particle's stream site with the new value, which
;; turns it into an ordinary observation; the population is resampled when
;; its effective sample size falls below half the particles; and each
;; particle runs on to its next stream site. Nothing already seen is re-run:
;; a particle is a world parked at a savepoint, and a push resumes it.
;;
;; A step also has `:close`, which gives the worlds back if you stop early;
;; after the last observation every particle has returned and the step is
;; `:done?`.
