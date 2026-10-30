(ns org.replikativ.foerster.particle-mcmc
  "Particle Markov chains over an injected sweep operation. World execution is
  the sweep's responsibility; normalization, retention and accept/reject live
  here, shared by savepoint and canonical-world drivers."
  (:refer-clojure :exclude [await])
  (:require [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.effects.await :refer [await]]))

(defn normalized
  "Normalize a settled sweep for pooling. Impossible particles remain zero."
  [measure]
  (let [ps (m/get-particles measure)
        z (m/log-sum-exp (mapv second ps))]
    (if (= ##-Inf z) ps
        (mapv (fn [[s lw]] [s (- lw z)]) ps))))

(defn retained-choices
  "Retained choices from a settled Sample trace, excluding deterministic and
  observed sites. The conditional sweep interprets these at its own frontier."
  [trace]
  (into {} (keep (fn [[a e]] (when-not (or (:observed? e) (:deterministic? e)) [a (:value e)]))) trace))

(defn- iterations! [iterations]
  (when-not (pos-int? iterations)
    (throw (ex-info "Particle chain iterations must be positive"
                    {:type ::invalid-iterations :iterations iterations}))))

(defn pimh
  "Run an independent particle MH chain. sweep receives [key retained seed]
  and returns an awaitable weighted sweep; retained is nil. The sweep must
  supply a valid evidence estimate. It owns and settles its execution worlds."
  [sweep iterations {:keys [seed accept-key first-iteration]
                     :or {accept-key ::accept first-iteration 0}}]
  (iterations! iterations)
  (spin
   (let [seed (or seed (random/fresh-seed))
         initial (await (sweep :initial nil seed))]
     (loop [current (normalized initial) log-z (m/log-marginal initial)
            i 0 samples []]
       (if (= i iterations)
         (m/markov-measure samples)
         (let [key (+ first-iteration i)
               proposed (await (sweep key nil seed))
               log-z' (m/log-marginal proposed)
               ratio (- log-z' log-z)
               u (random/with-stream* seed [accept-key key] m/uniform01)
               accept? (or (>= ratio 0.0) (< (Math/log u) ratio))
               [next z] (if accept? [(normalized proposed) log-z'] [current log-z])]
           (recur next z (inc i) (into samples next))))))))

(defn- pick-retained [measure seed role key]
  (let [ps (m/get-particles measure)
        selected (random/with-stream* seed [role key]
                   #(m/sample-categorical (m/normalize-log-weights (mapv second ps))))]
    (retained-choices (m/get-trace (first (nth ps selected))))))

(defn pgibbs
  "Run conditional particle sweeps, pooling their normalized output. sweep
  receives [key retained seed]; the initial retained argument is nil. Seed
  role options preserve existing entry-point streams during consolidation."
  [sweep iterations {:keys [seed pick-key initial-pick first-iteration pick-offset]
                     :or {pick-key ::pick initial-pick :initial first-iteration 0 pick-offset 0}}]
  (iterations! iterations)
  (spin
   (let [seed (or seed (random/fresh-seed))
         pick #(pick-retained %1 seed pick-key %2)
         initial (await (sweep :initial nil seed))]
     (loop [retained (pick initial initial-pick) i 0 samples []]
       (if (= i iterations)
         (m/markov-measure samples)
         (let [key (+ first-iteration i)
               measure (await (sweep key retained seed))]
           (recur (pick measure (+ key pick-offset)) (inc i)
                  (into samples (normalized measure)))))))))
