(ns org.replikativ.foerster.enumerate
  "Exactness of particle Gibbs with resample-move, by enumeration.

  Not part of the test suite (it takes about 20 minutes): `clojure -M:test:enumerate`.

  A three-step binary hidden Markov chain, two particles, one move per
  barrier over a lag-1 window. Every path through one conditional sweep is
  enumerated with its probability (`sweep-enumeration-test`), giving the
  sweep's exact transition matrix K over the eight trajectories; a valid
  kernel leaves the posterior π invariant. The retained trajectory is drawn
  backwards through the moves (`smc/retained-stages`): TV(πK, π) is 6·10⁻¹⁶.
  Keeping the retained particle unmoved instead, as before, gives
  1.7·10⁻⁴ — a bias no statistical test of this size detects. Exits 1 if
  the gap exceeds 1e-12."
  (:require [org.replikativ.foerster.sweep-enumeration-test :as t]))

(defn -main [& _]
  (let [t0 (System/nanoTime)
        {:keys [tv row-sums]} (t/stationarity-gap (t/chain 3) 2 {:anchors {:lag 1}
                                                                 :rejuvenate {:moves 1}})
        complete? (every? #(< (Math/abs (- 1.0 %)) 1e-12) (vals row-sums))]
    (println "TV(πK, π)" tv "rows complete?" complete?
             "in" (quot (- (System/nanoTime) t0) 1000000000) "s")
    (shutdown-agents)
    (System/exit (if (and complete? (< tv 1e-12)) 0 1))))
