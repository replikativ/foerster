(ns org.replikativ.foerster.sweep-enumeration-test
  "Exact checks of particle MCMC kernels by enumeration.

  Every decision an inference step draws — a discrete site's value, an
  ancestor, the site a move selects, its acceptance — goes through
  `random/decide`. With an oracle installed, a run takes the outcomes it is
  told, so one sweep can be run once per path through its decisions, each
  path with its exact probability. For a model with finitely many traces
  that gives the sweep's transition matrix between retained trajectories,
  and a valid particle Gibbs kernel must leave the exact posterior
  invariant: πK = π to rounding. Statistical tests cannot see biases of
  1e-4..1e-3 in total variation; this can."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.benchmark-test :as b]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

;; -----------------------------------------------------------------------------
;; Enumeration
;; -----------------------------------------------------------------------------

(defn enumerate-paths
  "Run `(run!)` once per path through its decisions. Returns {result
  probability}: the exact law of `(run!)`'s result.

  A run answers the decisions it was told to (by key: a decision's stream
  and position in it, so concurrent particles may decide in any order) and
  takes the first possible outcome of any other, recording them in the
  order they happen. From a run whose decisions were d1 … dm, the paths
  that agree with it up to d(i-1) and differ at di are explored by forcing
  d1 … d(i-1) to their outcomes and di to each other possible one. A
  decision happens after every decision it depends on, so these sets of
  paths are disjoint and together complete: each path runs once. Its
  probability is the product of its decisions'."
  [run!]
  (loop [stack [{}] law {} runs 0]
    (if-let [forced (peek stack)]
      (let [queried (atom [])
            result (random/with-oracle
                     (fn [k probs]
                       (let [f (get forced k)
                             v (if (and f (< f (count probs)) (pos? (nth probs f)))
                                 f
                                 (first (keep-indexed #(when (pos? %2) %1) probs)))]
                         (swap! queried conj [k v probs])
                         v))
                     run!)
            decisions @queried
            p (reduce * 1.0 (map (fn [[_ v probs]] (nth probs v)) decisions))
            children (loop [i 0 prefix forced acc []]
                       (if (= i (count decisions))
                         acc
                         (let [[k v probs] (nth decisions i)
                               acc (if (contains? forced k)
                                     acc
                                     (into acc (for [alt (range (count probs))
                                                     :when (and (not= alt v) (pos? (nth probs alt)))]
                                                 (assoc prefix k alt))))]
                           (recur (inc i) (assoc prefix k v) acc))))]
        (recur (into (pop stack) children)
               (update law result (fnil + 0.0) p)
               (inc runs)))
      law)))

;; -----------------------------------------------------------------------------
;; Models with finitely many trajectories, and their exact posteriors
;; -----------------------------------------------------------------------------

(def ^:private steps
  ;; per step: P(x = 1 | previous x), P(y observed | x)
  [{:x (fn [_] 0.5) :y (fn [x] (if (= 1 x) 0.8 0.3)) :obs true}
   {:x (fn [p] (if (= 1 p) 0.7 0.2)) :y (fn [x] (if (= 1 x) 0.9 0.4)) :obs false}
   {:x (fn [p] (if (= 1 p) 0.6 0.3)) :y (fn [x] (if (= 1 x) 0.75 0.2)) :obs true}])

(defn chain
  "A hidden Markov chain of `k` binary steps, each observed once."
  [k]
  (let [steps (subvec steps 0 k)
        ks (mapv #(keyword (str "x" (inc %))) (range k))]
    {:addresses ks
     :model (fn []
              (spin
               (loop [i 0 prev nil xs []]
                 (if (= i k)
                   xs
                   (let [{:keys [x y obs]} (nth steps i)
                         xi (sample (dist/bernoulli (x prev)) :id (nth ks i))]
                     (observe (dist/flip (y xi)) obs :id [:y i])
                     (recur (inc i) xi (conj xs xi)))))))
     :posterior (let [states (reduce (fn [acc _] (for [s acc b [0 1]] (conj s b))) [[]] (range k))
                      joint (fn [xs]
                              (reduce * 1.0 (map-indexed
                                             (fn [i xi]
                                               (let [{:keys [x y obs]} (nth steps i)
                                                     px (x (when (pos? i) (nth xs (dec i))))
                                                     py (y xi)]
                                                 (* (if (= 1 xi) px (- 1.0 px))
                                                    (if obs py (- 1.0 py)))))
                                             xs)))
                      z (reduce + (map joint states))]
                  (zipmap states (map #(/ (joint %) z) states)))}))

(defn- run-briefly
  "`b/run-infer`, failing after 30 s instead of 300: a path that hangs is a
  bug to see, not to wait out."
  [make-op]
  (let [r (deref (future (b/run-infer 1 make-op)) 30000 ::timeout)]
    (if (= ::timeout r)
      (throw (ex-info "A sweep did not finish" {:type ::hung}))
      r)))

(defn- sweep-law
  "The exact law of the next retained trajectory after one conditional sweep
  of `n` particles from `retained`, under `opts`.

  With moves the sweep first draws the retained path backwards through them,
  and the rest of the sweep depends on that only through the paths it
  yields: the two halves are enumerated apart and mixed, which costs the sum
  of their paths instead of the product."
  [{:keys [model addresses]} n retained opts]
  (let [seed [7 7]
        forward (fn [extra]
                  (enumerate-paths
                   (fn []
                     (run-briefly
                      (fn []
                        (fn [resolve reject]
                          ((smc/smc (model) n (merge {:seed seed} opts extra {:retained retained}))
                           (fn [measure]
                             (let [ps (m/get-particles measure)
                                   i (random/with-stream* [7 8] ::pick
                                       #(m/sample-categorical
                                         (m/normalize-log-weights (mapv second ps))))
                                   choices (smc/retained-choices (m/get-trace (first (nth ps i))))]
                               (resolve (mapv choices addresses))))
                           reject)))))))]
    (if-let [rejuvenate (:rejuvenate opts)]
      (let [anchors (:anchors opts)
            backward (enumerate-paths
                      (fn []
                        (run-briefly
                         (fn []
                           (@#'smc/retained-stages (model) retained nil rejuvenate
                                                   (@#'smc/anchor-predicate anchors) (:lag anchors) seed)))))]
        (reduce-kv (fn [law paths p]
                     (merge-with + law (update-vals (forward {::smc/retained-paths paths}) #(* p %))))
                   {} backward))
      (forward {}))))

(defn stationarity-gap
  "Total variation between π and πK for the sweep kernel K of `chain`."
  [{:keys [addresses posterior] :as chain} n opts]
  (let [states (keys posterior)
        kernel (into {} (for [s states] [s (sweep-law chain n (zipmap addresses s) opts)]))
        pi-k (into {} (for [t states]
                        [t (reduce + (for [s states] (* (posterior s) (get-in kernel [s t] 0.0))))]))]
    {:kernel kernel
     :row-sums (into {} (for [s states] [s (reduce + (vals (kernel s)))]))
     :tv (* 0.5 (reduce + (for [t states] (Math/abs (- (pi-k t) (posterior t))))))}))

(deftest the-enumerator-recovers-a-known-law
  (testing "a Bernoulli(0.3) draw and a fair choice among three"
    (is (= {0 0.7 1 0.3}
           (update-vals (enumerate-paths #(if (random/below? 0.3) 1 0)) #(/ (Math/round (* % 1e12)) 1e12))))
    (let [law (enumerate-paths #(m/sample-categorical [0.2 0.5 0.3]))]
      (is (< (Math/abs (- 0.5 (get law 1))) 1e-15)))))

(deftest particle-gibbs-leaves-the-posterior-invariant
  (let [{:keys [tv row-sums]} (stationarity-gap (chain 2) 2 {})]
    (is (every? #(< (Math/abs (- 1.0 %)) 1e-12) (vals row-sums)) (str "rows " row-sums))
    (is (< tv 1e-12) (str "TV(πK, π) " tv))))

(deftest particle-gibbs-with-resample-move-leaves-the-posterior-invariant
  ;; ~1.5 min. Two stages cannot tell this from the unmoved retained particle
  ;; (that is exact there too); three stages with a lag window can, and take
  ;; ~20 min: `clojure -M:test:enumerate` (experiments/…/enumerate.clj).
  (doseq [opts [{:anchors :all :rejuvenate {:moves 1}}]]
    (testing (pr-str opts)
      (let [{:keys [tv row-sums]} (stationarity-gap (chain 2) 2 opts)]
        (is (every? #(< (Math/abs (- 1.0 %)) 1e-12) (vals row-sums)) (str "rows " row-sums))
        (is (< tv 1e-12) (str "TV(πK, π) " tv))))))
