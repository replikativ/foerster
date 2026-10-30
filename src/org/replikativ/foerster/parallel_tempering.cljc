(ns org.replikativ.foerster.parallel-tempering
  "Non-reversible parallel tempering (Syed, Bouchard-Côté, Deligiannidis &
  Doucet 2022) on a model that is one block (`foerster.population`).

  N + 1 chains sit on the path π_β ∝ q^(1−β)·t^β from the block's draw
  density q (the reference) to its target t, at 0 = β_0 < … < β_N = 1. A
  scan explores every chain at its own β (the reference chain by an exact
  draw, the others by random-walk Metropolis), then offers swaps to
  neighbours: even pairs on even scans, odd pairs on odd scans. A swap of
  chains i and i+1 is accepted with min(1, exp((β_{i+1} − β_i)·(V(x_i) −
  V(x_{i+1})))), V = log t − log q.

  The schedule is tuned in rounds of doubling length: the swap rejection
  rates r_i give the cumulative barrier Λ(β_i) = Σ_{j<i} r_j, and the next
  round spaces its β evenly in Λ. Λ(1) is the global communication barrier.

  The last round's states of the target chain are the posterior sample; the
  log evidence (q normalized) is the stepping-stone estimate
  Σ_i log mean exp((β_{i+1} − β_i)·V) over chain i's states."
  (:require [org.replikativ.foerster.scope :as scope]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.block-mh :as block-mh]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.population :as pop]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]))

(def ^:private target-acceptance 0.234)

(defn rest-of
  "V(θ) = log t(θ) − log q(θ) of a block law."
  [logpdf draw-logpdf]
  (fn [q] (- (logpdf q) (draw-logpdf q))))

(defn evenly-in-barrier
  "`n` + 1 temperatures spaced evenly in the cumulative barrier through the
  points (β_i, Λ_i), by inverting its piecewise-linear interpolation."
  [betas lambdas n]
  (let [total (peek lambdas)]
    (if-not (pos? total)
      (mapv #(/ (double %) n) (range (inc n)))
      (mapv (fn [k]
              (cond
                (zero? k) 0.0
                (= k n) 1.0
                :else
                (let [target (* total (/ (double k) n))
                      i (or (first (filter #(>= (nth lambdas %) target) (range 1 (count lambdas))))
                            (dec (count lambdas)))
                      l0 (nth lambdas (dec i)) l1 (nth lambdas i)
                      b0 (nth betas (dec i)) b1 (nth betas i)]
                  (if (> l1 l0)
                    (+ b0 (* (- b1 b0) (/ (- target l0) (- l1 l0))))
                    b1))))
            (range (inc n))))))

(defn swap-log-ratio
  "log acceptance of swapping the states of chains at `b0` < `b1` holding
  states with V values `v0` and `v1`."
  [b0 b1 v0 v1]
  (* (- b1 b0) (- v0 v1)))

(defn- end-visit
  "A replica's record after it sits on `chain` of `n`: it reaches the target
  only coming from the reference, and a round trip counts when it returns to
  the reference from there."
  [{:keys [last trips] :as rec} chain n]
  (cond
    (and (= chain n) (= last :reference)) (assoc rec :last :target)
    (zero? chain) (cond-> (assoc rec :last :reference)
                    (= last :target) (assoc :trips (inc trips)))
    :else rec))

(defn scan
  "One scan of round `r`, number `s`: every chain explored by
  `(explore i beta x)` -> {:x :accepted} under its stream, then the swaps
  of parity s. `st` holds :xs :betas :labels :replicas and the round's
  tallies :rej :acc :swaps :swapped :vs :samples."
  [{:keys [betas] :as st} r s v explore seed]
  (let [n (dec (count betas))
        explored (mapv (fn [i]
                         (random/with-stream* seed [::explore r s i]
                           #(explore i (nth betas i) (nth (:xs st) i))))
                       (range (inc n)))
        xs (mapv :x explored)
        vxs (mapv v xs)
        st (-> st
               (assoc :xs xs)
               (update :acc #(mapv + % (map :accepted explored)))
               (update :vs #(mapv conj % vxs))
               (update :samples conj (peek xs)))
        st (reduce (fn [st i]
                     (let [{:keys [xs labels]} st
                           log-a (swap-log-ratio (nth betas i) (nth betas (inc i))
                                                 (v (nth xs i)) (v (nth xs (inc i))))
                           swap? (or (>= log-a 0.0)
                                     (random/with-stream* seed [::swap r s i]
                                       #(random/accept-log? log-a)))]
                       (cond-> (-> st
                                   (update-in [:rej i] + (- 1.0 (Math/exp (min 0.0 log-a))))
                                   (update :swaps inc))
                         swap? (-> (update :swapped inc)
                                   (assoc :xs (assoc xs i (nth xs (inc i)) (inc i) (nth xs i)))
                                   (assoc :labels (assoc labels i (nth labels (inc i)) (inc i) (nth labels i)))))))
                   st
                   (range (mod s 2) n 2))]
    (update st :replicas
            (fn [reps] (reduce (fn [reps chain] (update reps (nth (:labels st) chain) end-visit chain n))
                               reps (range (inc n)))))))

(defn stepping-stone
  "log Z of the target relative to the reference: Σ_i log mean
  exp((β_{i+1} − β_i)·V) over chain i's V values `vs`."
  [betas vs]
  (reduce + (for [i (range (dec (count betas)))
                  :let [d (- (nth betas (inc i)) (nth betas i))
                        terms (mapv #(* d %) (nth vs i))]]
              (- (weights/log-sum-exp terms) (Math/log (count terms))))))

(defn rounds
  "Run `n-rounds` rounds of 1, 2, 4, … scans from the chains' states `xs`,
  retuning the schedule after each. `explore-at` (fn [scales] (fn [i beta x]
  …)) builds a round's explorer from per-chain step scales, which adapt
  toward 0.234 acceptance from round to round. Returns the last round's
  {:samples :log-z :betas :barrier :local-barrier :round-trips
  :swap-acceptance :scans}."
  [xs v explore-at {:keys [n-rounds seed dim steps]}]
  (let [n (dec (count xs))]
    (loop [r 0 betas (mapv #(/ (double %) n) (range (inc n))) xs (mapv vec xs)
           log-scales (vec (repeat (inc n) (Math/log (/ 0.5 (Math/sqrt dim)))))
           ;; replica i starts on chain i: the one on the reference is there
           replicas (assoc (vec (repeat (inc n) {:last nil :trips 0})) 0 {:last :reference :trips 0})
           labels (vec (range (inc n)))]
      (let [scans (bit-shift-left 1 r)
            st (reduce (fn [st s] (scan st r s v (explore-at (mapv #(Math/exp %) log-scales)) seed))
                       {:xs xs :betas betas :labels labels :replicas replicas
                        :rej (vec (repeat n 0.0)) :acc (vec (repeat (inc n) 0))
                        :swaps 0 :swapped 0 :vs (vec (repeat (inc n) [])) :samples []}
                       (range scans))
            ;; each pair is offered a swap on every other scan
            offers (fn [i] (count (filter #(= (mod % 2) (mod i 2)) (range scans))))
            rates (mapv (fn [i] (if (pos? (offers i)) (/ (nth (:rej st) i) (offers i)) 0.0)) (range n))
            lambdas (vec (reductions + 0.0 rates))
            acc-rates (mapv #(/ (double %) (* scans steps)) (:acc st))]
        (if (= (inc r) n-rounds)
          {:samples (:samples st)
           :log-z (stepping-stone betas (:vs st))
           :betas betas
           :barrier (peek lambdas)
           :local-barrier (mapv vector (rest betas) rates)
           :round-trips (reduce + (map :trips (:replicas st)))
           :swap-acceptance (if (pos? (:swaps st)) (/ (double (:swapped st)) (:swaps st)) 0.0)
           :scans scans}
          (recur (inc r) (evenly-in-barrier betas lambdas n) (:xs st)
                 (mapv (fn [ls a i] (if (zero? i) ls (+ ls (- a target-acceptance))))
                       log-scales acc-rates (range))
                 (:replicas st) (:labels st)))))))

(defn parallel-tempering
  "Non-reversible parallel tempering of `model` (a spin) that is one block
  with a `:sample` and `:sample-log-density` (the reference). Options:

    :chains   N + 1, the number of temperatures (default 10; about twice the
              barrier Λ the result reports is enough)
    :rounds   tuning rounds of 1, 2, 4, … scans (default 10); the last
              round's 2^(rounds−1) target-chain states are the sample
    :steps    random-walk steps per chain per scan (default the dimension)
    :executor the worlds' executor
    :seed     fixes every draw (default: one from the process generator)

  Returns a CPS operation resolving an EmpiricalMeasure of `Sample`s, each
  written back to a trace once per distinct θ, whose `m/log-marginal` is the
  stepping-stone evidence; `:temperatures`, `:barrier` Λ, `:local-barrier`
  [[β r] …], `:round-trips` and `:swap-acceptance` diagnose the run."
  [model & [{:keys [chains steps executor seed] n-rounds :rounds :or {chains 10 n-rounds 10}}]]
  (when-not (and (int? chains) (>= chains 2))
    (throw (ex-info ":chains is at least 2" {:type ::invalid-chains :chains chains})))
  (when-not (pos-int? n-rounds)
    (throw (ex-info ":rounds is a positive count" {:type ::invalid-rounds :rounds n-rounds})))
  (when-not (or (nil? steps) (pos-int? steps))
    (throw (ex-info ":steps is a positive count" {:type ::invalid-steps :steps steps})))
  (scope/run
   (fn [resources]
     (fn [resolve reject]
       (let [seed (or seed (random/fresh-seed))
             {:keys [operation]} (gfi/run-policy* model (itrace/policy {:temperature 0.0})
                                                  (cond-> {:inference-scope resources :seed (sp/derive-seed seed ::run 0)}
                                                    executor (assoc :executor executor)))]
         (operation
          (fn [template]
            (try
              (cond
                (:trace/error template)
                (reject (ex-info "The program failed" {:type ::model-failed} (:trace/error template)))

                (pop/single-block [template])
                (let [{:keys [address dist]} (pop/single-block [template])
                      dim (count (get-in template [:trace/entries address :value]))
                      steps (or steps dim)
                      v (rest-of #(dist/logpdf dist %) #(dist/-draw-logpdf dist %))
                      explore-at (fn [scales]
                                   (fn [i beta x]
                                     (if (zero? beta)
                                       {:x (vec (dist/draw dist)) :accepted steps}
                                       (let [{:keys [q accepted]} (block-mh/walk (block-mh/density dist beta) x
                                                                                 (vec (repeat dim (nth scales i))) steps)]
                                         {:x q :accepted accepted}))))
                      xs (vec (repeat chains (vec (get-in template [:trace/entries address :value]))))
                      res (rounds xs v explore-at {:n-rounds n-rounds :seed seed :dim dim :steps steps})]
                  ((pop/write-back-all template address (:samples res))
                   (fn [traces]
                     (resolve
                      (assoc (m/empirical (mapv (fn [t] [(m/sample-particle (:trace/result t) (itrace/legacy-trace t)) 0.0])
                                                traces))
                             :log-normalizer (:log-z res)
                             :temperatures (:betas res)
                             :barrier (:barrier res)
                             :local-barrier (:local-barrier res)
                             :round-trips (:round-trips res)
                             :swap-acceptance (:swap-acceptance res))))
                   #(reject %)))

                :else
                (reject (ex-info "Parallel tempering runs a model that is one block site (with :sample and :sample-log-density) for now"
                                 {:type ::not-one-block})))
              (catch #?(:clj Throwable :cljs :default) e (reject e))))
          #(reject %)))))))
