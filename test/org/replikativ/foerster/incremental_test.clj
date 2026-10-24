(ns org.replikativ.foerster.incremental-test
  "Differential checks for trace updates: a trace moved by an update must be
  the trace a from-scratch run gives for the same choices — every entry's
  value and score, the log-weight and the result. An incremental update is
  only allowed to be faster than that, never different; these checks are
  what keeps it honest (Zaiser et al., PLDI 2026, on soundness bugs in
  ad-hoc incrementalization).

  `consistent` compares two traces; the tests run it on today's update
  (`mh-step`, a replay from the target's anchor) so the check itself is
  known to hold where it must."
  (:require [clojure.test :refer [deftest is testing]]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.effects :refer [observe sample]]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.await :as aw]
            [org.replikativ.spindel.engine.addressing :as addressing]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.core :as rtc]
            [org.replikativ.spindel.spin.combinators :refer [parallel]]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.keyed :refer [keyed]]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.trace :as trace]))

(defn- await* [op]
  (let [p (promise)]
    (op #(deliver p [:ok %]) #(deliver p [:error %]))
    (let [[k v :as r] (deref p 60000 [:timeout])]
      (case k :ok v :error (throw v) (throw (ex-info "timed out" {:result r}))))))

;; --- the comparison -----------------------------------------------------------

(defn- choices
  "{address value} of every sample site that is not observed."
  [t]
  (into {} (keep (fn [a]
                   (let [{:keys [site value note]} (get-in t [:trace/entries a])]
                     (when (and (= itrace/choose-site site) (not (:observed? note)))
                       [a value]))))
        (:trace/order t)))

(defn- scores
  "{address [value log-prob]} of every choose and factor entry."
  [t]
  (into {} (map (juxt :address (juxt :value (comp :log-prob :note))))
        (itrace/entries t)))

(defn- from-scratch
  "`model` run with `constraints`, in a session of its own and without
  anchors: scoring needs none, and a model under `parallel` must not be
  forked while its children are in flight (spindel savepoints.md)."
  [model constraints]
  (let [root (ctx/create-execution-context)
        session (sp/open! root {:seed (random/fresh-seed) :fork-opts {:systems :none}
                                :retain-released? false})]
    (try
      (await* (trace/run session model (itrace/policy {:constraints constraints})
                         {:anchor? (constantly false)}))
      (finally (await* (sp/close! session))))))

(defn consistent
  "Differences between `moved` (a trace an update produced) and a
  from-scratch run of `model` constrained to `moved`'s choices: nil when
  they agree on every entry's value and score, the log-joint and the
  result."
  [model moved]
  (let [fresh (from-scratch model (choices moved))
        a (scores moved)
        b (scores fresh)
        close? (fn [x y] (or (= x y) (and (number? x) (number? y) (< (Math/abs (- (double x) (double y))) 1e-9))))
        entry-diffs (for [k (distinct (concat (keys a) (keys b)))
                          :let [[va la] (get a k) [vb lb] (get b k)]
                          :when (not (and (contains? a k) (contains? b k) (= va vb) (close? la lb)))]
                      {:address k :moved (get a k) :fresh (get b k)})]
    (not-empty
     (cond-> {}
       (seq entry-diffs) (assoc :entries (vec (take 10 entry-diffs)))
       (not (close? (itrace/log-joint moved) (itrace/log-joint fresh)))
       (assoc :log-joint [(itrace/log-joint moved) (itrace/log-joint fresh)])
       (not= (:trace/result moved) (:trace/result fresh))
       (assoc :result [(:trace/result moved) (:trace/result fresh)])))))

;; --- models: a sub-spin per point, awaited in turn or under `parallel` ---------

(def ^:private xs (mapv #(- (* 0.5 %) 5.0) (range 21)))
(def ^:private ys (mapv #(+ (* -2.0 %) 1.0 (if (zero? (mod (long (* 2 %)) 7)) 40.0 0.0)) xs))

(defn- point [m b x y i]
  (spin
   (let [o (sample (dist/flip 0.1) :id [:o i])]
     (observe (dist/normal (if o 0.0 (+ (* m x) b)) (if o 100.0 0.5)) y :id [:y i])
     o)))

(defn- regression-items []
  (spin
   (let [m (sample (dist/normal 0.0 1.0) :id :m)
         b (sample (dist/normal 0.0 1.0) :id :b)]
     (loop [i 0 n-out 0]
       (if (= i (count xs))
         [m b n-out]
         (let [o (aw/await (addressing/with-key i (point m b (xs i) (ys i) i)))]
           (recur (inc i) (if o (inc n-out) n-out))))))))

(defn- regression-parallel []
  (spin
   (let [m (sample (dist/normal 0.0 1.0) :id :m)
         b (sample (dist/normal 0.0 1.0) :id :b)
         os (aw/await (apply parallel (map-indexed (fn [i x] (addressing/with-key i (point m b x (ys i) i))) xs)))]
     [m b (count (filter true? os))])))

(defn- flip-at [address]
  {:select (fn [_ _] {:targets #{address} :log-selection (fn [_] 0.0)})
   :propose (fn [_ old] {:value (not (:value old)) :symmetric? true})})

(defn- shift-at [address]
  {:select (fn [_ _] {:targets #{address} :log-selection (fn [_] 0.0)})
   :propose (fn [_ old] {:value (+ (:value old) 0.3) :symmetric? true})})

(defn- check-moves
  "Run `moves` (mh-step options) in turn from a generated trace of `model`,
  each accepted or not; after each, the current trace must be consistent."
  [model moves]
  (let [t0 (:trace (await* (gfi/generate (model) {})))]
    (try
      (is (nil? (consistent (model) t0)) "the generated trace")
      (reduce (fn [t [i opts]]
                (let [{t' :trace} (await* (itrace/mh-step t (assoc opts :iteration i)))]
                  (is (nil? (consistent (model) t')) (str "after move " i))
                  t'))
              t0 (map-indexed vector moves))
      (finally (await* (gfi/close! t0))))))

(deftest replay-moves-are-consistent
  ;; a replay starts from an anchor inside the moved item, so the model
  ;; awaits its items in turn (anchors inside `parallel` children are forks
  ;; taken while siblings are in flight)
  (random/set-seed! 11)
  (let [root (ctx/create-execution-context)]
    (try
      (binding [rtc/*execution-context* root]
        (check-moves regression-items
                     [(flip-at [:o 3]) (flip-at [:o 17]) (shift-at :b)
                      (flip-at [:o 3]) (shift-at :m) (flip-at [:o 0])]))
      (finally (ctx/stop-context! root)))))

(deftest a-parallel-model-scores-from-scratch
  (random/set-seed! 12)
  (let [root (ctx/create-execution-context)]
    (try
      (binding [rtc/*execution-context* root]
        (let [t (from-scratch (regression-parallel) {})]
          (is (= (* 2 (count xs)) (count (filter vector? (keys (:trace/entries t))))))
          (is (nil? (consistent (regression-parallel) t)))))
      (finally (ctx/stop-context! root)))))

;; --- incremental: points as one keyed loop --------------------------------------

(defn- keyed-point [m b x y i]
  (spin
   (let [o (sample (dist/flip 0.1) :id [:points i :o])]
     (observe (dist/normal (if o 0.0 (+ (* m x) b)) (if o 100.0 0.5)) y :id [:points i :y])
     o)))

(defn- regression-keyed []
  (spin
   (let [m (sample (dist/normal 0.0 1.0) :id :m)
         b (sample (dist/normal 0.0 1.0) :id :b)
         os (aw/await (keyed :points (range (count xs)) (fn [i] (keyed-point m b (xs i) (ys i) i))))]
     [m b (count (filter true? os))])))

(defn- keyed-flip [i] (flip-at [:points i :o]))

(deftest incremental-moves-are-consistent
  ;; flips of one point re-run that key only (update-step); a move of a root
  ;; parameter replays from its anchor. Every trace must equal a
  ;; from-scratch run with its choices, and an incremental move's ratio
  ;; must equal the one computed over the whole trace.
  (random/set-seed! 13)
  (let [root (ctx/create-execution-context)]
    (try
      (binding [rtc/*execution-context* root]
        (let [t0 (:trace (await* (gfi/generate (regression-keyed) {})))]
          (try
            (is (= 2 (count (filter :savepoint (vals (:trace/entries t0)))))
                "anchors only at the root's sites")
            (reduce (fn [t [i opts]]
                      (let [{t' :trace r :log-ratio a :accepted?}
                            (await* (itrace/update-step t (assoc opts :iteration i :keep-old? true)))]
                        ;; a rejected move hands back the old trace
                        (when a
                          (is (< (Math/abs (- r (itrace/mh-log-ratio t t' (fn [_] 0.0)))) 1e-9)
                              (str "move " i ": the ratio over changed entries is the full one")))
                        (is (nil? (consistent (regression-keyed) t')) (str "after move " i))
                        t'))
                    t0 (map-indexed vector [(keyed-flip 3) (keyed-flip 17) (shift-at :b)
                                            (keyed-flip 3) (shift-at :m) (keyed-flip 0)]))
            (finally (await* (gfi/close! t0))))))
      (finally (ctx/stop-context! root)))))

;; --- a mixture: Gibbs over one key's assignment ---------------------------------------

(defn- mix-point [w mu0 mu1 x i]
  (spin
   (let [z (sample (dist/flip w) :id [:points i :z])]
     (observe (dist/normal (if z mu1 mu0) 1.0) x :id [:points i :x])
     z)))

(defn- mixture [xs]
  (spin
   (let [w (sample (dist/beta 2.0 2.0) :id :w)
         mu0 (sample (dist/normal -2.0 1.0) :id :mu0)
         mu1 (sample (dist/normal 2.0 1.0) :id :mu1)
         zs (aw/await (keyed :points (range (count xs)) (fn [i] (mix-point w mu0 mu1 (xs i) i))))]
     (count (filter true? zs)))))

(deftest gibbs-over-an-assignment-is-exact
  ;; with the weight and means fixed, each assignment's posterior is known in
  ;; closed form; sweeps of update-gibbs must reproduce it
  (random/set-seed! 21)
  (let [root (ctx/create-execution-context)
        xs [-1.2 0.3 0.9 2.1]
        fixed {:w 0.4 :mu0 -1.0 :mu1 1.5}
        sweeps 800]
    (try
      (binding [rtc/*execution-context* root]
        (let [t0 (:trace (await* (gfi/generate (mixture xs) fixed)))
              counts (loop [t t0 s 0 it 0 counts (vec (repeat (count xs) 0))]
                       (if (= s sweeps)
                         counts
                         (let [[t it] (reduce (fn [[t it] i]
                                                [(:trace (await* (itrace/update-gibbs t [:points i :z]
                                                                                      {:iteration it :constraints fixed})))
                                                 (inc it)])
                                              [t it] (range (count xs)))]
                           (recur t (inc s) it
                                  (mapv (fn [c i] (if (get-in t [:trace/entries [:points i :z] :value]) (inc c) c))
                                        counts (range (count xs)))))))
              lik (fn [x mu] (Math/exp (dist/logpdf (dist/normal mu 1.0) x)))
              exact (mapv (fn [x] (let [a (* 0.4 (lik x 1.5)) b (* 0.6 (lik x -1.0))] (/ a (+ a b)))) xs)]
          (doseq [[p q] (map vector (map #(/ % (double sweeps)) counts) exact)]
            ;; four standard errors at 800 sweeps
            (is (< (Math/abs (- p q)) (* 4 (Math/sqrt (/ (* q (- 1 q)) sweeps)) 1.2)) (str p " vs " q)))))
      (finally (ctx/stop-context! root)))))

(deftest a-mixture-chain-stays-consistent
  ;; Gibbs over assignments (one key each) and random-walk moves on the
  ;; weight and means (root sites, replayed): every trace a from-scratch run
  (random/set-seed! 22)
  (let [root (ctx/create-execution-context)
        xs [-2.3 -1.8 -2.6 1.9 2.4 -1.1 2.2 -2.0]
        rw (fn [a] {:select (fn [_ _] {:targets #{a} :log-selection (fn [_] 0.0)})
                    :propose (itrace/random-walk-proposal 0.3)})]
    (try
      (binding [rtc/*execution-context* root]
        (let [t0 (:trace (await* (gfi/generate (mixture xs) {})))]
          (reduce (fn [[t it] s]
                    (let [[t it] (reduce (fn [[t it] i]
                                           [(:trace (await* (itrace/update-gibbs t [:points i :z] {:iteration it}))) (inc it)])
                                         [t it] (range (count xs)))
                          [t it] (reduce (fn [[t it] a] [(:trace (await* (itrace/update-step t (assoc (rw a) :iteration it)))) (inc it)])
                                         [t it] [:w :mu0 :mu1])]
                      (is (nil? (consistent (mixture xs) t)) (str "after sweep " s))
                      [t it]))
                  [t0 0] (range 4))))
      (finally (ctx/stop-context! root)))))
