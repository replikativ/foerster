;; # Inference in a Running Application

;; A queue already has a job in flight. We want to infer its service speed
;; from noisy completion reports, using the same scheduling function as the
;; application. Each particle starts from the caller's managed queue state
;; and records its own possible history. The caller keeps running with its
;; original state. This example accompanies the Foerster LAFI abstract.

(ns foerster.queue
  (:require [org.replikativ.foerster.core :as infer]
            [org.replikativ.foerster.dist :refer [categorical normal]]
            [org.replikativ.foerster.effects :refer [sample observe]]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.spindel.atom :as sa]
            [org.replikativ.spindel.core :as sp]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.spin.cps :refer [spin]]))

(def initial-state {:free-at 1.0 :jobs [{:id :in-flight :finish 1.0}]})
(def readings [{:id :a :arrival 0.5 :work 1.0 :reported 1.9}
               {:id :b :arrival 1.2 :work 0.7 :reported 2.7}
               {:id :c :arrival 3.0 :work 1.2 :reported 4.1}])

;; ## Reuse the application's scheduler
;;
;; `queue` is a Spindel runtime atom. Its state belongs to the current world;
;; an ordinary `clojure.core/atom` would be shared across particles.
;; Dereferencing a runtime atom reads state; it does not wait for a spin.

(defn serve! [queue {:keys [id arrival work]} speed jitter]
  (let [start (max arrival (:free-at @queue))
        finish (+ start (/ (* work jitter) speed))]
    (swap! queue (fn [state]
                   (-> state
                       (assoc :free-at finish)
                       (update :jobs conj {:id id :arrival arrival :work work
                                           :jitter jitter :start start :finish finish}))))
    finish))

;; ## Add uncertainty and evidence
;;
;; The model samples one shared speed and one service multiplier per job.
;; `observe` scores the reported completion against the scheduler's output.
;; Native `doseq` expands inside this `spin` scope; `serve!` is an ordinary
;; function with no suspension effects. An effectful helper would need its
;; own `spin` and an explicit `await` at the call site.

(defn queue-model [queue observations]
  (spin
   (let [speed (sample (categorical {0.8 0.5, 1.4 0.5}) :id :speed)]
     (doseq [{:keys [id reported] :as job} observations]
       (let [jitter (sample (categorical {0.8 0.5, 1.2 0.5}) :id [:jitter id])
             finish (serve! queue job speed jitter)]
         (observe (normal finish 0.35) reported :id [:completion id])))
     {:speed speed :queue @queue})))

;; ## An independent reference
;;
;; There are only sixteen combinations for these three jobs. A pure
;; recurrence, independent of `serve!` and Foerster, enumerates their
;; likelihoods. Equal prior masses and Normal normalizers cancel in the ratio.
;; This deliberately small fixture validates semantics, not performance.

(defn reference-p-fast [initial observations]
  (let [paths
        (for [speed [0.8 1.4] a [0.8 1.2] b [0.8 1.2] c [0.8 1.2]]
          (let [finishes (rest (reductions
                                (fn [free [job jitter]]
                                  (+ (max free (:arrival job))
                                     (/ (* (:work job) jitter) speed)))
                                (:free-at initial)
                                (map vector observations [a b c])))
                log-likelihood
                (reduce + (map (fn [job finish]
                                 (let [z (/ (- (:reported job) finish) 0.35)]
                                   (* -0.5 z z))) observations finishes))]
            {:speed speed :weight (Math/exp log-likelihood)}))]
    (/ (reduce + (map :weight (filter #(= 1.4 (:speed %)) paths)))
       (reduce + (map :weight paths)))))

(defn valid-history? [{:keys [speed queue]} initial]
  (let [history (:jobs queue)]
    (and (= 4 (count history))
         (= (:jobs initial) (subvec history 0 1))
         (= [:a :b :c] (mapv :id (rest history)))
         (= (:free-at queue) (:finish (last history)))
         (every? true?
                 (map (fn [previous job]
                        (and (= (:start job) (max (:finish previous) (:arrival job)))
                             (< (Math/abs (- (:finish job)
                                             (+ (:start job)
                                                (/ (* (:work job) (:jitter job)) speed))))
                                1.0e-10)))
                      history (rest history))))))

;; ## Infer in forks, inspect the caller
;;
;; Create the live queue before inference. `:world-policy :fork` imports its
;; managed state into each particle. Fresh worlds would not contain this atom.
;; The result holds settled values and world descriptors, not live particles.
;; Always retire the execution context when its application session ends.

(defn check-queue [seed]
  (let [world (sp/create-execution-context)]
    (try
      (sp/with-context world
        (let [queue (sa/atom initial-state)
              posterior @(infer/smc-infer (queue-model queue readings) 1000
                                          {:world-policy :fork :seed seed
                                           :resample-threshold 0.8})
              values (mapv m/get-value (m/get-contexts posterior))
              p-fast (:mean (infer/query posterior #(if (= 1.4 (:speed %)) 1.0 0.0)))
              exact (reference-p-fast initial-state readings)]
          (assert (= initial-state @queue) "Inference changed the caller's queue")
          (assert (every? #(valid-history? % initial-state) values) "Invalid particle history")
          (assert (> (count (set (map :queue values))) 1) "Expected distinct speculative histories")
          (assert (< (Math/abs (- p-fast exact)) 0.08) "SMC disagrees with enumeration")
          (assert (= 1000 (count (m/world-descriptors posterior))))
          {:seed seed :p-fast p-fast :reference exact :caller-state @queue
           :barriers (mapv #(select-keys % [:ess :resampled?]) (:history posterior))}))
      (finally (ctx/stop-context! world)))))

(mapv check-queue [17 31 73])

;; The initial backlog matters: ignoring it would change P(fast) from about
;; 0.566 to 0.026 for the same observations. This is a different model input,
;; not an identified causal intervention.

{:with-backlog (reference-p-fast initial-state readings)
 :without-backlog (reference-p-fast (assoc initial-state :free-at 0.0) readings)}

;; Forking isolates managed runtime state and registered forkable systems.
;; It does not isolate arbitrary I/O, ordinary host atoms or foreign mutable
;; buffers. Numerical blocks, causal models and amortized artifacts each add
;; their own contracts; they are separate routes through the documentation.
;; See [Models in Worlds](https://replikativ.github.io/foerster/foerster.worlds.html)
;; for attached systems and resource rules, and
;; [Concepts](https://github.com/replikativ/foerster/blob/main/doc/concepts.md)
;; for the distinction between a world, trace, measure, population and block.
