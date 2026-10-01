(ns org.replikativ.foerster.cascade
  "The particle cascade (Paige, Wood, Doucet & Teh 2014): asynchronous SMC
  without barriers. Each particle runs on its own; at an observation (or a
  barrier factor) it compares its weight W with the running mean W̄ of the
  weights that have arrived there so far, itself included, and branches into
  M ∈ {⌊r⌋, ⌈r⌉} copies with E[M] = r = W/W̄, each weighted W̄ — so no
  particle ever waits for another. Copies are forks of the particle's world.

  The evidence estimate (1/K₀) Σ W over the particles that reach the end (K₀
  launched) is unbiased whatever order the particles arrive in, so slow and
  fast particles (an uneven model call, a long simulation) cost no waiting.
  `:cap` bounds the particles alive at once: copies beyond it collapse into a
  multiplicity carried by the particle (Anglican's pcascade), which counts in
  the running means and multiplies its final weight.

  Inference ends when every particle has finished — by count, not by wall
  clock, which would favour fast particles (Murray, Singh & Lee 2021). The
  branching draws come from each particle's world stream, but the running
  means depend on arrival order, so runs on a multi-threaded executor are not
  reproducible from their seed."
  (:require [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.trace :as trace]
            [replikativ.logging :as log]))

(defn- weight-of [world] (or (rtp/get-state world [:inference :log-weight]) 0.0))
(defn- multiplicity [world] (or (rtp/get-state world [:inference :multiplicity]) 1.0))
(defn- stage-of [world] (or (rtp/get-state world [:inference :barriers]) 0))

(defn- log-sum-exp [a b]
  (cond (= ##-Inf a) b
        (= ##-Inf b) a
        :else (let [top (max a b)] (+ top (Math/log (+ (Math/exp (- a top)) (Math/exp (- b top))))))))

(defn- forks
  "Fork `sp` `k` times; resolves the children."
  [sp k]
  (fn [resolve reject]
    (if (zero? k)
      (resolve [])
      (let [out (object-array k)
            remaining (atom k)
            failed? (atom false)]
        (dotimes [i k]
          ((sp/fork sp)
           (fn [child]
             (aset out i child)
             (when (zero? (swap! remaining dec)) (resolve (vec out))))
           (fn [e] (when (compare-and-set! failed? false true) (reject e)))))))))

(defn cascade
  "The particle cascade of `model` (a spin) with `n` initial particles.
  Options:

    :cap      the most particles alive at once (default 4·n); copies beyond
              it collapse into a multiplicity
    :policy   a `foerster.trace/policy` deciding the sites (default the prior)
    :concurrency  how many particles run between barriers at once (default
              32); the others wait and are resumed in random order, which
              keeps the population stable (Paige et al. 2014, §4)
    :executor the worlds' executor

  Resolves an EmpiricalMeasure of `Sample`s over the particles that reached
  the end, weighted, whose `m/log-marginal` is the evidence estimate; its
  `:cascade` holds `:launched`, `:finished`, `:peak` (most alive at once) and
  `:collapsed` (copies folded into multiplicities)."
  [model n & [{:keys [cap policy executor concurrency] :as opts}]]
  (let [cap (or cap (* 4 n))
        concurrency (or concurrency 32)]
    (when (< cap n)
      (throw (ex-info ":cap must allow the initial particles" {:type ::invalid-cap :cap cap :n n})))
    (fn [resolve reject]
      (let [[resolve reject] (sp/in-callers-world resolve reject)
            seed (or (:seed opts) (random/fresh-seed))
            policy (or policy (itrace/policy))
            root (if executor (ctx/create-execution-context :executor executor) (ctx/create-execution-context))
            session (sp/open! root {:purpose :cascade
                                    :seed seed
                                    :fork-opts {:systems :none}
                                    :retain-released? false})
            ;; {stage [log Σ W·mult, Σ mult]} of the arrivals so far
            means (atom {})
            state (atom {:alive 0 :peak 0 :collapsed 0 :spawned? false :done [] :finished? false
                         ;; first in, random out: particles ready to run wait
                         ;; here; at most `concurrency` run at once
                         :queue [] :running 0 :picks 0})
            finish! (fn [callback value]
                      (when-not (:finished? (first (swap-vals! state assoc :finished? true)))
                        ((sp/close! session)
                         (fn [_] (callback value))
                         (fn [_] (callback value)))))
            fail! #(finish! reject %)
            decide! (fn [sp]
                      (let [decision (policy sp nil)]
                        (trace/record! (:savepoint/world sp) sp decision nil)
                        decision))
            dispatch! (fn dispatch! []
                        (let [[_ after]
                              (swap-vals! state
                                          (fn [{:keys [queue running picks] :as st}]
                                            (if (and (< running concurrency) (seq queue))
                                              ;; a pure function of the state, so
                                              ;; a retried swap picks the same
                                              (let [i (random/with-stream* seed [::pick picks]
                                                        #(long (* (random/uniform01) (count queue))))]
                                                (assoc st
                                                       :queue (into (subvec queue 0 i) (subvec queue (inc i)))
                                                       :running (inc running)
                                                       :picks (inc picks)
                                                       :launch (nth queue i)))
                                              (dissoc st :launch))))]
                          (when-let [launch (:launch after)]
                            (launch)
                            (dispatch!))))
            enqueue! (fn [thunks]
                       (swap! state update :queue into thunks)
                       (dispatch!))
            stopped! (fn []
                       ;; a running particle reached a barrier or ended
                       (swap! state update :running dec)
                       (dispatch!))
            ended! (fn []
                     ;; a particle ended (finished or died): when none is left
                     (let [{:keys [alive spawned? done collapsed peak]} (swap! state update :alive dec)]
                       (when (and spawned? (zero? alive))
                         (if (empty? done)
                           (fail! (ex-info "Every particle of the cascade died" {:type ::no-survivors}))
                           (finish! resolve
                                    (assoc (m/empirical done)
                                           ;; (1/K₀) Σ W: log-marginal adds the
                                           ;; log mean over the survivors
                                           :log-normalizer (- (Math/log (count done)) (Math/log n))
                                           :cascade {:launched n :finished (count done)
                                                     :peak peak :collapsed collapsed}))))))]
        (letfn [(branch! [sp value]
                  (let [world (:savepoint/world sp)
                        k (stage-of world)
                        w (weight-of world)
                        mult (multiplicity world)
                        [log-total count] (get (swap! means update k
                                                      (fn [[lt c]]
                                                        [(log-sum-exp (or lt ##-Inf) (+ w (Math/log mult)))
                                                         (+ (or c 0.0) mult)]))
                                               k)
                        log-mean (- log-total (Math/log count))
                        ratio (if (= ##-Inf log-total) 1.0 (Math/exp (- w log-mean)))
                        u (random/in-world-stream world [::branch k] random/uniform01)
                        ceiling (Math/ceil ratio)
                        copies (long (if (< (- ceiling ratio) u) ceiling (dec ceiling)))]
                    (if (zero? copies)
                      (do (sp/abandon sp)
                          (sp/release-world! session world)
                          (ended!)
                          (stopped!))
                      (let [extra (dec copies)
                            ;; room is reserved in one step: concurrent
                            ;; branchings must not overshoot the cap together
                            [before after] (swap-vals! state
                                                       (fn [st]
                                                         (let [forked (min extra (max 0 (- cap (:alive st))))]
                                                           (-> st
                                                               (update :alive + forked)
                                                               (update :peak max (+ (:alive st) forked))
                                                               (update :collapsed + (- extra forked))))))
                            forked (- (:alive after) (:alive before))
                            collapsed (- extra forked)]
                        ;; every copy weighs W̄
                        (rtp/swap-state! world [:inference :log-weight] (constantly (if (= ##-Inf log-total) w log-mean)))
                        ((forks sp forked)
                         (fn [children]
                           ;; the particle carries the copies that found no room
                           (when (pos? collapsed)
                             (rtp/swap-state! world [:inference :multiplicity] (constantly (* mult (inc collapsed)))))
                           (enqueue! (mapv (fn [s] #(sp/resume s value)) (cons sp children)))
                           (stopped!))
                         fail!)))))
                (run-site! [sp]
                  (try
                    (if (= smc/start-site (:savepoint/site sp))
                      (sp/resume sp nil)
                      (let [{:keys [value]} (decide! sp)]
                        (if (itrace/barrier-site? sp)
                          (if (:stream (:options (:savepoint/payload sp)))
                            (fail! (ex-info "The cascade does not take stream sites" {:type ::stream-site}))
                            (branch! sp value))
                          (sp/resume sp value))))
                    (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                (spawn! [first-sp]
                  (swap! state assoc :alive n :peak n :spawned? true)
                  ((forks first-sp n)
                   (fn [children]
                     (sp/abandon first-sp)
                     (doseq [child children]
                       (rtp/swap-state! (:savepoint/world child) [:inference :particle?] (constantly true)))
                     (enqueue! (mapv (fn [child] #(run-site! child)) children)))
                   fail!))]
          (try
            (sp/install-handlers!
             root
             {sp/any-site
              (fn [s]
                (if (rtp/get-state (:savepoint/world s) [:inference :particle?])
                  (run-site! s)
                  (spawn! s)))
              sp/result-site
              (fn [{world :savepoint/world result :savepoint/payload}]
                (if (rtp/get-state world [:inference :particle?])
                  (let [w (+ (weight-of world) (Math/log (multiplicity world)))]
                    (swap! state update :done conj
                           [(m/sample-particle result (itrace/legacy-trace (rtp/get-state world [:savepoint/trace]))) w])
                    (sp/release-world! session world)
                    (ended!)
                    (stopped!))
                  ;; a program without savepoints: one deterministic run
                  (finish! resolve (m/empirical [[(m/sample-particle result {}) 0.0]]))))
              sp/error-site (fn [{error :savepoint/payload}] (fail! error))
              sp/abandoned-site (fn [_] nil)})
            (sp/start! session model)
            (catch #?(:clj Throwable :cljs :default) e
              (log/error :cascade/start-failed {:error e})
              (fail! e))))))))
