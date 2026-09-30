(ns org.replikativ.foerster.smc
  "Sequential Monte Carlo as a savepoint handler.

  The program runs once in a session. The first savepoint it reaches is
  forked into N worlds, one per particle; every sample site is decided by
  the scoring policy of `foerster.trace` and recorded in its world's trace;
  every observe is scored and then PARKED — its savepoint left pending. When
  every particle is parked or has returned, the population is resampled
  (when its ESS is below the threshold, folding the log mean weight into the
  evidence): the parked savepoints of the chosen ancestors are forked into
  the next generation's worlds, the old ones are abandoned, and the new ones
  resume. A particle that returned early is carried along with its final
  weight. No coordinator, no barrier thread: the barrier is the arrival of
  the last particle.

  Worlds are forks of savepoints, so particles are frozen, coherent worlds
  (see docs/forking.md) and inherit whatever world state the program holds."
  (:require [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.context :as ctx]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.trace :as trace]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.dist :as dist]
            [replikativ.logging :as log]))

(def start-site
  "The site of a savepoint a program publishes before anything else when each
  particle must run it from its start: a program whose effects are random
  without a sample site (a model call, a tool) must not share its prefix.
  Otherwise the prefix up to the first random choice runs once and is
  forked."
  ::start)

(defn- slot-of [world] (rtp/get-state world [:inference :slot]))
(defn- weight-of [world] (or (rtp/get-state world [:inference :log-weight]) 0.0))

(defn- sample-of
  "The measure's particle for a world that returned `result`. A particle of a
  canonical world names it (`:world-id`), for its descriptor."
  [world result]
  (cond-> (m/sample-particle result (itrace/legacy-trace (rtp/get-state world [:savepoint/trace])))
    (rtp/get-state world [:inference :canonical?]) (assoc :world-id (:fork-id world))))

(defn- decide!
  "Decide `sp` under `policy` and record it in its world's trace, with
  `anchor` (a pending fork of `sp`, or nil). Returns the decision."
  ([policy sp] (decide! policy sp nil))
  ([policy sp anchor]
   (let [decision (policy sp nil)]
     (trace/record! (:savepoint/world sp) sp decision anchor)
     decision)))

;; -----------------------------------------------------------------------------
;; Resample-move: anchors and moves
;; -----------------------------------------------------------------------------
;;
;; With `:anchors`, a particle keeps an anchor (a pending fork, `spindel.trace`)
;; at the latent sites the policy names; with `:rejuvenate`, every particle
;; resampled at barrier k then makes Metropolis-Hastings moves that replay
;; from an anchor up to its k-th barrier (`trace/replay :until`), so each move
;; leaves the target — the program up to its k-th observation — invariant and
;; the weights (equal after resampling) unchanged. Siblings share their
;; ancestor's anchors, so anchors are released by reachability at barriers,
;; not by the traces that lose a move.

(defn- anchor-predicate
  "(fn [sp]) whether a particle anchors a latent site, for `:anchors`: `:all`,
  `{:lag L}` (every latent site; the last L barriers' are kept), a set of
  addresses, or a predicate of the savepoint."
  [anchors]
  (when anchors
    (let [chosen (cond
                   (or (= :all anchors) (map? anchors)) (constantly true)
                   (set? anchors) #(contains? anchors (:savepoint/address %))
                   (fn? anchors) anchors
                   :else (throw (ex-info "Unknown :anchors" {:type ::invalid-anchors
                                                             :anchors anchors})))]
      (fn [sp]
        (and (itrace/anchor? sp) (not (itrace/barrier-site? sp)) (boolean (chosen sp)))))))

(defn- window-entries
  "The entries of `trace` (with :address) a move may target when the trace
  stands at barrier `k`: every latent entry, or with `lag` those at
  barrier index ≥ k − lag, found from the end so the cost is the window's."
  [trace k lag]
  (let [by-address (:trace/entries trace)
        entry-of (fn [a] (assoc (get by-address a) :address a))]
    (if lag
      (loop [[a & more] (rseq (:trace/order trace)) acc []]
        (if-not a
          acc
          (let [e (entry-of a)
                b (:barrier (:note e))]
            (cond
              (and b (< b (- k lag))) acc
              :else (recur more (conj acc e))))))
      (mapv entry-of (:trace/order trace)))))

(defn- movable
  "Addresses a rejuvenation move may target: latent entries in the window
  that hold an anchor. The same rule on both traces of a move, so
  the reverse selection is scored alike."
  [trace k lag]
  (into [] (comp (filter itrace/latent?)
                 ;; decided by the trace alone: an anchor in a live window is
                 ;; never abandoned, and a replay from a dead one fails loudly
                 (filter :savepoint)
                 (map :address))
        (rseq (window-entries trace k lag))))

(defn- anchors-of
  "{anchor-id anchor} of the entries of `trace` in its window."
  [trace k lag]
  (into {} (keep (fn [e] (when-let [a (:savepoint e)] [(trace/anchor-id a) a])))
        (window-entries trace k lag)))

(defn- until-barrier
  "The `:until` of a replay that stops at the particle's k-th barrier."
  [k]
  (fn [s] (and (itrace/barrier-site? s)
               (= k (rtp/get-state (:savepoint/world s) [:inference :barriers])))))

(defn- particle-trace [session world]
  (assoc (rtp/get-state world [:savepoint/trace]) :trace/world world :trace/session session))

(defn- constrained-values
  "{address value} of the constrained entries of `trace` (stream data, the
  policy's constraints): what every move must keep."
  [trace]
  (into {} (keep (fn [[a e]] (when (:constrained? (:note e)) [a (:value e)])))
        (:trace/entries trace)))

(defn- all-forked
  "Fork each of `sps` (a vector) into a world; resolves the child savepoints
  in order. The forks are issued together and counted in, not chained: a
  fork may resolve inline, and a chain of a few hundred would exhaust the
  stack."
  [sps]
  (fn [resolve reject]
    (let [n (count sps)
          children (object-array n)
          remaining (atom n)
          failed? (atom false)]
      (if (zero? n)
        (resolve [])
        (dotimes [i n]
          ((sp/fork (nth sps i))
           (fn [child]
             (aset children i child)
             (when (zero? (swap! remaining dec))
               (resolve (vec children))))
           (fn [e] (when (compare-and-set! failed? false true) (reject e)))))))))

(defn- all-copied
  "Copy each distinct savepoint of `sps` (a vector) as many times as it
  occurs (`effects.savepoint/copy`, splitting its world's budget); resolves
  the copies in `sps`' order. Copying is synchronous."
  [sps]
  (fn [resolve reject]
    (try
      (let [by-world (group-by #(:fork-id (:savepoint/world %)) sps)
            copies (into {}
                         (map (fn [[id group]]
                                (let [out (volatile! nil)]
                                  ((sp/copy (first group) (count group) {:split? true})
                                   #(vreset! out [:ok %]) #(vreset! out [:error %]))
                                  (case (first @out)
                                    :ok [id (second @out)]
                                    :error (throw (second @out))
                                    (throw (ex-info "A copy did not settle synchronously"
                                                    {:type ::asynchronous-copy}))))))
                         by-world)
            taken (volatile! {})]
        (resolve (mapv (fn [sp]
                         (let [id (:fork-id (:savepoint/world sp))
                               i (get (vswap! taken update id (fnil inc 0)) id)]
                           (nth (get copies id) (dec i))))
                       sps)))
      (catch #?(:clj Throwable :cljs :default) e (reject e)))))

(defn- retained-policy
  "The policy of the retained particle: a sample site whose address `retained`
  holds takes that value, drawn — for the weight — with its own density as the
  proposal, so it contributes nothing to the weight; any other site is drawn
  from its prior."
  [retained]
  (itrace/policy
   {:draw (fn [sp _]
            (let [address (:savepoint/address sp)]
              (when (contains? retained address)
                (let [v (get retained address)]
                  {:value v :log-proposal (dist/logpdf (:dist (:savepoint/payload sp)) v)}))))}))

(defn- stream-site?
  "A sample site whose value arrives from outside (`(sample d :stream true)`),
  see `stream`."
  [sp]
  (and (= itrace/choose-site (:savepoint/site sp))
       (:stream (:options (:savepoint/payload sp)))))

(defn- run-particles
  "The particle machinery shared by `smc` and `stream`. Starts `model` with
  `n` particles and returns `{:supply! (fn [value])}`.

  Particles run until each is parked or has returned. Parked at an observe:
  when all are, the population is resampled (see `smc`) and resumed. Parked
  at a stream site: when all particles are parked there or have returned,
  `on-idle` gets the current measure; `supply!` then scores every stream site
  with the value, which turns them into an ordinary barrier. When all have
  returned, `on-done` gets the final measure; `on-error` any failure."
  [model n {:keys [resample-threshold policy executor retained ancestor-sampling? root copy?
                   anchors rejuvenate resampling genealogy? smcp3] :as opts}
   {:keys [on-idle on-done on-error]}]
  (when (and copy? anchors)
    (throw (ex-info "Anchors in copied (canonical) worlds are not supported yet"
                    {:type ::anchors-in-copies})))
  (when (and rejuvenate policy (not (contains? (meta policy) :org.replikativ.foerster.trace/options)))
    (throw (ex-info ":rejuvenate repeats the particles' policy in its moves: pass one made by foerster.trace/policy"
                    {:type ::opaque-policy})))
  (when (and rejuvenate retained)
    ;; the moves are not invariant for the conditional (particle Gibbs)
    ;; target: an exact enumeration puts the stationary law off by TV ~ 1e-3
    (throw (ex-info ":rejuvenate is not supported in conditional SMC (:retained)"
                    {:type ::rejuvenate-with-retained})))
  (when (and smcp3 (not anchors))
    (throw (ex-info ":smcp3 needs :anchors to replay from"
                    {:type ::smcp3-without-anchors})))
  (when (and smcp3 retained)
    (throw (ex-info ":smcp3 is not supported in conditional SMC (:retained)"
                    {:type ::smcp3-with-retained})))
  (when (and rejuvenate (not anchors))
    (throw (ex-info ":rejuvenate needs :anchors to replay from"
                    {:type ::rejuvenate-without-anchors})))
  (let [threshold (or resample-threshold 0.5)
        anchor? (anchor-predicate anchors)
        lag (:lag anchors)
        ;; {anchor-id anchor}: every anchor that may still be pending
        registry (atom {})
        handler-table (volatile! nil)
        policy (or policy (itrace/policy))
        policy-of (if retained
                    (let [rp (retained-policy retained)]
                      (fn [slot] (if (= 0 slot) rp policy)))
                    (constantly policy))
        root (cond
               root root
               executor (ctx/create-execution-context :executor executor)
               :else (ctx/create-execution-context))
        ;; one draw from the process generator per run: the session's seed
        ;; keys every world's streams and every barrier's draws, so runs in
        ;; parallel (IPMCMC's nodes) do not depend on how they interleave
        seed (or (:seed opts) (random/fresh-seed))
        barriers (atom 0)
        session (sp/open! root (merge {:purpose :smc
                                       :seed seed
                                       :fork-opts {:systems :none}
                                       :retain-released? false}
                                      (dissoc opts :resample-threshold :policy :executor :retained
                                              :resampling :genealogy? :smcp3
                                              :ancestor-sampling? :root :copy? :anchors :rejuvenate)))
        ;; {:parked {slot {:sp sp :value v}}  at an observe (or a supplied stream
        ;;                                    site), resumed with v after the barrier
        ;;  :streaming {slot sp}               at a stream site, waiting for a value
        ;;  :done {slot {:sample s :log-weight w}}
        ;;  :log-z accumulated :in-barrier? :finished?}
        state (atom {:parked {} :streaming {} :done {} :log-z 0.0
                     :rejuvenation {:moves 0 :accepted 0}
                     :history []})
        close! (fn []
                 ((sp/close! session)
                  (fn [_] nil)
                  (fn [e] (log/warn :smc/close-failed {:error e}))))
        finish! (fn [callback outcome]
                  (when-not (:finished? (first (swap-vals! state assoc :finished? true)))
                    (callback outcome)
                    (close!)))
        fail! #(finish! on-error %)
        measure (fn [{:keys [parked streaming done log-z rejuvenation history]}]
                  (cond-> (assoc (m/empirical
                                  (mapv (fn [slot]
                                          (if-let [d (get done slot)]
                                            [(:sample d) (:log-weight d)]
                                            (let [w (:savepoint/world (or (:sp (get parked slot))
                                                                          (get streaming slot)))]
                                              [(sample-of w nil) (weight-of w)])))
                                        (range n)))
                                 :log-normalizer log-z)
                    true (assoc :history history)
                    rejuvenate (assoc :rejuvenation rejuvenation)))]
    (letfn [(arrived! []
              ;; the last arrival claims the barrier, atomically: two particles
              ;; may arrive on two executor threads at once
              (let [[before after]
                    (swap-vals! state
                                (fn [{:keys [parked streaming done in-barrier?] :as st}]
                                  (if (and (not in-barrier?)
                                           (= n (+ (count parked) (count streaming) (count done))))
                                    (assoc st :in-barrier? true)
                                    st)))]
                (when (and (:in-barrier? after) (not (:in-barrier? before)))
                  (let [{:keys [parked streaming]} after]
                    (cond
                      (seq parked) (barrier!)
                      (seq streaming) (do (swap! state assoc :in-barrier? false)
                                          (on-idle (measure after)))
                      :else (finish! on-done (measure after)))))))

            (run-site! [sp]
              (try
                (let [slot (slot-of (:savepoint/world sp))]
                  (cond
                    (stream-site? sp)
                    (do (swap! state assoc-in [:streaming slot] sp)
                        (arrived!))

                    (and anchor? (anchor? sp))
                    ((sp/fork sp)
                     (fn [anchor]
                       (swap! registry assoc (trace/anchor-id anchor) anchor)
                       (decide-site! sp slot anchor))
                     fail!)

                    :else (decide-site! sp slot nil)))
                (catch #?(:clj Throwable :cljs :default) e (fail! e))))

            (decide-site! [sp slot anchor]
              (try
                (let [{:keys [value]} (decide! (policy-of slot) sp anchor)]
                  (if (itrace/barrier-site? sp)
                    (do (swap! state assoc-in [:parked slot] {:sp sp :value value})
                        (arrived!))
                    (sp/resume sp value)))
                (catch #?(:clj Throwable :cljs :default) e (fail! e))))

            (spawn! [first-sp]
              ;; the root's first savepoint becomes N particle worlds
              ((if copy?
                 (all-copied (vec (repeat n first-sp)))
                 (all-forked (vec (repeat n first-sp))))
               (fn [children]
                 (when-not copy? (sp/abandon first-sp))
                 (doseq [[slot child] (map-indexed vector children)]
                   (rtp/swap-state! (:savepoint/world child) [:inference :slot] (constantly slot))
                   (if (= start-site (:savepoint/site child))
                     (sp/resume child nil)
                     (run-site! child))))
               fail!))

            (barrier! []
              (let [k (swap! barriers inc)
                    go! #(random/with-stream* seed [::barrier k] (fn [] (barrier-at! k)))]
                (if smcp3 (smcp3-all! go!) (go!))))

            (smcp3-all! [continue!]
              ;; every parked particle takes its SMCP3 step, then the barrier
              ;; resamples on the updated weights
              (let [parked (:parked @state)
                    slots (vec (keys parked))
                    results (atom parked)
                    remaining (atom (count slots))]
                (if (empty? slots)
                  (continue!)
                  (doseq [slot slots]
                    (smcp3-particle! slot (get parked slot)
                                     (fn [entry]
                                       (swap! results assoc slot entry)
                                       (when (zero? (swap! remaining dec))
                                         (swap! state assoc :parked @results)
                                         (continue!))))))))

            (smcp3-particle! [slot {:keys [sp value] :as entry} done]
              (try
                (let [world0 (:savepoint/world sp)
                      k (rtp/get-state world0 [:inference :barriers])
                      w0 (weight-of world0)
                      t0 (assoc (particle-trace session world0) :trace/pending sp :trace/pending-value value)
                      {:keys [updates log-q reverse log-jacobian]}
                      (random/in-world-stream world0 [::smcp3 k]
                                              #((:forward smcp3) (itrace/choices t0) t0))]
                  (if (empty? updates)
                    ;; staying is a move too: its densities still weigh
                    (do (rtp/swap-state! world0 [:inference :log-weight]
                                         (constantly (+ w0 ((:backward smcp3) (itrace/choices t0) reverse)
                                                        (- (or log-q 0.0)) (or log-jacobian 0.0))))
                        (done entry))
                    (let [from (trace/earliest t0 (keys updates))
                          _ (when-let [fixed (seq (filter #(let [n (:note (get-in t0 [:trace/entries %]))]
                                                             (or (:observed? n) (:constrained? n)))
                                                          (keys updates)))]
                              (throw (ex-info "An SMCP3 update changes an observed or constrained site"
                                              {:type ::fixed-site :addresses (vec fixed)})))
                          _ (when-not (get-in t0 [:trace/entries from :savepoint])
                              (throw (ex-info "An SMCP3 update changes a site without an anchor"
                                              {:type ::no-anchor :address from})))
                          base (itrace/policy-options (policy-of slot))
                          move (itrace/policy
                                (assoc base
                                       :keep? true
                                       :constraints (merge (:constraints base) (constrained-values t0))
                                       :draw (fn [s _]
                                               (let [a (:savepoint/address s)]
                                                 (when (contains? updates a)
                                                   ;; K's value; its density is
                                                   ;; in K's log-q, not here
                                                   {:value (get updates a) :symmetric? true})))))]
                      ((trace/replay t0 from move {:anchor? anchor?
                                                   :until (until-barrier k)
                                                   :seed (sp/derive-seed (sp/seed world0) ::smcp3 k)})
                       (fn [t1]
                         (try
                           (if (or (:trace/error t1) (not (:trace/pending t1)))
                             ;; K proposed a state the program cannot reach:
                             ;; the particle stays, with no weight
                             (do (trace/release! t1 t0)
                                 (rtp/swap-state! world0 [:inference :log-weight] (constantly ##-Inf))
                                 (done entry))
                             (let [increment (+ (itrace/mh-log-ratio t0 t1 (constantly 0.0) from)
                                                ((:backward smcp3) (itrace/choices t1) reverse)
                                                (- log-q)
                                                (or log-jacobian 0.0))
                                   w1 (:trace/world t1)]
                               (rtp/swap-state! w1 [:inference :slot] (constantly slot))
                               (rtp/swap-state! w1 [:inference :log-weight] (constantly (+ w0 increment)))
                               (rtp/swap-state! w1 [:engine/reuse-source] (constantly nil))
                               (sp/install-handlers! w1 @handler-table)
                               (swap! registry merge (anchors-of t1 k lag))
                               ;; the particle continues in t1's world
                               (when (sp/pending? sp) (sp/abandon sp))
                               (sp/release-world! session world0)
                               (done {:sp (:trace/pending t1) :value (:trace/pending-value t1)})))
                           (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                       fail!))))
                (catch #?(:clj Throwable :cljs :default) e (fail! e))))

            (barrier-at! [k]
              (if (and retained ancestor-sampling? (contains? (:parked @state) 0))
                ;; PGAS: the retained particle redraws the past it continues
                ;; from, ∝ w_i · p(retained future | particle i's past)
                (let [parked (:parked @state)
                      candidates (vec (sort (keys parked)))]
                  ((score-futures (mapv #(get parked %) candidates))
                   (fn [scores]
                     (let [combined (mapv (fn [slot score]
                                            (+ (weight-of (:savepoint/world (:sp (get parked slot)))) score))
                                          candidates scores)]
                       (resample-with! k (nth candidates
                                              (random/with-stream* seed [::ancestor k]
                                                #(m/sample-categorical (m/normalize-log-weights combined)))))))
                   fail!))
                (resample-with! k 0)))

            (score-futures [entries]
              ;; each parked particle's future replayed on the retained values
              ;; in a fork under a handler table of its own; its final weight
              ;; (from 0) is log p(retained latents, observations | its past)
              (fn [resolve reject]
                (let [k (count entries)
                      scores (object-array k)
                      remaining (atom k)
                      failed? (atom false)
                      scoring (itrace/policy {:constraints retained})
                      done! (fn [i score]
                              (aset scores i score)
                              (when (zero? (swap! remaining dec))
                                (resolve (vec scores))))]
                  (dotimes [i k]
                    (let [{:keys [sp value]} (nth entries i)]
                      ((sp/fork sp {:handlers
                                    {sp/any-site (fn [s] (sp/resume s (:value (decide! scoring s))))
                                     sp/result-site (fn [{w :savepoint/world}]
                                                      (let [score (weight-of w)]
                                                        (sp/release-world! session w)
                                                        (done! i score)))
                                     sp/error-site (fn [{w :savepoint/world}]
                                                     (sp/release-world! session w)
                                                     (done! i ##-Inf))
                                     sp/abandoned-site (fn [_] nil)}})
                       (fn [child]
                         (rtp/swap-state! (:savepoint/world child) [:inference :log-weight] (constantly 0.0))
                         (sp/resume child value))
                       (fn [e] (when (compare-and-set! failed? false true) (reject e)))))))))

            (resample-with! [k retained-ancestor]
              (let [{:keys [parked streaming done]} @state
                    slots (vec (range n))
                    world-of (fn [slot] (:savepoint/world (or (:sp (get parked slot))
                                                              (get streaming slot))))
                    log-ws (mapv #(if-let [d (get done %)] (:log-weight d) (weight-of (world-of %)))
                                 slots)
                    weights (m/normalize-log-weights log-ws)
                    ess (m/compute-ess weights)
                    resample? (or (some? retained)
                                  (< ess (* threshold n)))]
                (swap! state update :history conj
                       {:ess ess :resampled? resample?
                        ;; the barrier's factor of the evidence estimate
                        :log-mean-weight (when resample? (m/log-mean-exp log-ws))})
                (if-not resample?
                  (do (when lag (gc! parked streaming))
                      (swap! state assoc :parked {} :in-barrier? false)
                      (doseq [[_ {:keys [sp value]}] (sort-by key parked)]
                        (sp/resume sp value)))
                  (let [ancestors (random/with-stream*
                                    seed [::resample k]
                                    #(if retained
                                      ;; conditional: slot 0 continues from its
                                      ;; own lineage, or the past PGAS drew for it
                                       (into [retained-ancestor]
                                             (repeatedly (dec n) (fn [] (m/sample-categorical weights))))
                                       (m/resample resampling weights n)))
                        live? #(or (contains? parked %) (contains? streaming %))
                        forked-slots (filterv #(live? (nth ancestors %)) slots)
                        source (fn [a] (or (:sp (get parked a)) (get streaming a)))]
                    (swap! state update :log-z + (m/log-mean-exp log-ws))
                    (when genealogy?
                      (swap! state update :history
                             (fn [h] (update h (dec (count h)) assoc :ancestors ancestors))))
                    ((if copy?
                       (all-copied (mapv #(source (nth ancestors %)) forked-slots))
                       (all-forked (mapv #(source (nth ancestors %)) forked-slots)))
                     (fn [children]
                       (let [carried (into {} (keep (fn [slot]
                                                      (when-let [d (get done (nth ancestors slot))]
                                                        [slot (assoc d :log-weight 0.0)])))
                                           slots)
                             placed (map (fn [slot child]
                                           (let [a (nth ancestors slot)
                                                 w (:savepoint/world child)]
                                             (rtp/swap-state! w [:inference :slot] (constantly slot))
                                             (rtp/swap-state! w [:inference :log-weight] (constantly 0.0))
                                             [slot child (get parked a)]))
                                         forked-slots children)
                             parked' (into {} (keep (fn [[slot child p]]
                                                      (when p [slot {:sp child :value (:value p)}])))
                                           placed)
                             streaming' (into {} (keep (fn [[slot child p]] (when-not p [slot child])))
                                              placed)]
                         ;; a copied source continues in its copies; the
                         ;; others end here
                         (doseq [sp (concat (map :sp (vals parked)) (vals streaming))
                                 :when (sp/pending? sp)]
                           (sp/abandon sp))
                         (let [continue!
                               (fn [parked']
                                 (when anchor? (gc! parked' streaming'))
                                 (swap! state assoc :parked {} :streaming streaming' :done carried
                                        :in-barrier? false)
                                 (if (empty? parked')
                                   (arrived!)
                                   (doseq [[_ {:keys [sp value]}] (sort-by key parked')]
                                     (sp/resume sp value))))]
                           (if (and rejuvenate (seq parked'))
                             (rejuvenate! parked' continue!)
                             (continue! parked')))))
                     fail!)))))

            (gc! [parked streaming]
              ;; anchors no live particle's window reaches are given back
              (let [live (reduce (fn [acc sp]
                                   (let [w (:savepoint/world sp)]
                                     (merge acc (anchors-of (particle-trace session w)
                                                            (or (rtp/get-state w [:inference :barriers]) 0)
                                                            lag))))
                                 {}
                                 (concat (map :sp (vals parked)) (vals streaming)))]
                (doseq [[id a] @registry
                        :when (and (not (contains? live id)) (sp/pending? a))]
                  (sp/abandon a))
                (reset! registry live)
                (swap! state update-in [:rejuvenation :max-anchors] (fnil max 0) (count live))))

            (rejuvenate! [parked done!]
              ;; every resampled particle moves, the retained one excepted
              (let [slots (vec (remove #(and retained (= 0 %)) (keys parked)))
                    results (atom parked)
                    remaining (atom (count slots))]
                (if (empty? slots)
                  (done! parked)
                  (doseq [slot slots]
                    (move-particle! slot (get parked slot)
                                    (fn [entry]
                                      (swap! results assoc slot entry)
                                      (when (zero? (swap! remaining dec))
                                        (done! @results))))))))

            (move-particle! [slot {:keys [sp value]} done]
              (try
                (let [world0 (:savepoint/world sp)
                      k (rtp/get-state world0 [:inference :barriers])
                      w0 (weight-of world0)
                      t0 (assoc (particle-trace session world0) :trace/pending sp :trace/pending-value value)
                      count-movable #(count (movable % k lag))
                      opts {:select (fn [t _]
                                      (let [ms (movable t k lag)]
                                        (if (empty? ms)
                                          {:targets #{} :log-selection (constantly 0.0)}
                                          {:targets #{(m/pick-uniformly ms)}
                                           :log-selection #(- (Math/log (double (count-movable %))))})))
                            :propose (or (:propose rejuvenate) itrace/prior-proposal)
                            :policy-options (itrace/policy-options (policy-of slot))
                            :constraints (constrained-values
                                          (if lag
                                            {:trace/entries (into {} (map (juxt :address identity))
                                                                  (window-entries t0 k lag))}
                                            t0))
                            ;; stop at this particle's own k-th barrier
                            :until (until-barrier k)
                            :anchor? anchor?
                            :shared-anchors? true}
                      steps (:moves rejuvenate 1)
                      finish (fn [t]
                               (if (= (:fork-id (:trace/world t)) (:fork-id world0))
                                 (done {:sp sp :value value})
                                 (let [w (:trace/world t)]
                                   ;; the moved particle continues in the
                                   ;; replay's world: it is the slot's now, with
                                   ;; the slot's weight
                                   (rtp/swap-state! w [:inference :slot] (constantly slot))
                                   (rtp/swap-state! w [:inference :log-weight] (constantly w0))
                                   ;; and does not keep the displaced world
                                   ;; alive as the source of reused spins
                                   (rtp/swap-state! w [:engine/reuse-source] (constantly nil))
                                   (sp/install-handlers! w @handler-table)
                                   (done {:sp (:trace/pending t) :value (:trace/pending-value t)}))))]
                  (letfn [(step [j t]
                            (if (= j steps)
                              (finish t)
                              ((itrace/mh-step t (assoc opts :iteration j))
                               (fn [{t' :trace accepted? :accepted?}]
                                 (swap! state update :rejuvenation
                                        #(-> % (update :moves inc)
                                             (update :accepted + (if accepted? 1 0))))
                                 (when accepted?
                                   (swap! registry merge (anchors-of t' k lag)))
                                 (step (inc j) t'))
                               fail!)))]
                    (step 0 t0)))
                (catch #?(:clj Throwable :cljs :default) e (fail! e))))

            (supply! [value]
              ;; every particle waiting at a stream site scores `value` there;
              ;; that is an ordinary barrier from here on
              (try
                (let [{:keys [streaming]} @state
                      parked (into {} (map (fn [[slot sp]]
                                             (let [policy (itrace/policy
                                                           {:constraints {(:savepoint/address sp) value}})]
                                               (decide! policy sp)
                                               [slot {:sp sp :value value}])))
                                   streaming)]
                  (swap! state assoc :streaming {} :parked parked)
                  (arrived!))
                (catch #?(:clj Throwable :cljs :default) e (fail! e))))]
      (try
        (vreset! handler-table
                 {sp/any-site
                  (fn [s]
                    (if (nil? (slot-of (:savepoint/world s)))
                      (spawn! s)
                      (run-site! s)))
                  sp/result-site
                  (fn [{world :savepoint/world result :savepoint/payload}]
                    (if-let [slot (slot-of world)]
                      (do (swap! state assoc-in [:done slot] {:sample (sample-of world result)
                                                              :log-weight (weight-of world)})
                  ;; the Sample holds what the measure needs; give the world
                  ;; back now instead of holding every finished particle until
                  ;; the session closes
                          (sp/release-world! session world)
                          (arrived!))
              ;; a program without savepoints: one deterministic particle
                      (finish! on-done (m/empirical (vec (repeat n [(sample-of world result) 0.0]))))))
                  sp/error-site
                  (fn [{error :savepoint/payload}] (fail! error))
                  sp/abandoned-site (fn [_] nil)})
        (sp/install-handlers! root @handler-table)
        (sp/start! session model)
        (catch #?(:clj Throwable :cljs :default) e
          (log/error :smc/start-failed {:error e})
          (fail! e)))
      {:supply! supply! :close! close!})))

(defn smc
  "Run `model` (a spin) with `n` particles. Options: `:resample-threshold`
  (ESS fraction, default 0.5), `:policy` (an `foerster.trace/policy`,
  default the prior with no options), `:executor` for the root world,
  `:root` a world to run in instead of a fresh one, `:copy? true` to make
  particles by copying worlds (`effects.savepoint/copy`: a world holding a
  system that may not be copied is refused, each copy gets an even share of
  its world's budget; the root must be a world of the session's `:scope`),
  and session options (`effects.savepoint/open!`).

  `:retained` {address value} makes it CONDITIONAL SMC (particle Gibbs):
  particle 0 follows those choices, every barrier resamples, and particle 0
  keeps its own lineage while the other n−1 draw their ancestors. With
  `:ancestor-sampling? true` particle 0 instead redraws its ancestor at every
  barrier, ∝ w_i · p(retained future | particle i's past), the future scored
  by replaying each particle in a fork on the retained values (PGAS).

  Resample-move: `:anchors` keeps an anchor at latent sites — `:all`,
  `{:lag L}` (those of the last L barriers), a set of addresses, or a
  predicate of the savepoint — and `:rejuvenate {:moves m :propose p}` makes
  every particle take m Metropolis-Hastings moves after each resampling,
  each replaying from an anchor up to the particle's current barrier (`p` a
  proposal as for `foerster.trace/mh-step`, default the prior). The moves
  leave the target up to that barrier invariant, so the weights and the
  evidence estimate are unchanged. Not in copied worlds or with
  `:retained`. The measure's `:rejuvenation` counts `:moves`, `:accepted`
  and `:max-anchors`, the most anchors alive at a barrier.

  SMCP3 (Lew et al. 2023): `:smcp3 {:forward K :backward L}` gives every
  particle parked at a barrier a move-reweight step before the resampling
  decision. `(K choices trace)` — the particle's `foerster.trace/choices` and
  its partial trace — returns `{:updates {address value} :log-q lq :reverse
  u' :log-jacobian lj}`: new values for some latent sites (earlier ones
  included, so K may revise the past), the log density of K's own random
  choices, the auxiliary value the backward kernel would draw to go back, and
  an optional log |Jacobian| for a deterministic continuous map. `(L choices'
  u')` is the log density of that reverse draw given the new choices. The
  particle is replayed from the earliest updated site (it needs an anchor
  there, `:anchors`) to its barrier, and its weight multiplied by
  p(x')·L(u'|x') / (p(x)·K(u|x)) · |J|. K draws from `foerster.random` (it
  runs in the particle's stream). Sites K does not update are kept, and those
  the replay reaches afresh are drawn from their prior, as in a move.

  `:resampling` is `:systematic` (default), `:stratified`, `:residual` or
  `:multinomial` (`measure/resample`). The measure's `:history` has a map per
  barrier: `:ess` before resampling, `:resampled?`, `:log-mean-weight` (the
  barrier's factor of the evidence) and, with `:genealogy? true`, the
  `:ancestors` each slot was resampled from.

  Resolves an `EmpiricalMeasure` of `Sample`s (result + trace) whose
  `log-marginal` is the SMC evidence estimate. A model with stream sites
  runs with `stream` instead."
  [model n & [opts]]
  (fn [resolve reject]
    (let [[resolve reject] (sp/in-callers-world resolve reject)]
      (run-particles model n opts
                     {:on-done resolve
                      :on-error reject
                      :on-idle (fn [_]
                                 (reject (ex-info "The model has stream sites; run it with smc/stream"
                                                  {:type ::stream-sites})))}))))

(defn stream
  "Online SMC: `model` marks the sites whose values arrive from outside as
  stream sites — `(sample (normal x 1) :id [:y t] :stream true)` — and
  particles run until each waits at its next one. Resolves a step

    {:measure  the posterior over the trajectories so far
     :push     (fn [y]) -> CPS resolving the next step, after every particle
               scored y at its stream site, the population was resampled as
               needed, and each ran on to its next stream site (or returned)
     :done?    true once every particle has returned (no :push then)
     :close    (fn []) giving the worlds back}

  Each push costs the particles' work up to their next stream site; nothing
  already seen is re-run. `opts` as for `smc`."
  [model n & [opts]]
  (fn [resolve reject]
    (let [waiting (atom (sp/in-callers-world resolve reject))
          controller (atom nil)
          step (fn [done? m]
                 (let [[res _] @waiting]
                   (res (cond-> {:measure m :done? done?
                                 :close (fn [] ((:close! @controller)))}
                          (not done?)
                          (assoc :push (fn [y]
                                         (fn [res' rej']
                                           (reset! waiting (sp/in-callers-world res' rej'))
                                           ((:supply! @controller) y))))))))]
      (reset! controller
              (run-particles model n opts
                             {:on-idle #(step false %)
                              :on-done #(step true %)
                              :on-error (fn [e] ((second @waiting) e))})))))

;; =============================================================================
;; Particle MCMC on savepoint SMC
;; =============================================================================

(defn- normalized
  "A measure's particles with their weights normalized to sum to one."
  [measure]
  (let [ps (m/get-particles measure)
        lse (m/log-sum-exp (mapv second ps))]
    (mapv (fn [[s lw]] [s (- lw lse)]) ps)))

(defn retained-choices
  "{address value} of the unobserved sample sites of `trace` (a Sample's):
  the `:retained` of a conditional sweep that keeps that trajectory."
  [trace]
  (into {} (keep (fn [[a e]] (when-not (or (:observed? e) (:deterministic? e)) [a (:value e)]))) trace))

(defn- choices-of [sample] (retained-choices (m/get-trace sample)))

(defn- sweeps
  "Run `step` — (fn [state]) -> CPS resolving [state' samples] — `k` times,
  pooling the samples. Resolves an EmpiricalMeasure."
  [k init step]
  (fn [resolve reject]
    (letfn [(go [i state acc]
                (if (= i k)
                  (resolve (m/empirical acc))
                  ((step state)
                   (fn [[state' samples]] (go (inc i) state' (into acc samples)))
                   reject)))]
      (go 0 init []))))

(defn pgibbs
  "Particle Gibbs (Andrieu et al. 2010) as iterated conditional SMC: each
  sweep keeps the trajectory drawn from the previous one, and every sweep's
  particles are pooled, normalized per sweep. `opts` as for `smc`."
  [model n iterations & [opts]]
  (fn [resolve reject]
    ((smc model n opts)
     (fn [initial]
       (let [pick (fn [measure]
                    (let [ps (m/get-particles measure)]
                      (choices-of (first (nth ps (m/sample-categorical
                                                  (m/normalize-log-weights (mapv second ps))))))))]
         ((sweeps iterations (pick initial)
                  (fn [retained]
                    (fn [res rej]
                      ((smc model n (assoc opts :retained retained))
                       (fn [sweep] (res [(pick sweep) (normalized sweep)]))
                       rej))))
          resolve reject)))
     reject)))

(defn pgas
  "Particle Gibbs with ancestor sampling (Lindsten et al. 2014): `pgibbs`
  whose retained particle redraws its ancestor at every barrier. Improves
  mixing where plain particle Gibbs degenerates; costs a replay of the
  retained future per parked particle per barrier."
  [model n iterations & [opts]]
  (pgibbs model n iterations (assoc opts :ancestor-sampling? true)))

(defn pimh
  "Particle independent Metropolis-Hastings: each iteration proposes a fresh
  SMC sweep and accepts it on the ratio of evidence estimates; the current
  sweep's particles, normalized, are emitted every iteration. `opts` as for
  `smc`."
  [model n iterations & [opts]]
  (fn [resolve reject]
    ;; each iteration's accept draws from its own stream of the chain's seed
    (let [seed (random/fresh-seed)
          iteration (volatile! 0)]
      ((smc model n opts)
       (fn [initial]
         ((sweeps iterations [(normalized initial) (m/log-marginal initial)]
                  (fn [[current log-z]]
                    (fn [res rej]
                      ((smc model n opts)
                       (fn [proposed]
                         (let [log-z' (m/log-marginal proposed)
                               ratio (- log-z' log-z)
                               u (random/with-stream* seed [::accept (vswap! iteration inc)]
                                   m/uniform01)
                               accept? (or (>= ratio 0.0) (< (Math/log u) ratio))
                               state' (if accept? [(normalized proposed) log-z'] [current log-z])]
                           (res [state' (first state')])))
                       rej))))
          resolve reject))
       reject))))
