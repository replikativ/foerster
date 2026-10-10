(ns org.replikativ.foerster.tempering
  "Tempered SMC: an SMC sampler (Del Moral, Doucet & Jasra 2006) over whole
  program traces, from the prior to the posterior through the targets

    π_β(x) ∝ p(x) · L(x)^β,   0 = β_0 < β_1 < … < β_T = 1,

  L the product of the program's observations and factors. Every particle is
  a complete run of the program (`foerster.gfi`), recorded at temperature 0
  with each observation's untempered log-likelihood in its note. A step picks
  the next β adaptively, so that the conditional ESS of the incremental
  weights L^(β'−β) is `:ess-target`·N (Zhou, Johansen & Aston 2016), reweights,
  resamples and moves every particle by Metropolis-Hastings targeting π_β'
  (`foerster.trace/mh-step` at that temperature; random-walk proposals scaled
  by the population's spread on continuous sites, prior proposals on the
  others). The evidence estimate is the product of the steps' mean
  incremental weights.

  With `:waste-free P` (Dau & Chopin 2022) a step resamples N/P particles
  and keeps every state of their P-step chains as particles.

  For data that arrive one observation at a time (IBIS), use SMC with
  resample-move instead (`foerster.smc`, `:anchors` on the static
  parameters)."
  (:require [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.block-mh :as block-mh]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.executor :as ex]
            [org.replikativ.spindel.engine.protocols :as rtp]
            [org.replikativ.spindel.trace :as trace]))

(defn- log-likelihood
  "Σ untempered log-likelihood of `t`'s observations and factors."
  [t]
  (transduce (keep (comp :log-lik :note)) + 0.0 (itrace/entries t)))

(defn- log-sum-exp [xs]
  (let [top (reduce max ##-Inf xs)]
    (if (= ##-Inf top)
      ##-Inf
      (+ top (Math/log (reduce + (map #(Math/exp (- % top)) xs)))))))

(defn- relative-cess
  "The conditional ESS of the incremental weights exp(δ·ℓ) under the
  normalized log weights `lw`, as a fraction of N."
  [lw ls delta]
  (let [a (mapv (fn [w l] (+ w (* delta l))) lw ls)
        b (mapv (fn [w l] (+ w (* 2.0 delta l))) lw ls)
        r (- (* 2.0 (log-sum-exp a)) (log-sum-exp b))]
    (if (#?(:clj Double/isNaN :cljs js/isNaN) r) 0.0 (Math/exp r))))

(defn- next-delta
  "The temperature increment whose conditional ESS is `target`: the rest of
  the way to 1 when that keeps it, else by bisection."
  [lw ls beta target]
  (let [rest (- 1.0 beta)]
    (if (>= (relative-cess lw ls rest) target)
      rest
      (loop [lo 0.0 hi rest i 0]
        (let [mid (* 0.5 (+ lo hi))]
          (if (or (= i 60) (< (- hi lo) 1e-12))
            (max mid 1e-12)
            (if (>= (relative-cess lw ls mid) target)
              (recur mid hi (inc i))
              (recur lo mid (inc i)))))))))

(defn- population-scales
  "{address sd} of the continuous latent sites of `traces`, over the
  particles that reach them."
  [traces]
  (let [values (reduce (fn [acc t]
                         (reduce (fn [acc {:keys [address value note] :as e}]
                                   (if (and (itrace/latent? e) (number? value)
                                            (dist/continuous? (:dist note)))
                                     (update acc address (fnil conj []) (double value))
                                     acc))
                                 acc (itrace/entries t)))
                       {} traces)]
    (into {} (keep (fn [[a xs]]
                     (when (> (count xs) 1)
                       (let [mu (/ (reduce + xs) (count xs))
                             var (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) (count xs))]
                         (when (pos? var) [a (Math/sqrt var)])))))
          values)))

(defn- adaptive-proposal
  "A random walk of `scale`·sd on continuous sites the population spreads
  over, the prior elsewhere. Chosen from the population before the step's
  moves, so each move is an ordinary MH kernel."
  [sds scale]
  (fn [sp old-entry]
    (let [sd (get sds (:savepoint/address sp))
          v (:value old-entry)]
      (if (and sd (number? v) (dist/continuous? (:dist (:savepoint/payload sp))))
        {:value (+ v (* scale sd (dist/draw (dist/normal 0.0 1.0)))) :symmetric? true}
        (itrace/prior-proposal sp old-entry)))))

(defn- all-settled
  "Run `(start i resolve reject)` for i < n together, spread over `executor`'s
  threads (`executor/spread!`; on this thread when nil); resolves the results
  in order, or rejects with the first failure."
  [executor n start]
  (fn [resolve reject]
    (if (zero? n)
      (resolve [])
      (let [out (object-array n)
            remaining (atom n)
            failed? (atom false)
            launch (fn [i]
                     (start i
                            (fn [v]
                              (aset out i v)
                              (when (zero? (swap! remaining dec))
                                (resolve (vec out))))
                            (fn [e]
                              (when (compare-and-set! failed? false true)
                                (reject e)))))]
        (if executor
          (ex/spread! executor n launch)
          (dotimes [i n] (launch i)))))))

(defn- executor-of [trace] (:executor (:trace/world trace)))

(defn- anchor-ids [traces]
  (into #{} (comp (mapcat (comp vals :trace/entries))
                  (keep :savepoint)
                  (map trace/anchor-id))
        traces))

(defn- release-dropped!
  "Give back what `dropped` traces hold that no `live` trace refers to."
  [dropped live]
  (let [live-anchors (anchor-ids live)
        live-worlds (into #{} (map (comp :fork-id :trace/world)) live)]
    (doseq [t (vals (into {} (map (juxt (comp :fork-id :trace/world) identity)) dropped))]
      (doseq [a (keep :savepoint (vals (:trace/entries t)))
              :when (and (not (contains? live-anchors (trace/anchor-id a)))
                         (sp/pending? a))]
        (try (sp/abandon a) (catch #?(:clj Throwable :cljs :default) _ nil)))
      (let [w (:trace/world t)]
        (when-not (contains? live-worlds (:fork-id w))
          (sp/release-world! (:trace/session t) w))))))

(defn- block-scales
  "{address [step …]} for the block sites of `traces`: each coordinate's
  population standard deviation times `scale`/√dimension, a random walk sized
  to the spread of the particles (as `adaptive-proposal` for single sites)."
  [traces scale]
  (let [values (reduce (fn [acc t]
                         (reduce (fn [acc address]
                                   (if (block-mh/block-site? t address)
                                     (update acc address (fnil conj [])
                                             (mapv double (get-in t [:trace/entries address :value])))
                                     acc))
                                 acc (itrace/latent-addresses t)))
                       {} traces)]
    (into {} (map (fn [[address qs]]
                    (let [n (count qs) dim (count (first qs))
                          mean (apply mapv (fn [& xs] (/ (reduce + xs) n)) qs)
                          sd (apply mapv (fn [mu & xs]
                                           (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs))
                                                         (max 1 (dec n)))))
                                    mean qs)]
                      [address (mapv #(* (/ scale (Math/sqrt dim)) (max % 1e-9)) sd)])))
          values)))

(defn- move-chain
  "`steps` moves of `t` at temperature `beta`; resolves the state after every
  step, every trace the chain passed through (`:seen`, to be released) and
  the acceptance counts. A step moves each block site by a block move
  (`foerster.block-mh`, `bscales` its random walk), then makes `moves`
  single-site MH moves of the other latent sites (default: one sweep)."
  [t steps moves beta propose bscales key]
  (fn [resolve reject]
    (letfn [(others [t] (vec (remove #(block-mh/block-site? t %) (itrace/latent-addresses t))))
            (blocks [t] (filterv #(block-mh/block-site? t %) (itrace/latent-addresses t)))
            (single-site [t step j n states seen accepted total]
              (if (= j n)
                (go t (inc step) (conj states t) seen accepted total)
                ((itrace/mh-step t (cond-> {:temperature beta :propose propose
                                            :iteration [key step j] :keep-old? true}
                                     (seq (blocks t))
                                     (assoc :select (fn [t' _]
                                                      {:targets #{(m/pick-uniformly (others t'))}
                                                       :log-selection #(- (Math/log (double (count (others %)))))}))))
                 (fn [{t' :trace accepted? :accepted?}]
                   ;; the accepted world need not keep the one it replayed
                   ;; from alive as a source of reused spins
                   (when accepted?
                     (rtp/swap-state! (:trace/world t') [:engine/reuse-source] (constantly nil)))
                   (single-site t' step (inc j) n states (if accepted? (conj seen t') seen)
                                (if accepted? (inc accepted) accepted) (inc total)))
                 reject)))
            (block-moves [t step [address & more] states seen accepted total]
              (if-not address
                (let [n (if (seq (others t)) (or moves (count (others t))) 0)]
                  (single-site t step 0 n states seen accepted total))
                ((block-mh/move t address {:beta beta :scales (get bscales address)
                                           :steps (count (get bscales address))
                                           :key [key step] :keep-old? true})
                 (fn [{t' :trace n-accepted :accepted}]
                   (let [moved? (pos? n-accepted)]
                     (when moved?
                       (rtp/swap-state! (:trace/world t') [:engine/reuse-source] (constantly nil)))
                     (block-moves t' step more states (if moved? (conj seen t') seen)
                                  (if moved? (inc accepted) accepted) (inc total))))
                 reject)))
            (go [t step states seen accepted total]
                (if (= step steps)
                  (resolve {:states states :seen seen :accepted accepted :moves total})
                  (block-moves t step (blocks t) states seen accepted total)))]
      (go t 0 [] [] 0 0))))

;; -----------------------------------------------------------------------------
;; A model that is one block: particles are θ vectors
;; -----------------------------------------------------------------------------

(defn- single-block
  "{:address :dist} when every trace in `ts` is one latent block site and
  nothing else (no other choice, observation or factor), all under one law:
  then a particle is its θ, and tempering needs no world until the end."
  [ts]
  (let [shape (fn [t]
                (let [es (itrace/entries t)]
                  (when (and (= 1 (count es)) (itrace/latent? (first es))
                             (block-mh/block-site? t (:address (first es))))
                    {:address (:address (first es)) :dist (:dist (:note (first es)))})))
        s (shape (first ts))]
    (when (and s (every? #(= s (shape %)) (rest ts)))
      s)))

(defn- theta-scales
  "The block random walk of `block-scales`, over θ vectors."
  [qs scale]
  (let [n (count qs) dim (count (first qs))
        mean (apply mapv (fn [& xs] (/ (reduce + xs) n)) qs)
        sd (apply mapv (fn [mu & xs]
                         (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs))
                                       (max 1 (dec n)))))
                  mean qs)]
    (mapv #(* (/ scale (Math/sqrt dim)) (max % 1e-9)) sd)))

(defn- theta-tempered
  "Tempered SMC of a one-block model on θ: weights from the block's rest
  log t − log q, resampling and population-scaled random walks at each
  temperature on the vectors, the same seeded streams as the world path.
  The final particles are written back to traces by one replay each from
  `template`'s anchor (equal θ once). Resolves what `finish` takes."
  [template {:keys [address dist]} qs {:keys [ess-target scale waste-free max-steps n seed stats]}]
  (let [rest-of (fn [q] (let [lq (dist/-draw-logpdf dist q)] (- (dist/logpdf dist q) lq)))
        steps-per-chain (if waste-free (dec waste-free) 1)
        chains (if waste-free (quot n waste-free) n)
        dim (count (first qs))]
    (loop [qs qs ls (mapv rest-of qs) lw (vec (repeat n (- (Math/log n)))) beta 0.0 log-z 0.0
           temperatures [0.0] k 0]
      (cond
        (>= beta 1.0) {:qs qs :log-z log-z :temperatures temperatures}
        (>= k max-steps) (throw (ex-info "Tempering did not reach β = 1" {:type ::too-many-steps :beta beta}))
        :else
        (let [delta (next-delta lw ls beta ess-target)
              beta' (min 1.0 (+ beta delta))
              delta (- beta' beta)
              inc-lw (mapv (fn [w l] (if (zero? delta) w (+ w (* delta l)))) lw ls)
              log-z' (+ log-z (log-sum-exp inc-lw))
              weights (m/normalize-log-weights inc-lw)
              ancestors (random/with-stream* seed [::resample k]
                          #(m/systematic-resample weights chains))
              starts (mapv #(nth qs %) ancestors)
              scales (theta-scales starts scale)
              f (block-mh/density dist beta')
              moved (vec (map-indexed
                          (fn [i q0]
                            (random/with-stream* seed [::move k i]
                              #(loop [s 0 q q0 states []]
                                 (if (= s steps-per-chain)
                                   states
                                   (let [{q' :q a :accepted} (block-mh/walk f q scales dim)]
                                     (swap! stats (fn [st] (-> st (update :moves + dim) (update :accepted + a))))
                                     (recur (inc s) q' (conj states q')))))))
                          starts))
              qs' (vec (mapcat (fn [start states] (if waste-free (into [start] states) states))
                               starts moved))]
          (recur qs' (mapv rest-of qs') (vec (repeat (count qs') (- (Math/log (count qs')))))
                 beta' log-z' (conj temperatures beta') (inc k)))))))

(defn- write-back-all
  "Traces of `template` with the block site at `address` set to each of `qs`
  (one replay per distinct θ); resolves them in order."
  [template address qs]
  (fn [resolve reject]
    (let [distinct-qs (vec (distinct qs))
          policy (fn [q] (itrace/policy {:constraints {address q}}))]
      ((all-settled (executor-of template) (count distinct-qs)
                    (fn [i res rej]
                      ((trace/replay template address (policy (nth distinct-qs i)) {:anchor? itrace/anchor?})
                       res rej)))
       (fn [ts] (let [by-q (zipmap distinct-qs ts)] (resolve (mapv by-q qs))))
       reject))))

(defn tempered
  "Tempered SMC of `model` (a spin) with `n` particles. Options:

    :ess-target  the conditional ESS fraction each step keeps (default 0.5):
                 higher, more and smaller temperature steps
    :moves       single-site MH moves per particle per step (default: one
                 sweep, as many as the particle has latent sites)
    :scale       random-walk scale, in population standard deviations
                 (default 2.38)
    :waste-free  P: resample n/P particles and keep every state of their
                 P-step chains (n must be a multiple of P)
    :max-steps   refuse to run longer (default 1000)
    :executor    the particle worlds' executor

  Returns a CPS operation resolving an EmpiricalMeasure of `Sample`s whose
  `m/log-marginal` estimates the evidence; `:temperatures` holds the
  schedule and `:rejuvenation` the moves made and accepted."
  [model n & [{:keys [ess-target moves scale waste-free max-steps executor]
               :or {ess-target 0.5 scale 2.38 max-steps 1000}}]]
  (when-not (and (number? ess-target) (< 0.0 ess-target 1.0))
    (throw (ex-info ":ess-target must lie in (0, 1)" {:type ::invalid-ess-target :ess-target ess-target})))
  (when-not (or (nil? moves) (nat-int? moves))
    (throw (ex-info ":moves must be a count" {:type ::invalid-moves :moves moves})))
  (when (and waste-free (or (not (pos-int? waste-free)) (pos? (mod n waste-free))))
    (throw (ex-info ":waste-free P must divide the number of particles"
                    {:type ::invalid-waste-free :n n :waste-free waste-free})))
  (fn [resolve reject]
    (let [seed (random/fresh-seed)
          ;; every particle's session, closed when inference ends however it
          ;; ends
          sessions (atom [])
          run! (fn []
                 (let [r (gfi/run-policy* model (itrace/policy {:temperature 0.0})
                                          (cond-> {} executor (assoc :executor executor)))]
                   (swap! sessions conj (:session r))
                   (:operation r)))
          close-all! (fn [k v]
                       (let [ss (distinct @sessions)
                             remaining (atom (count ss))]
                         (if (empty? ss)
                           (k v)
                           (doseq [s ss]
                             ((sp/close! s)
                              (fn [_] (when (zero? (swap! remaining dec)) (k v)))
                              (fn [_] (when (zero? (swap! remaining dec)) (k v))))))))
          fail! (fn [e] (close-all! reject e))
          stats (atom {:moves 0 :accepted 0})]
      (letfn [(step [ts ls lw beta log-z temperatures k]
                (if (>= beta 1.0)
                  (finish ts log-z temperatures)
                  (if (>= k max-steps)
                    (fail! (ex-info "Tempering did not reach β = 1" {:type ::too-many-steps
                                                                     :beta beta}))
                    (let [delta (next-delta lw ls beta ess-target)
                          beta' (min 1.0 (+ beta delta))
                          delta (- beta' beta)
                          inc-lw (mapv (fn [w l] (if (zero? delta) w (+ w (* delta l)))) lw ls)
                          log-z' (+ log-z (log-sum-exp inc-lw))
                          weights (m/normalize-log-weights inc-lw)
                          chains (if waste-free (quot n waste-free) n)
                          ancestors (random/with-stream* seed [::resample k]
                                      #(m/systematic-resample weights chains))
                          starts (mapv #(nth ts %) ancestors)
                          propose (adaptive-proposal (population-scales starts) scale)
                          bscales (block-scales starts scale)
                          steps-per-chain (if waste-free (dec waste-free) 1)]
                      ((all-settled (executor-of (first starts)) chains
                                    (fn [i res rej]
                                      ((move-chain (nth starts i) steps-per-chain moves beta' propose bscales [k i])
                                       res rej)))
                       (fn [results]
                         (try
                           (let [ts' (vec (mapcat (fn [start {:keys [states]}]
                                                    (if waste-free (into [start] states) states))
                                                  starts results))
                                 visited (into ts (mapcat :seen) results)]
                             (swap! stats (fn [s] (-> s
                                                      (update :moves + (reduce + (map :moves results)))
                                                      (update :accepted + (reduce + (map :accepted results))))))
                             (release-dropped! visited ts')
                             (step ts' (mapv log-likelihood ts')
                                   (vec (repeat (count ts') (- (Math/log (count ts')))))
                                   beta' log-z' (conj temperatures beta') (inc k)))
                           (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                       fail!)))))
              (finish [ts log-z temperatures]
                (let [particles (mapv (fn [t] [(m/sample-particle (:trace/result t) (itrace/legacy-trace t)) 0.0])
                                      ts)]
                  (close-all! resolve
                              (assoc (m/empirical particles)
                                     :log-normalizer log-z
                                     :temperatures temperatures
                                     :rejuvenation @stats))))]
        ;; One run first: a model that is one block needs no other run, its
        ;; particles are draws of the block until the end.
        ((run!)
         (fn [template]
           (try
             (if-let [error (:trace/error template)]
               (fail! (ex-info "A particle's program failed" {:type ::model-failed} error))
               (if-let [{:keys [address dist] :as blk} (single-block [template])]
                 (let [qs (into [(get-in template [:trace/entries address :value])]
                                (map (fn [i] (random/with-stream* seed [::init i] #(vec (dist/draw dist)))))
                                (range 1 n))
                       {:keys [qs log-z temperatures]}
                       (theta-tempered template blk qs
                                       {:ess-target ess-target :scale scale :waste-free waste-free
                                        :max-steps max-steps :n n :seed seed :stats stats})]
                   ((write-back-all template address qs)
                    (fn [final] (try (finish final log-z temperatures)
                                     (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                    fail!))
                 (let [operations (vec (repeatedly (dec n) run!))]
                   ((all-settled (executor-of template) (dec n) (fn [i res rej] ((nth operations i) res rej)))
                    (fn [more]
                      (try
                        (let [ts (into [template] more)]
                          (if-let [error (some :trace/error ts)]
                            (fail! (ex-info "A particle's program failed" {:type ::model-failed} error))
                            (step ts (mapv log-likelihood ts) (vec (repeat n (- (Math/log n))))
                                  0.0 0.0 [0.0] 0)))
                        (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                    fail!))))
             (catch #?(:clj Throwable :cljs :default) e (fail! e))))
         fail!)))))
