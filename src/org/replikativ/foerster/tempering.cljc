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
  (:require [org.replikativ.foerster.scope :as scope]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.block-mh :as block-mh]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.population :as pop]
            [org.replikativ.foerster.population.path :as path]
            [org.replikativ.foerster.execution :as execution]
            [org.replikativ.foerster.trace-owner :as trace-owner]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.protocols :as rtp]))

(defn- log-likelihood
  "Σ untempered log-likelihood of `t`'s observations and factors."
  [t]
  (transduce (keep (comp :log-lik :note)) + 0.0 (itrace/entries t)))

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
;; A model that is one block: particles are θ vectors (`foerster.population`)
;; -----------------------------------------------------------------------------

(defn- tempered-path
  "The geometric path q^(1-β)·t^β of a block law `dist`, from its draw density
  q to its target t, for `population/theta-smc`: β advances by the
  conditional-ESS rule, each particle's incremental log weight is δβ times
  its log t − log q, and the moves at β target the path's density."
  [dist ess-target]
  (let [rest-of (fn [q] (let [lq (dist/-draw-logpdf dist q)] (- (dist/logpdf dist q) lq)))]
    {:start 0.0
     :done? #(>= % 1.0)
     :advance (fn [qs lw beta]
                (let [ls (mapv rest-of qs)
                      beta' (min 1.0 (+ beta (pop/next-delta lw ls beta ess-target)))
                      delta (- beta' beta)]
                  {:pos beta' :log-incr (mapv (fn [l] (if (zero? delta) 0.0 (* delta l))) ls)}))
     :density #(block-mh/density dist %)
     :exhausted (fn [beta] (ex-info "Tempering did not reach β = 1" {:type ::too-many-steps :beta beta}))}))

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
    :seed        fixes every draw (default: one from the process generator)

  Returns a CPS operation resolving an EmpiricalMeasure of `Sample`s whose
  `m/log-marginal` estimates the evidence; `:temperatures` holds the
  schedule and `:rejuvenation` the moves made and accepted."
  [model n & [{:keys [ess-target moves scale waste-free max-steps executor seed]
               :or {ess-target 0.5 scale 2.38 max-steps 1000}}]]
  (when-not (and (number? ess-target) (< 0.0 ess-target 1.0))
    (throw (ex-info ":ess-target must lie in (0, 1)" {:type ::invalid-ess-target :ess-target ess-target})))
  (when-not (or (nil? moves) (nat-int? moves))
    (throw (ex-info ":moves must be a count" {:type ::invalid-moves :moves moves})))
  (when (and waste-free (or (not (pos-int? waste-free)) (pos? (mod n waste-free))))
    (throw (ex-info ":waste-free P must divide the number of particles"
                    {:type ::invalid-waste-free :n n :waste-free waste-free})))
  (scope/run
   (fn [resources]
     (fn [resolve reject]
       (let [seed (or seed (random/fresh-seed))
          ;; every particle's session, closed when inference ends however it
          ;; ends
             runs (atom -1)
             run! (fn []
                    (let [r (gfi/run-policy* model (itrace/policy {:temperature 0.0})
                                             (cond-> {:inference-scope resources :seed (sp/derive-seed seed ::run (swap! runs inc))}
                                               executor (assoc :executor executor)))]
                      (:operation r)))
             fail! reject
             stats (atom {:moves 0 :accepted 0})]
         (letfn [(run-traces [ts]
                   (let [target-path
                         {:start 0.0 :done? #(>= % 1.0)
                          :advance (fn [particles lw beta]
                                     (let [ls (mapv log-likelihood particles)
                                           delta (pop/next-delta lw ls beta ess-target)
                                           next-beta (min 1.0 (+ beta delta))
                                           delta (- next-beta beta)]
                                       {:pos next-beta
                                        :log-incr (mapv #(if (zero? delta) 0.0 (* delta %)) ls)}))
                          :exhausted (fn [beta] (ex-info "Tempering did not reach β = 1"
                                                         {:type ::too-many-steps :beta beta}))}
                         backend
                         {:move (fn [starts beta k steps]
                                  (let [propose (adaptive-proposal (population-scales starts) scale)
                                        bscales (block-scales starts scale)]
                                    (fn [resolve reject]
                                      ((execution/parallel (pop/executor-of (first starts)) (count starts)
                                                           (fn [i res rej]
                                                             ((move-chain (nth starts i) steps moves beta propose bscales [k i])
                                                              res rej)))
                                       (fn [results]
                                         (try
                                           (swap! stats (fn [st] (-> st
                                                                     (update :moves + (reduce + (map :moves results)))
                                                                     (update :accepted + (reduce + (map :accepted results))))))
                                           (resolve {:chains (mapv :states results) :seen (into [] (mapcat :seen) results)})
                                           (catch #?(:clj Throwable :cljs :default) e (reject e))))
                                       reject))))
                          :reclaim trace-owner/reclaim}]
                     ((path/run {:particles ts :mass (weights/uniform n) :position 0.0}
                                target-path backend {:n n :waste-free waste-free :seed seed
                                                     :keys {:resample ::resample} :max-steps max-steps})
                      (fn [result] (finish (:particles result) (get-in result [:mass :log-mass]) (:positions result)))
                      fail!)))
                 (finish [ts log-z temperatures]
                   (let [particles (mapv (fn [t] [(m/sample-particle (:trace/result t) (itrace/legacy-trace t)) 0.0])
                                         ts)]
                     (resolve
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
                  (if-let [{:keys [address dist]} (pop/single-block [template])]
                    (let [qs (into [(get-in template [:trace/entries address :value])]
                                   (map (fn [i] (random/with-stream* seed [::init i] #(vec (dist/draw dist)))))
                                   (range 1 n))
                          {:keys [qs log-z] temperatures :positions}
                          (pop/theta-smc qs (tempered-path dist ess-target)
                                         {:scale scale :waste-free waste-free :max-steps max-steps
                                          :n n :seed seed :stats stats
                                          :keys {:resample ::resample :move ::move}})]
                      ((pop/write-back-all template address qs)
                       (fn [final] (try (finish final log-z temperatures)
                                        (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                       fail!))
                    (let [operations (vec (repeatedly (dec n) run!))]
                      ((execution/parallel (pop/executor-of template) (dec n) (fn [i res rej] ((nth operations i) res rej)))
                       (fn [more]
                         (try
                           (let [ts (into [template] more)]
                             (if-let [error (some :trace/error ts)]
                               (fail! (ex-info "A particle's program failed" {:type ::model-failed} error))
                               (run-traces ts)))
                           (catch #?(:clj Throwable :cljs :default) e (fail! e))))
                       fail!))))
                (catch #?(:clj Throwable :cljs :default) e (fail! e))))
            fail!)))))))
