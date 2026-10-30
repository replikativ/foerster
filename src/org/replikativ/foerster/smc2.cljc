(ns org.replikativ.foerster.smc2
  "Static parameters of state-space programs: particle marginal
  Metropolis-Hastings (PMMH; Andrieu, Doucet & Holenstein 2010) and SMC²
  (Chopin, Jacob & Papaspiliopoulos 2013).

  A program's parameters are named sample sites (`:params`, a set of
  addresses). Run with those sites constrained to θ (`foerster.trace/policy
  :constraints`), SMC's evidence estimate is p(θ)·p̂(y | θ) — a constrained
  site adds its prior density to the weight — so the posterior ratio of two
  parameter values is the difference of two SMC log-evidences.

  - `pmmh` is a Markov chain over θ whose every proposal runs an SMC over
    the program's other sites: exact for any number of particles.
  - `smc2` is online: θ-particles each carry an inner streaming SMC
    (`foerster.smc/stream`); a pushed observation reweights each by its
    inner evidence increment; when the θ-particles' ESS falls they are
    resampled — a duplicate forks its inner population's worlds, so copies
    evolve independently — and moved by a PMMH step that runs a fresh inner
    filter on the data so far.

  Priors for θ come from the program itself: a θ-particle starts from a
  simulation of the program (`foerster.gfi/simulate`). Correlated PMMH, which
  needs SMC driven by explicit noise, is not provided."
  (:require [org.replikativ.foerster.execution :as execution]
            [org.replikativ.foerster.scope :as scope]
            [org.replikativ.foerster.cursor :as cursor]
            [org.replikativ.foerster.weights :as weights]
            [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.spindel.effects.savepoint :as sp]
            [org.replikativ.spindel.engine.protocols :as rtp]))

(defn- then [operation f]
  (fn [resolve reject]
    (operation (fn [x] (try (f x resolve reject)
                            (catch #?(:clj Throwable :cljs :default) e (reject e))))
               reject)))

(defn- prior-draw
  "θ from the program's own prior: a simulation's values at `params`."
  [model params opts & [parameter-prefix?]]
  (then (gfi/simulate model (select-keys opts [:executor :inference-scope]))
        (fn [t resolve reject]
          (let [choices (itrace/choices t)
                missing (remove #(contains? choices %) params)
                misplaced (when parameter-prefix?
                            (loop [[entry & more] (itrace/entries t) inner? false]
                              (when entry
                                (cond
                                  (and inner? (contains? params (:address entry))) (:address entry)
                                  :else (recur more (or inner? (and (itrace/latent? entry)
                                                                    (not (contains? params (:address entry))))))))))]
            ((gfi/close! t)
             (fn [_]
               (cond
                 (seq missing) (reject (ex-info "A parameter site was not reached" {:type ::missing-params
                                                                                    :params (vec missing)}))
                 misplaced (reject (ex-info "SMC² parameters must precede every inner latent site"
                                            {:type ::parameter-prior-not-prefix :address misplaced}))
                 :else (resolve (select-keys choices params))))
             reject)))))

(defn- invalid-parameters?
  "Whether `error` comes of a distribution refusing its parameters: a θ
  outside the program's support, which has density zero."
  [error]
  (some #(= ::dist/invalid-parameters (:type (ex-data %)))
        (take-while some? (iterate ex-cause error))))

(defn- or-impossible
  "`operation`, resolving nil when θ makes the program build an impossible
  distribution (a proposal with density zero, to be rejected)."
  [operation]
  (fn [resolve reject]
    (operation resolve (fn [e] (if (invalid-parameters? e) (resolve nil) (reject e))))))

(defn- parameter-prefix-policy
  "Check every conditioned execution, including proposal branches and forks.
  World-local trace entries follow forks; a shared callback atom would not."
  [policy params]
  (with-meta
    (fn [savepoint old-entry]
      (when (and (= itrace/choose-site (:savepoint/site savepoint))
                 (not (contains? params (:savepoint/address savepoint))))
        (let [entries (:trace/entries (rtp/get-state (:savepoint/world savepoint) [:savepoint/trace]))
              missing (remove #(= itrace/choose-site (:site (get entries %))) params)]
          (when (seq missing)
            (throw (ex-info "SMC² parameters must precede every inner sample site"
                            {:type ::parameter-prior-not-prefix
                             :address (:savepoint/address savepoint) :params (vec missing)})))))
      (policy savepoint old-entry))
    (meta policy)))

(defn- conditioned
  "SMC options holding θ. SMC² additionally guards its prior-prefix contract;
  ordinary PMMH keeps the full joint estimate and needs no prior cancellation."
  [theta opts & [prefix-params]]
  (let [base (some-> (:policy opts) itrace/policy-options)
        policy (itrace/policy (update base :constraints merge theta))]
    (assoc (dissoc opts :params :propose :scale :ess-target :moves :n-theta :n-x :burn)
           :policy (if prefix-params (parameter-prefix-policy policy prefix-params) policy))))

(defn- parameter-step!
  "Do not publish a stream barrier or completion that skipped a parameter."
  [step params]
  (doseq [[particle _] (m/get-particles (:measure step))]
    (let [t (m/get-trace particle)
          missing (remove #(contains? t %) params)]
      (when (seq missing)
        (throw (ex-info "SMC² parameters must precede the first inner stream barrier"
                        {:type ::parameter-prior-not-prefix :params (vec missing)})))))
  step)

(defn- checked-step [operation params]
  (then operation (fn [step resolve _] (resolve (parameter-step! step params)))))

(defn- log-prior
  "Σ log p of the parameter sites in a particle of `measure`: what the
  constraints added to the inner evidence."
  [measure params]
  (let [trace (m/get-trace (ffirst (m/get-particles measure)))]
    (reduce + 0.0 (map #(or (:log-prob (get trace %)) 0.0) params))))

(defn- random-walk
  "A symmetric Gaussian random walk on every parameter, with `scales`
  {address sd}. Draws from the current generator."
  [scales]
  (fn [theta]
    {:theta (into {} (map (fn [[a v]]
                            [a (+ v (* (get scales a 0.1) (dist/draw (dist/normal 0.0 1.0))))]))
                  theta)
     :log-ratio 0.0}))

(defn- population-scales
  "{address sd·scale} of the parameters over the θ-particles, for a random
  walk: `scale`/√d population standard deviations."
  [thetas scale]
  (let [d (count (first thetas))]
    (into {} (for [a (keys (first thetas))]
               (let [xs (map #(double (get % a)) thetas)
                     mu (/ (reduce + xs) (count xs))
                     sd (Math/sqrt (/ (reduce + (map #(let [e (- % mu)] (* e e)) xs)) (count xs)))]
                 [a (* (/ scale (Math/sqrt d)) (if (pos? sd) sd 1e-3))])))))

;; =============================================================================
;; PMMH
;; =============================================================================

(defn pmmh
  "Particle marginal Metropolis-Hastings over the parameter sites of `model`
  (a spin): `iterations` proposals, each scored by an SMC of `n` particles
  over the program's other sites. Options:

    :params   the parameter addresses (required)
    :propose  (fn [θ]) -> {:theta θ' :log-ratio log q(θ|θ') − log q(θ'|θ)},
              drawing from the current generator (default: a Gaussian
              random walk of `:scale`, default 0.1, on every parameter)
    :burn     iterations left out of the result (default 0)
    and SMC options (`:resample-threshold`, `:resampling`, `:executor` …)

  Resolves an EmpiricalMeasure, equally weighted, of one particle per kept
  iteration drawn from that iteration's SMC: its value the program's, its
  trace with θ and a state trajectory. `:acceptance` is the acceptance rate,
  `:thetas` the chain."
  [model n iterations {:keys [params propose scale burn] :or {scale 0.1 burn 0} :as opts}]
  (when-not (seq params)
    (throw (ex-info "pmmh needs :params" {:type ::no-params})))
  (when (or (= :fork (:world-policy opts)) (:world-opts opts))
    (throw (ex-info "Parameter filters currently use fresh inner worlds"
                    {:type ::unsupported-world-policy :supported #{:fresh}})))
  (fn [resolve reject]
    (let [seed (or (:seed opts) (random/fresh-seed))
          propose (or propose (random-walk (zipmap params (repeat scale))))
          ;; every run its own seed: a fixed one would make the chain target
          ;; the posterior of one noise draw
          run (fn [theta key]
                (or-impossible (smc/smc model n (assoc (conditioned theta opts)
                                                       :seed (sp/derive-seed seed ::inner key)))))
          pick (fn [measure k]
                 (let [ps (m/get-particles measure)
                       i (random/with-stream* seed [::pick k]
                           #(m/sample-categorical (m/normalize-log-weights (mapv second ps))))]
                   [(first (nth ps i)) 0.0]))]
      ((then (prior-draw model params opts)
             (fn [theta0 res rej]
               ((run theta0 -1)
                (fn [measure0]
                  (letfn [(step [i theta log-z sample accepted out]
                            (if (= i iterations)
                              (res (assoc (m/markov-measure out)
                                          :acceptance (/ (double accepted) (max 1 iterations))
                                          ;; after burn-in, as the measure
                                          :thetas (mapv (comp :theta meta first) out)))
                              (let [{theta' :theta lr :log-ratio}
                                    (random/with-stream* seed [::propose i] #(propose theta))]
                                ((run theta' i)
                                 (fn [measure']
                                   (let [log-z' (if measure' (m/log-marginal measure') ##-Inf)
                                         ratio (+ (- log-z' log-z) (or lr 0.0))
                                         u (random/with-stream* seed [::accept i] m/uniform01)
                                         accept? (and (not (#?(:clj Double/isNaN :cljs js/isNaN) ratio))
                                                      (< (Math/log u) ratio))
                                         [theta log-z sample] (if accept?
                                                                [theta' log-z' (pick measure' i)]
                                                                [theta log-z sample])
                                         kept (vary-meta (first sample) assoc :theta theta)]
                                     (step (inc i) theta log-z sample
                                           (if accept? (inc accepted) accepted)
                                           (if (>= i burn) (conj out [kept 0.0]) out))))
                                 rej))))]
                    (step 0 theta0 (m/log-marginal measure0) (pick measure0 -1) 0 [])))
                rej)))
       resolve reject))))

;; =============================================================================
;; SMC²
;; =============================================================================

(defn smc2
  "SMC² over the parameter sites of `model` (a spin whose data arrive at
  stream sites, as for `foerster.smc/stream`). Options:

    :params      the parameter addresses (required); all must be sampled
                 before every other sample/observation and the first stream
                 barrier, on every initial or proposed execution
    :n-theta     θ-particles (default 50)
    :n-x         particles of each inner filter (default 100)
    :ess-target  resample and move the θ-particles when their ESS falls below
                 this fraction (default 0.5)
    :moves       PMMH moves per θ-particle after a resampling (default 1)
    :scale       random-walk scale in population standard deviations
                 (default 2.38, divided by √d)
    and inner SMC options (`:resample-threshold`, `:resampling`, `:executor`)

  Resolves a step {:measure :push :done? :close} as `smc/stream` does: the
  measure is over the θ-particles (each a `Sample` whose value is θ, its
  trace empty), weighted, with the evidence of the data so far as
  `m/log-marginal`; `:moves` and `:accepted` count the PMMH steps."
  [model {:keys [params n-theta n-x ess-target moves scale]
          :or {n-theta 50 n-x 100 ess-target 0.5 moves 1 scale 2.38} :as opts}]
  (when-not (seq params)
    (throw (ex-info "smc2 needs :params" {:type ::no-params})))
  (when (or (= :fork (:world-policy opts)) (:world-opts opts))
    (throw (ex-info "Parameter filters currently use fresh inner worlds"
                    {:type ::unsupported-world-policy :supported #{:fresh}})))
  (when-let [option (some #(when (get opts %) %) [:anchors :retained :copy?])]
    (throw (ex-info "SMC² requires independently forkable active inner filters"
                    {:type ::nonforkable-inner-filter :option option})))
  (cursor/operation
   (fn [resolve reject]
     (let [resources (scope/create)
           seed (or (:seed opts) (random/fresh-seed))
          ;; each inner filter's seed comes from this run's and a fixed key:
          ;; they start inside concurrent callbacks
           inner (fn [theta key]
                   (checked-step
                    (smc/stream model n-x (assoc (conditioned theta opts params)
                                                 :inference-scope resources
                                                 :seed (sp/derive-seed seed ::inner key))) params))
          ;; a fresh inner filter for θ fed the data so far
           refit (fn [theta ys key]
                   (or-impossible
                    (fn [res rej]
                      ((inner theta key)
                       (fn [step0]
                         (letfn [(feed [step [y & more :as left]]
                                   (if (empty? left)
                                     (res step)
                                     ((checked-step ((:push step) y) params) #(feed % more) rej)))]
                           (feed step0 ys)))
                       rej))))
           stats (atom {:moves 0 :accepted 0})]
       (letfn [(fail [reject error]
                 ;; Sessions register before starting. Closing the owning scope
                 ;; cancels and joins all children, including concurrent starts.
                 ((scope/close! resources)
                  (fn [_] (reject error))
                  (fn [cleanup]
                    (reject (ex-info "SMC² failed and cleanup also failed"
                                     {:type ::scope/cleanup-failed :cleanup cleanup} error)))))
               (measure [ps lw log-z]
                 (assoc (m/empirical (mapv (fn [p w] [(m/sample-particle (:theta p) {}) w]) ps lw))
                        :log-normalizer log-z
                        :moves (:moves @stats) :accepted (:accepted @stats)))
               (step-of [ps lw log-z ys]
                 (let [done? (every? (comp :done? :step) ps)]
                   (cond-> {:measure (measure ps lw log-z)
                            :done? done?
                            :close (fn [] ((scope/close! resources) (fn [_]) (fn [_])))
                            :close-operation (scope/close! resources)}
                     (not done?) (assoc :push (fn [y] (push ps lw log-z (conj ys y) y))))))
               (push [ps lw log-z ys y]
                 (fn [res reject]
                   (let [rej #(fail reject %)]
                     ((execution/parallel nil (count ps) (fn [i r j]
                                                           (let [step (:step (nth ps i))]
                                               ;; a population that has finished
                                               ;; takes no more data
                                                             (if-let [push-step (:push step)]
                                                               ((push-step y) r j)
                                                               (r step)))))
                      (fn [steps]
                        (try
                          (doseq [step steps] (parameter-step! step params))
                          (let [incs (mapv (fn [p s]
                                             (let [z (m/log-marginal (:measure s))]
                                               (if (= ##-Inf z) ##-Inf (- z (:log-z p)))))
                                           ps steps)
                                ps (mapv (fn [p s] (assoc p :step s :log-z (m/log-marginal (:measure s)))) ps steps)
                                lw (mapv + lw incs)
                                mass (weights/from-log lw log-z)
                                selection (weights/normalize lw {:zero :uniform})]
                            (if (< (m/compute-ess selection) (* ess-target (count ps)))
                              ((rejuvenate ps selection (:log-mass mass) ys)
                               res rej)
                              (res (step-of ps lw log-z ys))))
                          (catch #?(:clj Throwable :cljs :default) e (rej e))))
                      rej))))
               (rejuvenate [ps weights log-z ys]
                ;; resample: a duplicate forks its inner population
                 (fn [res rej]
                   (let [k (count ys)
                         ancestors (random/with-stream* seed [::resample k]
                                     #(m/systematic-resample weights (count ps)))
                         firsts (atom #{})
                         pick (fn [_ a res' rej']
                                (let [p (nth ps a)]
                                  (if-not (contains? (first (swap-vals! firsts conj a)) a)
                                    (res' p)
                                    (cond
                                      (:done? (:step p)) (res' p)
                                      (:fork (:step p)) (((:fork (:step p))) #(res' (assoc p :step %)) rej')
                                      :else (rej' (ex-info "An active inner filter cannot be shared by outer particles"
                                                           {:type ::nonforkable-inner-filter}))))))]
                    ;; give back the populations no one descends from
                     (doseq [i (range (count ps)) :when (not (some #{i} ancestors))]
                       ((:close (:step (nth ps i)))))
                     ((execution/parallel nil (count ancestors) (fn [i r j] (pick i (nth ancestors i) r j)))
                      (fn [resampled]
                        (let [scales (population-scales (map :theta resampled) scale)
                              propose (random-walk scales)]
                          ((execution/parallel nil (count resampled)
                                               (fn [i r j] ((move (nth resampled i) propose ys [k i] moves) r j)))
                           (fn [moved]
                             (res (step-of moved (vec (repeat (count moved) 0.0)) log-z ys)))
                           rej)))
                      rej))))
               (move [p propose ys key n]
                ;; n PMMH steps of one θ-particle on the data so far
                 (fn [res rej]
                   (if (zero? n)
                     (res p)
                     (let [{theta' :theta lr :log-ratio}
                           (random/with-stream* seed [::propose key n] #(propose (:theta p)))]
                       ((refit theta' ys [key n])
                        (fn [step']
                          (let [log-z' (if step' (m/log-marginal (:measure step')) ##-Inf)
                                ratio (+ (- log-z' (:log-z p)) (or lr 0.0))
                                u (random/with-stream* seed [::accept key n] m/uniform01)
                                accept? (and (not (#?(:clj Double/isNaN :cljs js/isNaN) ratio))
                                             (< (Math/log u) ratio))]
                            (swap! stats #(-> % (update :moves inc)
                                              (update :accepted + (if accept? 1 0))))
                            (if accept?
                              (do ((:close (:step p)))
                                  ((move {:theta theta' :step step' :log-z log-z'} propose ys key (dec n)) res rej))
                              (do (when step' ((:close step')))
                                  ((move p propose ys key (dec n)) res rej)))))
                        rej)))))]
         ((execution/parallel nil n-theta
                              (fn [i res rej]
                                ((then (prior-draw model params (assoc opts :inference-scope resources) true)
                                       (fn [theta r j]
                                         ((inner theta [::init i])
                                          (fn [step]
                                            (let [p {:theta theta :step step
                                                     :log-z (m/log-marginal (:measure step))}]
                                              (r p)))
                                          j)))
                                 res rej)))
          (fn [ps]
           ;; a θ-particle starts weighted by what the program observed before
           ;; its first stream site: its inner evidence without the prior
           ;; it was drawn from
            (resolve (step-of ps (mapv (fn [p] (- (:log-z p) (log-prior (:measure (:step p)) params))) ps)
                              0.0 [])))
          (fn [e] (fail reject e))))))))
