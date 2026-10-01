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
  (:require [org.replikativ.foerster.gfi :as gfi]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.foerster.smc :as smc]
            [org.replikativ.foerster.trace :as itrace]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.spindel.effects.savepoint :as sp]))

(defn- then [operation f]
  (fn [resolve reject]
    (operation (fn [x] (try (f x resolve reject)
                            (catch #?(:clj Throwable :cljs :default) e (reject e))))
               reject)))

(defn- all-settled
  "Run `(start i resolve reject)` for i < n together; resolves the results in
  order, or rejects with the first failure."
  [n start]
  (fn [resolve reject]
    (if (zero? n)
      (resolve [])
      (let [out (object-array n)
            remaining (atom n)
            failed? (atom false)]
        (dotimes [i n]
          (start i
                 (fn [v]
                   (aset out i v)
                   (when (zero? (swap! remaining dec))
                     (resolve (vec out))))
                 (fn [e]
                   (when (compare-and-set! failed? false true)
                     (reject e)))))))))

(defn- prior-draw
  "θ from the program's own prior: a simulation's values at `params`."
  [model params opts]
  (then (gfi/simulate model (select-keys opts [:executor]))
        (fn [t resolve reject]
          (let [choices (itrace/choices t)
                missing (remove #(contains? choices %) params)]
            ((gfi/close! t)
             (fn [_]
               (if (seq missing)
                 (reject (ex-info "A parameter site was not reached" {:type ::missing-params
                                                                      :params (vec missing)}))
                 (resolve (select-keys choices params))))
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

(defn- conditioned
  "SMC options that hold the parameter sites at θ, over the caller's policy."
  [theta opts]
  (let [base (some-> (:policy opts) itrace/policy-options)]
    (assoc (dissoc opts :params :propose :scale :ess-target :moves :n-theta :n-x :burn)
           :policy (itrace/policy (update base :constraints merge theta)))))

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
  (fn [resolve reject]
    (let [seed (random/fresh-seed)
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
                              (res (assoc (m/empirical out)
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

(defn- log-sum-exp [xs]
  (let [top (reduce max ##-Inf xs)]
    (if (= ##-Inf top) ##-Inf (+ top (Math/log (reduce + (map #(Math/exp (- % top)) xs)))))))

(defn smc2
  "SMC² over the parameter sites of `model` (a spin whose data arrive at
  stream sites, as for `foerster.smc/stream`). Options:

    :params      the parameter addresses (required)
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
  (fn [resolve reject]
    (let [seed (random/fresh-seed)
          ;; each inner filter's seed comes from this run's and a fixed key:
          ;; they start inside concurrent callbacks
          inner (fn [theta key]
                  (smc/stream model n-x (assoc (conditioned theta opts)
                                               :seed (sp/derive-seed seed ::inner key))))
          ;; a fresh inner filter for θ fed the data so far
          refit (fn [theta ys key]
                  (or-impossible
                   (fn [res rej]
                     ((inner theta key)
                      (fn [step0]
                        (letfn [(feed [step [y & more :as left]]
                                  (if (empty? left)
                                    (res step)
                                    (((:push step) y) #(feed % more) rej)))]
                          (feed step0 ys)))
                      rej))))
          close-all! (fn [ps] (doseq [p ps] ((:close (:step p)))))
          started (atom [])
          stats (atom {:moves 0 :accepted 0})]
      (letfn [(measure [ps lw log-z]
                (assoc (m/empirical (mapv (fn [p w] [(m/sample-particle (:theta p) {}) w]) ps lw))
                       :log-normalizer log-z
                       :moves (:moves @stats) :accepted (:accepted @stats)))
              (step-of [ps lw log-z ys]
                (let [done? (every? (comp :done? :step) ps)]
                  (cond-> {:measure (measure ps lw log-z)
                           :done? done?
                           :close (fn [] (close-all! ps))}
                    (not done?) (assoc :push (fn [y] (push ps lw log-z (conj ys y) y))))))
              (push [ps lw log-z ys y]
                (fn [res rej]
                  ((all-settled (count ps) (fn [i r j]
                                             (let [step (:step (nth ps i))]
                                               ;; a population that has finished
                                               ;; takes no more data
                                               (if-let [push (:push step)]
                                                 ((push y) r j)
                                                 (r step)))))
                   (fn [steps]
                     (try
                       (let [incs (mapv (fn [p s]
                                          (let [z (m/log-marginal (:measure s))]
                                            (if (= ##-Inf z) ##-Inf (- z (:log-z p)))))
                                        ps steps)
                             ps (mapv (fn [p s] (assoc p :step s :log-z (m/log-marginal (:measure s)))) ps steps)
                             lw (mapv + lw incs)
                             weights (m/normalize-log-weights lw)]
                         (if (< (m/compute-ess weights) (* ess-target (count ps)))
                           ((rejuvenate ps weights (+ log-z (- (log-sum-exp lw) (Math/log (count ps)))) ys)
                            res rej)
                           (res (step-of ps lw log-z ys))))
                       (catch #?(:clj Throwable :cljs :default) e (rej e))))
                   (fn [e] (close-all! ps) (rej e)))))
              (rejuvenate [ps weights log-z ys]
                ;; resample: a duplicate forks its inner population
                (fn [res rej]
                  (let [k (count ys)
                        ancestors (random/with-stream* seed [::resample k]
                                    #(m/systematic-resample weights (count ps)))
                        firsts (atom #{})
                        pick (fn [i a res' rej']
                               (let [p (nth ps a)]
                                 (if (and (contains? (first (swap-vals! firsts conj a)) a)
                                          ;; a finished population takes no more
                                          ;; data: duplicates may share it
                                          (:fork (:step p)))
                                   (((:fork (:step p))) #(res' (assoc p :step %)) rej')
                                   (res' p))))]
                    ;; give back the populations no one descends from
                    (doseq [i (range (count ps)) :when (not (some #{i} ancestors))]
                      ((:close (:step (nth ps i)))))
                    ((all-settled (count ancestors) (fn [i r j] (pick i (nth ancestors i) r j)))
                     (fn [resampled]
                       (let [scales (population-scales (map :theta resampled) scale)
                             propose (random-walk scales)]
                         ((all-settled (count resampled)
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
        ((all-settled n-theta
                      (fn [i res rej]
                        ((then (prior-draw model params opts)
                               (fn [theta r j]
                                 ((inner theta [::init i])
                                  (fn [step]
                                    (let [p {:theta theta :step step
                                             :log-z (m/log-marginal (:measure step))}]
                                      (swap! started conj p)
                                      (r p)))
                                  j)))
                         res rej)))
         (fn [ps]
           ;; a θ-particle starts weighted by what the program observed before
           ;; its first stream site: its inner evidence without the prior
           ;; it was drawn from
           (resolve (step-of ps (mapv (fn [p] (- (:log-z p) (log-prior (:measure (:step p)) params))) ps)
                             0.0 [])))
         (fn [e] (close-all! @started) (reject e)))))))
