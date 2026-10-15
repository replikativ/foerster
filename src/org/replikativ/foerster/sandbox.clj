(ns org.replikativ.foerster.sandbox
  "foerster for code that runs in an interpreter sandbox (dvergr's SCI
  sandbox): what such code may call, as namespace maps a host installs, and
  the plain data inference results become before they reach it.

  `(namespaces opts)` gives {ns-sym {sym value}} and `docs` the arglists and
  docstrings for each. The host owns everything an interpreter cannot: it
  makes `sample`/`observe` effect breakpoints, installs these maps, decides
  the world policy and the limits, and owns the lifecycle of started jobs.

    infer/  run inference (every `foerster.core/infer` method), as a spin or
            as a job; posterior accessors and summaries
    model/  numerical blocks: a log density over θ as one choice site
    opt/    MAP, Laplace, Pathfinder on a block site
    diag/   R-hat, ESS, MCSE and summaries of a posterior's chains
    lp/     log densities summed over data vectors, computed on the host

  No result carries an execution context, a world handle or an executor:
  `posterior` projects a measure to values, weights and its chain
  structure. Seeds are per run (`:seed`); the process generator is not
  exposed."
  (:require [org.replikativ.foerster.block :as block]
            [org.replikativ.foerster.core :as core]
            [org.replikativ.foerster.diagnostics :as diag]
            [org.replikativ.foerster.dist :as dist]
            [org.replikativ.foerster.measure :as m]
            [org.replikativ.foerster.optimize :as opt]
            [org.replikativ.foerster.random :as random]
            [org.replikativ.spindel.effects.savepoint :as savepoint]
            [org.replikativ.spindel.engine.core :as ec]
            [org.replikativ.spindel.effects.await :refer [await]]
            [org.replikativ.spindel.spin.core :as spin-core]
            [org.replikativ.spindel.spin.cps :refer [spin]]
            [org.replikativ.spindel.spin.sync :as sync]
            [clojure.string :as str]
            [clojure.walk :as walk]))

;; =============================================================================
;; Posteriors as data
;; =============================================================================

(defn data?
  "Whether `x` is plain data: nil, numbers, strings, keywords, symbols,
  booleans, characters, and vectors, lists, sets and maps of them."
  [x]
  (cond (or (nil? x) (number? x) (string? x) (keyword? x) (symbol? x) (boolean? x) (char? x)) true
        (map? x) (and (not (record? x)) (every? (fn [[k v]] (and (data? k) (data? v))) x))
        (or (vector? x) (set? x) (list? x) (seq? x)) (every? data? x)
        :else false))

(defn- without-meta
  "`x` with no metadata anywhere: metadata could carry what `data?` does
  not look at."
  [x]
  (walk/postwalk #(if (instance? clojure.lang.IObj %) (with-meta % nil) %) x))

(defn posterior
  "The plain-data projection of `measure`: program values, log weights and
  normalized weights, ESS, log marginal, canonical world descriptors, and —
  for Markov chains — the chain lengths, in draw order. Refuses a measure
  whose particles all have zero weight, and program values that are not
  plain data (`data?`): nothing else crosses into the sandbox."
  [measure]
  (let [log-weights (vec (m/get-log-weights measure))]
    (when (every? #(= ##-Inf %) log-weights)
      (throw (ex-info "Every particle has zero likelihood: no posterior" {:type ::zero-likelihood})))
    (cond-> {:posterior/values (mapv m/get-value (m/get-contexts measure))
             :posterior/log-weights log-weights
             :posterior/weights (vec (m/normalize-log-weights log-weights))
             :posterior/ess (m/effective-sample-size measure)
             :posterior/log-marginal (m/log-marginal measure)
             :posterior/worlds (m/world-descriptors measure)}
      (:chain-lengths measure) (assoc :posterior/chain-lengths (vec (:chain-lengths measure)))
      true (as-> post
                 (if (data? post)
                   (without-meta post)
                   (throw (ex-info "Inference results must be plain data: return numbers, vectors and maps from the model"
                                   {:type ::not-data})))))))

(defn- value-fn [f]
  (cond (nil? f) identity
        (keyword? f) f
        (integer? f) #(nth % f)
        :else f))

(defn chains
  "Draws of `f` by chain: one chain when the posterior has no chain structure."
  [post f]
  (let [g (value-fn f)
        xs (mapv (comp double g) (:posterior/values post))]
    (loop [xs xs [n & more] (or (:posterior/chain-lengths post) [(count xs)]) out []]
      (if n (recur (subvec xs n) more (conj out (subvec xs 0 n))) out))))

(defn summary
  "Weighted mean, sd and 5/25/50/75/95% quantiles of the quantity `f` of the
  posterior's values (a function, a keyword, an index, or nil for the value
  itself), and for Markov chains R-hat, bulk and tail ESS and MCSE."
  [post & [f]]
  (let [g (value-fn f)
        xs (mapv (comp double g) (:posterior/values post))
        ws (:posterior/weights post)
        mean (reduce + (map * ws xs))
        sd (Math/sqrt (reduce + (map (fn [w x] (* w (- x mean) (- x mean))) ws xs)))
        sorted (sort-by first (map vector xs ws))
        quantile (fn [p] (loop [[[x w] & more] sorted acc 0.0]
                           (let [acc (+ acc w)] (if (or (>= acc p) (empty? more)) x (recur more acc)))))
        base {:mean mean :sd sd
              :quantiles (into (sorted-map) (map (fn [p] [p (quantile p)])) [0.05 0.25 0.5 0.75 0.95])
              :draws (count xs)}]
    (if (:posterior/chain-lengths post)
      (let [cs (chains post f)]
        (assoc base :chains (count cs) :rhat (diag/rhat cs) :ess-bulk (diag/ess-bulk cs)
               :ess-tail (diag/ess-tail cs) :mcse (diag/mcse cs)))
      (assoc base :ess (:posterior/ess post)))))

(defn query
  "`f` of every value with its weight: [[x w] …]."
  [post f]
  (mapv vector (map f (:posterior/values post)) (:posterior/weights post)))

;; =============================================================================
;; Running inference
;; =============================================================================

(def ^:private method-work
  "The methods a sandbox may run, and what a run costs: draws × particle or
  chain executions, from its sizes with foerster's defaults."
  {:importance (fn [{:keys [particles]}] particles)
   :smc (fn [{:keys [particles]}] particles)
   :mh (fn [{:keys [iterations chains] :or {chains 4}}] (* iterations chains))
   :rmh (fn [{:keys [iterations chains] :or {chains 4}}] (* iterations chains))
   :nuts (fn [{:keys [iterations chains] :or {chains 4}}] (* iterations chains))
   ;; a sweep of :particles per iteration, and an initial one
   :pimh (fn [{:keys [particles iterations]}] (* particles (inc iterations)))
   :pgibbs (fn [{:keys [particles iterations]}] (* particles (inc iterations)))
   :pgas (fn [{:keys [particles iterations]}] (* particles (inc iterations)))
   ;; on each of its 8 nodes
   :ipmcmc (fn [{:keys [particles iterations]}] (* 8 particles (inc iterations)))})
;; Not yet: tempering, parallel tempering and enumeration have costs their
;; sizes do not bound.

(def ^:private particle-methods
  "The methods whose particles can run in canonical forks (`:world-policy`)."
  #{:importance :smc :pimh :pgibbs :pgas :ipmcmc})

(def ^:private run-options
  "Options a sandbox may pass: data only, no functions, kernels or worlds."
  #{:method :particles :iterations :chains :burn :seed :world-policy :resampling
    :resample-threshold :step-size :target-accept :max-depth :metric})

(defn- refuse [message data] (throw (ex-info message data)))

(defn- seeder
  "Per-session seeds: the run's own `:seed`, else the next of a sequence
  derived from the host's `:seed`, never the process generator."
  [{:keys [seed]}]
  (let [n (atom -1)
        base (or seed (System/nanoTime))]
    (fn [given] (or given (savepoint/derive-seed base ::run (swap! n inc))))))

(defn- checked
  "`opts` checked against the sandbox's methods, options and `:max-work`,
  with the host's world policy as default and a seed."
  [opts {:keys [world-policy max-work]} next-seed]
  (let [{:keys [method]} opts
        work-of (method-work method)]
    (when-not work-of
      (refuse (str "Method " method " is not available here; one of " (vec (sort (keys method-work))))
              {:type ::method :method method}))
    (when-let [other (seq (remove run-options (keys opts)))]
      (refuse (str "Options not available here: " (vec other)) {:type ::options :options (vec other)}))
    (when-not (data? opts)
      (refuse "Inference options must be plain data" {:type ::options}))
    (doseq [[k least] {:particles 1 :iterations 1 :chains 1 :burn 0}
            :when (contains? opts k)]
      (when-not (and (integer? (get opts k)) (>= (get opts k) least))
        (refuse (str k " must be an integer of at least " least) {:type ::sizes :option k})))
    (let [work (try (work-of opts) (catch Exception _ nil))]
      (when-not (and (number? work) (pos? work))
        (refuse (str "Sizes missing for " method ": :particles, or :iterations (and :chains)")
                {:type ::sizes :method method}))
      (when (and max-work (> work max-work))
        (refuse (str "Inference larger than this sandbox allows: " work " > " max-work)
                {:type ::too-large :work work :max-work max-work})))
    (cond-> (assoc opts :seed (next-seed (:seed opts)))
      ;; Markov chains run in fresh worlds of their own (foerster.core)
      (and world-policy (particle-methods method) (not (contains? opts :world-policy)))
      (assoc :world-policy world-policy))))

(defn- sanitized
  "`e` without its data or causes, which may hold host handles: its message
  and those of its causes, and its :type."
  [e]
  (let [messages (loop [e e acc []] (if e (recur (ex-cause e) (conj acc (ex-message e))) acc))]
    (ex-info (str/join " — caused by: " (remove nil? messages))
             (select-keys (ex-data e) [:type]))))

(defn- run [model opts host next-seed]
  (let [native (try (core/infer model (checked opts host next-seed))
                    (catch Exception e (throw (sanitized e))))]
    (spin (try (posterior (await native))
               (catch Exception e (throw (sanitized e)))))))

(defrecord Job [spin context terminal done])

(defn- finish!
  "The job's one terminal transition; false if it already had one. A
  finished job keeps only its outcome in `jobs`, not its spin or context."
  [^Job job jobs id outcome]
  (when (compare-and-set! (:terminal job) nil outcome)
    (deliver (:done job) outcome)
    (swap! jobs (fn [js] (if (contains? js id) (assoc js id (select-keys job [:terminal :done])) js)))
    true))

(defn- start
  "Start a job; the sandbox gets {:job/id uuid}, the job stays in `jobs`
  until forgotten."
  [model opts {:keys [on-start] :as host} next-seed jobs]
  (let [s (run model opts host next-seed)
        job (->Job s ec/*execution-context* (atom nil) (promise))
        id (random-uuid)]
    (swap! jobs assoc id job)
    (sync/spawn! s {:on-success #(finish! job jobs id {:status :done :value %})
                    :on-error #(finish! job jobs id {:status :failed :error (ex-message %)})})
    (when on-start (on-start job))
    {:job/id id}))

(defn- job-of [jobs handle]
  (or (get @jobs (:job/id handle))
      (refuse "Not a job of this sandbox" {:type ::no-job})))

(defn- status [^Job job]
  (select-keys (or @(:terminal job) {:status :running}) [:status :error]))

(defn- result
  "The job's posterior, waiting at most `timeout-ms`; :pending when it is
  not done by then. A failed or cancelled job throws."
  [^Job job timeout-ms]
  (let [r (deref (:done job) timeout-ms ::pending)]
    (case (if (= ::pending r) ::pending (:status r))
      ::pending :pending
      :done (:value r)
      :failed (throw (ex-info (str "Inference failed: " (:error r)) {:type ::failed}))
      :cancelled (throw (ex-info "Inference was cancelled" {:type ::cancelled})))))

(defn- cancel
  "Stop the job, in the context it was started in; a finished job stays
  as it finished."
  [job jobs id]
  (let [s (:spin job) context (:context job)]
    (when (finish! job jobs id {:status :cancelled})
      (binding [ec/*execution-context* context]
        (spin-core/cancel-spin! s))))
  nil)

(defn cancel-job!
  "Cancel `job`, as `:on-start` received it (for the host: on a timeout or
  when its world ends)."
  [job]
  (cancel job (atom {}) nil))

;; =============================================================================
;; Vectorized log densities
;; =============================================================================

(def ^:private half-log-2pi 0.9189385332046727)

(defn- dot ^double [beta x]
  (loop [k 0 acc 0.0] (if (< k (count beta)) (recur (inc k) (+ acc (* (double (nth beta k)) (double (nth x k))))) acc)))

(defn- softplus ^double [^double t] (if (> t 0.0) (+ t (Math/log1p (Math/exp (- t)))) (Math/log1p (Math/exp t))))

(defn- sum-over ^double [n f]
  (loop [i 0 acc 0.0] (if (< i n) (recur (inc i) (+ acc (double (f i)))) acc)))

(defn normal
  "Σᵢ log N(yᵢ | μᵢ, σ): `mu` a number or a vector like `ys`."
  [ys mu sigma]
  (let [sigma (double sigma) at (if (number? mu) (constantly mu) #(nth mu %))]
    (if (pos? sigma)
      (- (sum-over (count ys) #(let [z (/ (- (double (nth ys %)) (double (at %))) sigma)] (* -0.5 z z)))
         (* (count ys) (+ (Math/log sigma) half-log-2pi)))
      ##-Inf)))

(defn normal-glm
  "Σᵢ log N(yᵢ | α + xᵢ·β, σ), `xs` a vector of rows."
  [ys xs alpha beta sigma]
  (normal ys (mapv #(+ (double alpha) (dot beta %)) xs) sigma))

(defn bernoulli-logit-glm
  "Σᵢ log Bernoulli(yᵢ | logit⁻¹(α + xᵢ·β)), yᵢ in {0, 1}."
  [ys xs alpha beta]
  (if (every? #(or (== 0 %) (== 1 %)) ys)
    (sum-over (count ys) #(let [eta (+ (double alpha) (dot beta (nth xs %)))]
                            (- (* (double (nth ys %)) eta) (softplus eta))))
    ##-Inf))

(defn poisson-log-glm
  "Σᵢ log Poisson(yᵢ | exp(α + xᵢ·β)), yᵢ counts."
  [ys xs alpha beta]
  (if (every? #(and (>= % 0) (== % (Math/floor %))) ys)
    (sum-over (count ys) #(let [eta (+ (double alpha) (dot beta (nth xs %))) y (double (nth ys %))]
                            (- (* y eta) (Math/exp eta) (dist/lgamma (inc y)))))
    ##-Inf))

;; =============================================================================
;; The namespaces
;; =============================================================================

(defn- arrays->vectors [x]
  (cond (instance? (Class/forName "[D") x) (vec x)
        (map? x) (into {} (map (fn [[k v]] [k (arrays->vectors v)])) x)
        (vector? x) (mapv arrays->vectors x)
        :else x))

(def ^:private optimizer-limits
  {:init nil :jacobian? nil :tolerance nil :h nil :seed nil
   :max-iterations 10000 :history 50 :paths 16 :draws 10000 :elbo-draws 500})

(defn- optimizer-work
  "An upper bound on the target evaluations of `f` under `opts`, in L-BFGS
  iterations: Pathfinder's paths × iterations × (ELBO draws + 1) plus its
  draws; MAP and Laplace's iterations plus their draws."
  [f {:keys [paths max-iterations elbo-draws draws] :or {paths 4 max-iterations 1000 elbo-draws 30 draws 0}}]
  (if (= f opt/pathfinder)
    (+ (* paths max-iterations (inc elbo-draws)) (if (zero? draws) 1000 draws))
    (+ max-iterations draws)))

(defn- optimizer
  "`f` of foerster.optimize on a block site, for a sandbox: options checked
  against `optimizer-limits` and the host's `:max-optimizer-work`, its draws
  from a seeded stream, the result as plain data (`:draws` draws for
  Laplace; no closures)."
  [f next-seed {:keys [max-optimizer-work]}]
  (fn [site & [opts]]
    (let [opts (or opts {})]
      (when-let [other (seq (remove (set (keys optimizer-limits)) (keys opts)))]
        (refuse (str "Options not available here: " (vec other)) {:type ::options :options (vec other)}))
      (doseq [[k limit] optimizer-limits
              :when (and limit (contains? opts k))]
        (when-not (and (integer? (get opts k)) (<= 0 (get opts k) limit))
          (refuse (str k " must be an integer from 0 to " limit) {:type ::options :option k})))
      (let [work (optimizer-work f opts)]
        (when (and max-optimizer-work (> work max-optimizer-work))
          (refuse (str "Optimization larger than this sandbox allows: " work " > " max-optimizer-work)
                  {:type ::too-large :work work :max-work max-optimizer-work})))
      (let [seed (next-seed (:seed opts))
            r (try (random/with-stream* seed ::optimize
                     #(let [r (f site (cond-> (dissoc opts :seed) (not= f opt/pathfinder) (dissoc :draws)))]
                        (cond-> r
                          (and (:draw r) (= f opt/laplace))
                          (assoc :draws (vec (repeatedly (get opts :draws 0) (:draw r)))))))
                   (catch Exception e (throw (sanitized e))))
            r (arrays->vectors (dissoc r :draw))]
        (if (data? r) (without-meta r) (refuse "Optimizer result is not plain data" {:type ::not-data}))))))

(defn namespaces
  "{ns-sym {sym value}} for a sandbox. Host options: `:world-policy` (the
  default for particle runs that do not choose one; Markov chains always
  run in fresh worlds), `:max-work` (the largest
  particles or iterations × chains a run may ask for), `:max-optimizer-work`
  (the same for optimizers, in target evaluations, `optimizer-work`),
  `:seed` (the
  session's: runs and optimizers without their own `:seed` draw theirs from
  it, never from the process generator), `:on-start` (called with every
  started job: the host cancels what it still runs when its world ends,
  `cancel-job!`)."
  [host]
  (let [next-seed (seeder host)
        jobs (atom {})]
    {'infer {'run (fn [model opts] (run model opts host next-seed))
             'start (fn [model opts] (start model opts host next-seed jobs))
             'status #(status (job-of jobs %))
             'result #(result (job-of jobs %1) %2)
             'cancel #(cancel (job-of jobs %) jobs (:job/id %))
             'forget #(do (swap! jobs dissoc (:job/id %)) nil)
             'summary summary
             'query query
             'values :posterior/values
             'weights :posterior/weights
             'log-weights :posterior/log-weights
             'ess :posterior/ess
             'log-marginal :posterior/log-marginal}
     'model {'block block/block
             'site block/block-dist
             'constrain (fn [b theta] (vec (block/constrain b theta)))
             'numeric-gradient block/with-numeric-gradient}
     'opt {'map-estimate (optimizer opt/map-estimate next-seed host)
           'laplace (optimizer opt/laplace next-seed host)
           'pathfinder (optimizer opt/pathfinder next-seed host)}
     'diag {'rhat diag/rhat 'ess-bulk diag/ess-bulk 'ess-tail diag/ess-tail 'mcse diag/mcse
            'chains chains 'summary summary}
     'lp {'normal normal 'normal-glm normal-glm 'bernoulli-logit-glm bernoulli-logit-glm
          'poisson-log-glm poisson-log-glm}}))

(def docs
  "{ns-sym {sym [arglists docstring]}} for `namespaces`."
  '{infer {run [([model opts]) "Run inference on `model` (a spin with sample/observe sites); a spin of the posterior as data. opts: :method (:importance :smc :mh :rmh :nuts; particle MCMC :pimh :pgibbs :pgas :ipmcmc), its sizes (:particles; or :iterations :chains :burn; particle MCMC :particles per sweep and :iterations), :seed, :world-policy, :resampling, :resample-threshold, :step-size (:rmh), :target-accept :max-depth :metric (:nuts). NUTS needs a block site (model/site). Values the model returns must be plain data."]
           start [([model opts]) "As `run`, started in the background: a job handle {:job/id …} for status, result and cancel. Use it for runs longer than one evaluation may take."]
           status [([job]) "{:status :running|:done|:failed|:cancelled}"]
           result [([job timeout-ms]) "The job's posterior, waiting at most timeout-ms; :pending if not done yet."]
           cancel [([job]) "Stop the job."]
           forget [([job]) "Drop a job and its result from this sandbox."]
           summary [([posterior] [posterior f]) "Weighted mean, sd, quantiles of f (fn, keyword, index or nil) and, for Markov chains, :rhat :ess-bulk :ess-tail :mcse."]
           query [([posterior f]) "[[f(value) weight] …]"]
           values [([posterior]) "Each draw's program value, in draw order."]
           weights [([posterior]) "Normalized weights."]
           log-weights [([posterior]) "Log weights."]
           ess [([posterior]) "Effective sample size of the weights."]
           log-marginal [([posterior]) "Log marginal likelihood estimate (particle methods)."]}
    model {block [([description capabilities]) "A block: fixed-shape latents {:block/id :block/latents [{:name :shape :support}] :block/target :complete-conditional} with capabilities {:log-density (fn [theta inputs]) :value+grad …}; :block/coordinates :constrained for :positive or [:interval a b] latents."]
           site [([block inputs]) "The block at its inputs as a distribution: (sample (model/site b data) :id :theta :init [...])."]
           constrain [([block theta]) "θ (unconstrained) to the latents' own coordinates."]
           numeric-gradient [([capabilities]) "Capabilities with :value+grad by finite differences of :log-density."]}
    opt {map-estimate [([site] [site opts]) "The mode of a block site. opts :init."]
         laplace [([site] [site opts]) "Laplace approximation: :theta :covariance :log-evidence, and :draws approximate posterior draws of the latents when opts has :draws n."]
         pathfinder [([site] [site opts]) "Pathfinder: :mean :covariance :draws :pareto-k (above 0.7: use it to start a sampler only)."]}
    diag {rhat [([chains]) "Rank-normalized split R-hat of draws by chain."]
          ess-bulk [([chains]) "Bulk effective sample size."]
          ess-tail [([chains]) "Tail effective sample size."]
          mcse [([chains]) "Monte Carlo standard error of the mean."]
          chains [([posterior f]) "Draws of f by chain."]
          summary [([posterior] [posterior f]) "As infer/summary."]}
    lp {normal [([ys mu sigma]) "Σ log N(y | μ, σ); mu a number or a vector."]
        normal-glm [([ys xs alpha beta sigma]) "Σ log N(y | α + x·β, σ); xs rows."]
        bernoulli-logit-glm [([ys xs alpha beta]) "Σ log Bernoulli(y | logit⁻¹(α + x·β))."]
        poisson-log-glm [([ys xs alpha beta]) "Σ log Poisson(y | exp(α + x·β))."]}})
